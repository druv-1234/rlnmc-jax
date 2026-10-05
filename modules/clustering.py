import sys
sys.path.append('.')

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from typing import Callable, List

from modules.nmc_types import *


def _grow_to_fixed_point(one_step_expansion: Callable, cl_seed: jax.Array) -> jax.Array:
	"""
	Iterate a monotone one-hop expansion until it stops changing.

	The carry is (previous, current) rather than just the mask so that each while_loop
	iteration evaluates the expansion exactly ONCE - writing the condition as
	`one_step(cl) != cl` would evaluate it twice per hop, doubling the work.

	The initial `previous` is the complement of cl_seed, which differs from cl_seed for any
	N > 0, so the loop always executes at least one hop. Under vmap the loop runs until
	every batch element has converged; the extra iterations are no-ops on the ones that
	already have, since the expansion is monotone and idempotent at its fixed point.
	"""

	def cond(carry):
		previous, current = carry
		return jnp.any(previous != current)

	def body(carry):
		_, current = carry
		return current, one_step_expansion(current)

	_, final_clst = jax.lax.while_loop(cond, body, (jnp.logical_not(cl_seed), cl_seed))

	return final_clst


def _check_weight_length(J_indices: List[jax.Array], weight: jax.Array, offsets: tuple):
	n_factors = sum(int(idx.shape[0]) for idx in J_indices)

	if weight.shape[-1] != n_factors:
		raise ValueError(
			f"weight has {weight.shape[-1]} entries but J carries {n_factors} factors. "
			f"weight must be |concatenate(Jt)| - factor ids from 0, no bias factors, which "
			f"is NOT init_correlation_index's numbering (that starts at N)."
		)

	if len(offsets) != len(J_indices):
		raise ValueError(
			f"got {len(offsets)} group offsets for {len(J_indices)} interaction orders"
		)

	#EVERY WINDOW MUST LIE INSIDE `weight`. The length check above is invariant to SHIFTING
	#the offsets, so on its own it does NOT catch the init_correlation_index numbering its
	#own message warns about - offsets + N passes it. And a shifted start does not raise
	#downstream either: jax.lax.dynamic_slice CLAMPS an out-of-range start rather than
	#erroring, so the slice silently returns a shifted window and every factor is scored
	#with another factor's J_tilde. Verified: factor_seed_mask(J, w, N, th, offs + N)
	#returned a mask with no error before this check existed.
	#
	#This also admits a legitimate SUBSET - one group's indices with its weight already
	#sliced and offsets = (0,) - which is how backbone_factor_order restricts the gate to
	#one interaction order without weakening anything.
	for idx, start in zip(J_indices, offsets):
		end = int(start) + int(idx.shape[0])
		if int(start) < 0 or end > int(weight.shape[-1]):
			raise ValueError(
				f"group offset {start} with {int(idx.shape[0])} factors addresses "
				f"weight[{start}:{end}] but weight holds {int(weight.shape[-1])} entries. "
				f"Offsets index concatenate(Jt) from 0 with no bias factors; "
				f"init_correlation_index's numbering starts at N and will not fit."
			)


@partial(jax.jit, static_argnums = (2, 4))
def factor_seed_mask(J_indices: List[jax.Array],
										 weight: jax.Array,
										 N: int,
										 theta_seed: float,
										 offsets: tuple) -> jax.Array:
	"""
	Seeds are FACTORS, not variables (NMC paper Sec. V, Eq. 15).

	Every factor whose effective coupling clears theta_seed is a seed, and ALL k of its
	members enter the backbone together. This is the piece no retuning of a per-variable
	threshold can express: |J_tilde| is defined on a factor, and a k-body term is rigid or
	not as a unit.

	weight is the flat |concatenate(Jt)| array, so its factor ids start at 0. Note this is
	NOT init_correlation_index's numbering, which starts at N because the N bias factors
	come first in the pgmax factor graph - the two differ by exactly N.

	offsets is a static tuple of per-group start positions into weight.
	"""

	#dynamic_slice CLAMPS an out-of-range start instead of raising, so a weight array of the
	#wrong length would silently read the wrong factors rather than fail. Shapes are static
	#here, so this costs nothing and fires at trace time.
	_check_weight_length(J_indices, weight, offsets)

	seed = jnp.zeros(N, dtype = bool)

	for idx, start in zip(J_indices, offsets):
		w_g = jax.lax.dynamic_slice(weight, (start,), (idx.shape[0],))

		#absolute value, never signed: strong ANTI-correlation is rigidity too, and RLNMC
		#states the convention as atanh|<s_i s_j ...>|/beta
		seed = seed | _scatter_members(idx, jnp.abs(w_g) >= theta_seed, N)

	return seed


@partial(jax.jit, static_argnums = (4,))
def factor_clustering(J_indices: List[jax.Array],
											weight: jax.Array,
											cl_seed: jax.Array,
											theta_cut: float,
											offsets: tuple) -> jax.Array:
	"""
	Grow clusters gated PER FACTOR, to the connected component. No depth cap, no size cap.

	Sec. V: "The next spins to be added are those spins outside the cluster that are
	connected to one or more spins inside the cluster and their effective couplings ... are
	above the threshold cutoff." A factor is traversable iff it touches the cluster and its
	own |J_tilde| clears theta_cut; when it is, all k of its members are admitted. A factor
	is never skipped because one of its members is already claimed - the "already in the
	cluster" test exists only to terminate.

	Requires theta_seed >= theta_cut (checked by the caller), so that every seed factor is
	itself above cutoff and each seed's component is well defined. The result is a fixed
	point of a monotone expansion, so it does not depend on any enumeration order at all -
	unlike the reference implementation, whose np.setdiff1d against already-claimed spins
	(cnmc.py:290, :307) makes its clusters order-dependent.

	k appears only as an array dimension; nothing here branches on it, so mixed interaction
	orders are just more groups in the loop.
	"""

	#dynamic_slice CLAMPS an out-of-range start instead of raising, so a weight array of the
	#wrong length would silently read the wrong factors rather than fail. Shapes are static
	#here, so this costs nothing and fires at trace time.
	_check_weight_length(J_indices, weight, offsets)

	N = cl_seed.shape[0]

	def one_step_expansion(current_cl):
		new_cl = current_cl

		for idx, start in zip(J_indices, offsets):
			w_g = jax.lax.dynamic_slice(weight, (start,), (idx.shape[0],))

			#"connected to one or more spins inside the cluster" AND above cutoff
			active = jnp.any(current_cl[idx], axis = 1) & (jnp.abs(w_g) >= theta_cut)

			new_cl = new_cl | _scatter_members(idx, active, N)

		return new_cl

	return _grow_to_fixed_point(one_step_expansion, cl_seed)


def _scatter_members(idx: jax.Array, active: jax.Array, N: int) -> jax.Array:
	"""
	Mark every member of every active factor. idx is (n_terms, k), active is (n_terms,).

	Scatter-max on int8 rather than on bool: jax has no bool scatter-max, and add-then-test
	would need a dtype wide enough for the maximum number of factors touching one spin.
	"""

	hit = jnp.zeros(N, dtype = jnp.int8).at[idx].max(
		jnp.broadcast_to(active[:, None].astype(jnp.int8), idx.shape)
	)

	return hit > 0


@jax.jit
def anygraph_clustering(indices: jax.Array, 
										 		cl_seed: jax.Array,
										 		cl_cndt: jax.Array) -> jax.Array:
	"""
	Grow clusters on any graph, gated PER VARIABLE, to the connected component.

	A neighbour joins iff it is a valid candidate (cl_cndt), so the result is the connected
	component of the seed set inside the candidate-induced subgraph. There is no depth cap
	and no size cap: size is controlled by the threshold that produced cl_cndt.

	This previously ran a fixed CLUSTERING_DEPTH = 20 hops, which returned the 20-hop BALL
	rather than the component. That constant existed only because lax.scan needs a static
	trip count - "iterate until converged" cannot be written with scan. lax.while_loop can,
	so the constraint is dissolved rather than retuned. Measured on the interaction graph of
	DCL_instances/C8/00.txt (largest component 431, diameter 24), the cap reached 393 of 431.

	The docstring here once claimed a depth of int(log(N)/log(nbrhd_size) + 2), which never
	ran and is worse than the constant it was meant to replace: that formula assumes a
	radius-d ball holds nbrhd_size**d spins (expander growth), while chimera and DCL are
	quasi-2D lattices whose balls grow like d**2. On chimera128 (N = 128, nbrhd_size = 6,
	diameter 8) it yields depth 4, reaching 65 of 128.

	Iterating to the fixed point is also CHEAPER than the old scan, which always ran all 20
	steps: while_loop stops on convergence, 8 steps on chimera128.
	"""

	N = cl_seed.shape[0]
	assert N == cl_cndt.shape[0] and "Candidates and the problem size do not match!"

	def get_valid_nbrs(nbrs: jax.Array, clst: jax.Array):
		#set all neighbours to be part of a cluster
		new_clst = jnp.zeros_like(cl_seed, dtype = bool).at[nbrs].set(clst, mode = "drop")

		#check if the neighbours are valid cantidates
		return jnp.logical_and(new_clst, cl_cndt)

	def one_step_expansion(current_cl):
		all_new_cl = jax.vmap(get_valid_nbrs)(indices, current_cl)
		new_cl = all_new_cl.sum(axis = 0).astype(bool)

		return jnp.logical_or(current_cl, new_cl)

	return _grow_to_fixed_point(one_step_expansion, cl_seed)


@jax.jit
def choose_a_cluster_seed_nmc(key: jax.Array, cndt: jax.Array, cl_n_seeds: int):
	#several cluster seeds can be chosen, if necessary

	N = cndt.size
	perm = jax.random.permutation(key, N)

	cl_seeds = jnp.argwhere(cndt[perm], size = N, fill_value = N).reshape(-1)
	cl_seeds = jnp.where(jnp.arange(N) < cl_n_seeds, cl_seeds, N)
	cl_seed_pos = perm.at[cl_seeds].get(mode = 'fill', fill_value = N)
	
	return jnp.full((N,), False).at[cl_seed_pos].set(True, mode = 'drop')


@partial(jax.jit, static_argnums=(1))
def get_clusters_anygraph(
	couplings: Jhdata, 
	problem_mode: str, 
	cl_seeds: jax.Array, 
	cl_candidates: jax.Array
):
	
	N = cl_seeds.shape[0]
	assert N == couplings.Jat[0].indices.shape[0] and "wrong shapes!"

	if problem_mode in ["pubo", "pising"]:
		#add fillers
		#A real Jat slot has data != 0; a padding slot has data == 0 and carries index 0
		#(np.zeros), which is a real spin - so the mask must key on data, and the padded
		#slots are the ones replaced by the out-of-range filler N. anygraph_clustering's
		#.at[nbrs].set(..., mode = "drop") then discards them.
		#Every interaction order contributes neighbours. The groups carry different last
		#axes (k-1 per order), so each is flattened before they are concatenated.
		# pubo/p-Ising (they use same indexing 0, N-1)
		nbhr_blocks = [
			jnp.where(jat.data[:, :, None], jat.indices, N).reshape(N, -1)
			for jat in couplings.Jat
		]

		action = anygraph_clustering(jnp.concatenate(nbhr_blocks, axis = 1),
																 cl_seeds,
																 cl_candidates)
	elif problem_mode == "wcnf":
		# wcnf (indices are signed)
		# add fillers
		nbhr_indices = jnp.where(
			couplings.Jat[0].data[:, :, None],
			couplings.Jat[0].indices, 
			N+1
		)
		action = anygraph_clustering((jnp.abs(nbhr_indices[:, :, :-1]) - 1).reshape(N, -1), 
																 cl_seeds,
																 cl_candidates)

	else:
		raise RuntimeError("Unsupported problem mode!")
	
	return action
