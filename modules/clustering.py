import sys
sys.path.append('.')

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from modules.nmc_types import *

@jax.jit
def anygraph_clustering(indices: jax.Array, 
										 		cl_seed: jax.Array,
										 		cl_cndt: jax.Array) -> jax.Array:
	"""
	Growing clusters on any graph to a int(jnp.log(N)/jnp.log(nbrhd_size) + 2) depth
	"""

	N = cl_seed.shape[0]
	assert N == cl_cndt.shape[0] and "Candidates and the problem size do not match!"

	nbrhd_size = indices[0].size
	clustering_depth = 20
	
	def get_valid_nbrs(nbrs: jax.Array, clst: jax.Array):
		#set all neighbours to be part of a cluster
		new_clst = jnp.zeros_like(cl_seed, dtype = bool).at[nbrs].set(clst, mode = "drop")

		#check if the neighbours are valid cantidates
		return jnp.logical_and(new_clst, cl_cndt)

	def one_step_expansion(current_cl, not_used):
		all_new_cl = jax.vmap(get_valid_nbrs)(indices, current_cl)
		new_cl = all_new_cl.sum(axis = 0).astype(bool)
		
		return jnp.logical_or(current_cl, new_cl), None
	
	final_clst, _ = jax.lax.scan(one_step_expansion, cl_seed, length = clustering_depth)

	return final_clst


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
		nbhr_indices = jnp.where(
			jnp.logical_not(couplings.Jat[0].data[:, :, None]),
			couplings.Jat[0].indices, 
			N
		)
		# pubo/p-Ising (they use same indexing 0, N-1)
		action = anygraph_clustering(nbhr_indices.reshape(N, -1), 
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
