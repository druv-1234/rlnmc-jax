import sys
import dataclasses
import itertools

sys.path.append('.')

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from typing import Dict, Tuple, List, Union
from functools import partial
from dataclasses import replace

from pgmax import fgraph
from pgmax import fgroup
from pgmax import vgroup
from pgmax import infer
from pgmax.infer.bp import pass_var_to_fac_messages
from pgmax.factor import enum

from modules.nmc_types import *


def clip_off_unit(x: jax.Array) -> jax.Array:
	"""
	Clip into (-1, 1), one ulp short of each end, in the array's own dtype.

	The single place the "keep tanh-like quantities strictly inside the unit interval"
	rule is expressed. Both callers need it for the same reason - in float32 tanh
	saturates to EXACTLY +-1.0 at |x| >~ 9.02, and every downstream formula here divides
	by or takes arctanh of something that is then degenerate.
	"""
	lim = jnp.asarray(1.0, x.dtype) - jnp.finfo(x.dtype).eps
	return jnp.clip(x, -lim, lim)


def atanh_saturated(x: jax.Array) -> jax.Array:
	"""
	arctanh with saturation, mirroring Nonlocal-Monte-Carlo/NMC/cnmc.py:230-255 but
	dtype-aware instead of hard-coding a float64 constant.

	The NumPy version clips at tanh(19.06), which is where tanh saturates to 1.0 in
	float64. That constant does NOT transfer: in float32 tanh(19.06) is exactly 1.0, so
	the clip is a no-op and arctanh returns inf. Clipping at 1 - eps(dtype) reproduces
	both limits from the dtype itself - about 8.32 in float32 and 18.4 in float64.

	This matters because J_tilde = arctanh(correlation)/beta diverges exactly in the
	rigid-backbone regime the algorithm selects for.
	"""
	return jnp.arctanh(clip_off_unit(x))

def init_factor_graph(J: Union[List[Jsp], List[jax.Array]],
											h: jax.Array, 
											lbp_beta: float,
											problem_mode: str = "wcnf") -> fgraph.FactorGraph:
	N = h.size

	variables = vgroup.NDVarArray(num_states = 2, shape = (N,)) 
	factor_graph = fgraph.FactorGraph(variable_groups = variables)

	if problem_mode == "pubo": #pubo biases
		h_potentials = np.stack([h, np.zeros_like(h)], 
														axis = 1).astype(np.float32)*lbp_beta # E = -h_ix_i, p ~ exp(h_ix_i)

	elif problem_mode == "pising": #pising magnetic fields
		h_potentials = np.stack([h, -h], 
														axis = 1).astype(np.float32)*lbp_beta # E = -h_is_i, p ~ exp(h_ix_i)

	else: #cnf weighted single clauses
		h_potentials = np.stack([-jnp.abs(h)*jnp.signbit(h), 
													 	 -jnp.abs(h)*jnp.logical_not(jnp.signbit(h))], 
														 axis = 1).astype(np.float32)*lbp_beta # E = h_ix_i, p ~ exp(-h_ix_i)

	h_factors = fgroup.EnumFactorGroup(
    variables_for_factors = [[variables[i]] for i in range(N)],  #type: ignore
    factor_configs = np.array([1, 0])[:, None],
    log_potentials = h_potentials
	)

	J_factors = []
	for j in J:
		if isinstance(j, Jsp):
			data = j.data
			indices = j.indices
		else:
			indices = j
			data = jnp.full((indices.shape[0],), 1.0)

		k = indices.shape[1]
		if problem_mode in ["pubo", "pising"]:
			variables_for_factors = [[variables[i] for i in idx] for idx in indices]
		else:
			#sort indices in each clause for possible duplicates removal
			sort_ind = np.argsort(np.abs(indices), axis = 1)
			indices = jnp.take_along_axis(indices, sort_ind, axis = 1)

			#in cnf indexing in clauses is with a sign and starts from 1
			variables_for_factors = [[variables[np.abs(i)-1] for i in idx] for idx in indices]

		factor_configs = np.array(list(itertools.product([1, 0], repeat = k)))
		if problem_mode == "pubo":
			log_potentials = data[:, None]*np.prod(factor_configs, axis=1).astype(np.float32)*lbp_beta

		elif problem_mode == "pising":
			s_configs = np.array(list(itertools.product([1, -1], repeat = k)))
			log_potentials = data[:, None]*np.prod(s_configs, axis = 1).astype(np.float32)*lbp_beta

		else:
			s_configs = np.array(list(itertools.product([1, -1], repeat = k)))
			clause_evaluations = np.prod(indices[:, None, :]*s_configs > 0, axis=2).astype(np.float32)
			log_potentials = -data[:, None]*clause_evaluations*lbp_beta

			#remove duplicates from the factor graph (merge factors with the same variables)
			indices_unique = np.unique(np.abs(indices), axis = 0, return_counts = True)
			indices_unique_where = np.where(indices_unique[1] > 1)[0]
			if indices_unique_where.size > 0:
				indices_unique = indices_unique[0][indices_unique_where]
				for idx in indices_unique:
					where_duplicates = np.where((idx == np.abs(indices)).all(axis = 1))[0]
					log_potential_sum = np.sum(log_potentials[where_duplicates], axis = 0)
					indices = np.stack([x for i, x in enumerate(indices) if i not in where_duplicates] + [indices[where_duplicates[0]]])
					variables_for_factors = (
						[x for i, x in enumerate(variables_for_factors) if i not in where_duplicates] + 
						[variables_for_factors[where_duplicates[0]]]
					)
					log_potentials = np.stack([x for i, x in enumerate(log_potentials) if i not in where_duplicates] + [log_potential_sum])
		
		J_factors.append(fgroup.EnumFactorGroup(variables_for_factors, factor_configs, log_potentials))  #type: ignore

	factor_graph.add_factors([h_factors] + J_factors)

	return factor_graph


def init_correlation_index(FG: fgraph.FactorGraph,
													 J: List[Jsp],
													 N: int,
													 problem_mode: str = "pising"):
	"""
	Precompute the static index arrays needed to read k-local correlations off a
	converged pgmax run. Call once at setup; the result is passed to factor_correlations.

	pgmax exposes no factor beliefs and no correlation API, but it does expose both
	message directions, which is all Eq. (C3) of the NMC paper needs. The wiring array
	var_states_for_edges has shape (num_edge_states, 3):
		col 0 = global variable-state index (variable = idx // 2)
		col 1 = global edge index
		col 2 = global FACTOR index
	so the edge -> factor grouping is stored directly, with no offsets to derive.

	Factors are laid out contiguously in the order they were added by init_factor_graph:
	the N single-variable bias factors first, then one group per interaction order in
	J's order. So each group occupies a contiguous factor-index range.

	Returns (vsfe0, rows_per_group) where rows_per_group[g] has shape (n_terms, k, 2)
	and indexes into the flat message array.
	"""

	if problem_mode != "pising":
		#Eq. (C3) contains tanh(beta*J_a), which presumes the factor IS an Ising k-body
		#term J*prod(s). wcnf factors are clause indicators - a different object - so the
		#formula would return meaningless numbers rather than fail.
		raise NotImplementedError(
			f"k-local correlations are only defined for pising factors, got {problem_mode!r}"
		)

	vsfe = np.asarray(FG.fg_state.wiring[enum.EnumFactor].var_states_for_edges)

	rows_per_group = []
	start = N                                     #bias factors occupy 0 .. N-1

	for j in J:
		n_terms, k = j.indices.shape

		sel = np.nonzero((vsfe[:, 2] >= start) & (vsfe[:, 2] < start + n_terms))[0]
		if sel.size != n_terms*k*2:
			raise RuntimeError(
				f"correlation index: expected {n_terms*k*2} edge states for the order-{k} "
				f"group, found {sel.size}. Factor layout assumption broken."
			)

		rows = sel.reshape(n_terms, k, 2)
		vs   = vsfe[:, 0][rows]                     #(n_terms, k, 2) global variable-STATE indices

		#The reshape above assumes the ascending np.nonzero rows arrive (a) blocked by factor
		#and (b) within a factor, in the same member order as j.indices. The size check above
		#counts elements and says nothing about that arrangement.
		#
		#Both hold by construction in pgmax 0.6.1 - enum.compile_wiring builds the edge list
		#with a nested `for factor_idx, variables_for_factor in enumerate(...): for variable in
		#variables_for_factor:` and never sorts, then compile_var_states_for_edges_numba emits
		#num_states consecutive rows per edge in ascending state order - but NOT by contract:
		#the Wiring docstring specifies only what the three COLUMNS mean, never the row order.
		#A pgmax upgrade can therefore break this silently.
		#
		#Silently matters literally here. factor_correlations computes h_ia - pin ELEMENTWISE
		#(this file, in the Jt loop below) before reducing over the member axis, so a
		#permutation subtracts one variable's pinning field from another's cavity field. The
		#result stays inside (-1, 1) with no NaN and no exception - it is just the wrong
		#backbone. corr_pin survives it because jnp.prod over the member axis is
		#permutation-invariant; corr_dep does not, and "depinned" is the default lbp_pinning.
		if not (vs[:, :, 0]//2 == vs[:, :, 1]//2).all():
			raise RuntimeError(
				f"correlation index: the two edge states of an order-{k} factor member do not "
				f"belong to the same variable. pgmax edge-state layout assumption broken."
			)

		if not ((vs[:, :, 0] % 2 == 0).all() and (vs[:, :, 1] % 2 == 1).all()):
			raise RuntimeError(
				f"correlation index: order-{k} edge states are not ordered (state 0, state 1). "
				f"pgmax edge-state layout assumption broken."
			)

		if not (vs[:, :, 0]//2 == np.asarray(j.indices)).all():
			raise RuntimeError(
				f"correlation index: order-{k} member order does not match j.indices. "
				f"factor_correlations subtracts the pinning field elementwise, so this would "
				f"silently corrupt the de-pinned correlations."
			)

		rows_per_group.append(jnp.array(rows, dtype = jnp.int32))
		start += n_terms

	return jnp.array(vsfe[:, 0], dtype = jnp.int32), rows_per_group


@jax.jit
def _atanh_prod_tanh(x: jax.Array) -> jax.Array:
	"""
	arctanh(prod_i tanh(x_i)) along the last axis, without ever forming the product.

	WHY NOT jnp.arctanh(jnp.prod(jnp.tanh(x), axis = -1)). float32 spacing at 1 is
	eps = 1.19e-7, so any tanh(x) with x >~ 9.02 is stored as exactly 1.0, and a product
	containing one is too. arctanh(y) needs 1 - y, and 1 - y is precisely what the
	rounding destroyed, so the direct form caps at arctanh(1 - eps) = 8.3178 however large
	x is - measured, it returns 8.3178 for x = 9, 15 and 30 alike. Computing the arctanh
	in float64 does NOT help: the loss happens at the tanh, before arctanh is called.

	THE FORM. With u = exp(-2|x|), tanh|x| = (1 - u)/(1 + u) = 1 - 2u/(1 + u), so
	log|tanh| = log1p(-2u/(1 + u)) - a small negative number rather than "1 minus a small
	number". Summing gives d = log|prod tanh|, and

		arctanh(y) = 0.5*log((1 + y)/(1 - y))
					= 0.5*(log1p(exp d) - log(-expm1 d)),    |y| = exp(d)

	expm1 delivers 1 - |y| at full relative precision where 1 - exp(d) would cancel. The
	rule throughout: no intermediate ever stores a value of the shape 1 - tiny.

	THE SIGN, which d does not carry. prod tanh(x_i) is negative whenever an ODD number of
	the x_i are, and dropping that is not a rounding error - it flips the effective
	coupling. Measured against 60-digit decimal on exactly those cases: |err| = 1.8 to 3.4
	in J_tilde units, delivered with confident-looking precision. So the sign is tracked
	separately and multiplied back.

	sign(0) = 0 is correct here rather than a case to special-case: an x_i of 0 makes its
	tanh 0, hence the product 0 and arctanh 0, which the zero sign delivers. The magnitude
	branch does reach d = -inf there, but evaluates to a finite 0
	(log1p(exp(-inf)) - log(-expm1(-inf)) = 0 - 0), so there is no 0*inf.

	VERIFIED against Python's stdlib decimal at 60 digits computing the ORIGINAL formula
	(build the tanhs, form the product, then arctanh), which shares none of this algebra:
	|err| <= 2.9e-07 for lbp_beta 5 to 20, against 0.14 to 1.9 for the direct form.
	"""
	ax   = jnp.abs(x)
	sign = jnp.prod(jnp.sign(x), axis = -1)
	u    = jnp.exp(-2.0*ax)
	d    = jnp.sum(jnp.log1p(-2.0*u/(1.0 + u)), axis = -1)          #= log|prod tanh|

	#DEEP SATURATION. u underflows to exactly 0 in float32 once 2|x| > ~87, i.e.
	#|x| > ~43.5, which makes d exactly 0 and log(-expm1(0)) = log(0) = -inf. The
	#operating point is nowhere near it (the coldest NMC rung at 9.54 against a bare
	#field of 1.75 gives |x| ~ 16.7), but it is the same class of silent cliff this
	#function exists to remove, so it is handled rather than left to be found later.
	#
	#There 1 - |P| ~ sum_i 2u_i, so arctanh -> -0.5*logsumexp_i(-2|x_i|), which the
	#max-subtraction below evaluates without ever forming exp of a large negative:
	#with m = min|x_i| it is m - 0.5*log(sum_i exp(-2(|x_i| - m))), every exponent <= 0.
	#That is also the lbp_beta -> infinity limit of the whole expression, J_g + sign*min|h|.
	m    = jnp.min(ax, axis = -1)
	deep = m - 0.5*jnp.log(jnp.sum(jnp.exp(-2.0*(ax - m[..., None])), axis = -1))

	#d < 0 is the ordinary case; d == 0 means every term underflowed. Both branches are
	#evaluated under jit and the discarded one is +-inf rather than NaN, so the select
	#is safe.
	mag  = jnp.where(d < 0.0,
					0.5*(jnp.log1p(jnp.exp(d)) - jnp.log(-jnp.expm1(d))),
					deep)
	return sign*mag


def factor_correlations(bp_arrays,
												vsfe0: jax.Array,
												rows_per_group: List[jax.Array],
												J_per_group: List[jax.Array],
												idx_per_group: List[jax.Array],
												lbp_beta: jax.Array,
												s: jax.Array,
												epsilon: jax.Array,
												final_lambda: jax.Array):
	"""
	k-local correlations and effective couplings from a converged LBP run.

	NMC paper Eq. (13) / (C3):
		<prod_{i in a} r_i> = ( tanh(beta*J_a) + prod_i tanh(beta*h_{i->a}) )
													/ ( 1 + tanh(beta*J_a) * prod_i tanh(beta*h_{i->a}) )
	and Eq. (15):
		J_tilde_a = arctanh(<prod r_i>) / beta

	J_tilde is the effective coupling - the quantity the paper thresholds to seed and
	grow backbones (p9). Pairwise correlations are NOT the right object here: the paper
	notes they "bring almost no information in high-order interacting models".

	h_{i->a} are the variable-to-factor (cavity) messages. pgmax stores messages as
	2-state log-potentials rather than scalar fields, so each is converted with
	(m1 - m0)/2/beta - the same conversion already used on beliefs at nmcgym.py:216-217.
	Convention: state 1 <-> s = +1.

	PINNED vs DE-PINNED. The surrogate's evidence is lambda*epsilon_i*s_i*beta, a field
	pinning each spin to its current value, and pass_var_to_fac_messages sums evidence
	into the cavity message - so h_ia carries that pin. get_geometry subtracts exactly
	this contribution from `mag` because it is an artifact of the surrogate, so leaving
	it in the correlations puts the two quantities on different footings. It is not
	small: at lbp_lambdas ~ 0.0075, lbp_beta = 5 and epsilon ~ 20 the residual pin alone
	contributes tanh(0.75) ~ 0.63 inside the product over cavity fields, which biases
	J_tilde toward whatever state the walker is already in - the opposite of what
	backbone detection is for. Both forms are computed so the choice is a config flag
	(nmc_setup.lbp_pinning) rather than a hard-coded convention.

	Returns (corr_depinned, Jt_depinned, corr_pinned, Jt_pinned), each a list with one
	entry per interaction order.
	"""

	vtof = pass_var_to_fac_messages(bp_arrays.ftov_msgs, bp_arrays.evidence, vsfe0)

	corr_dep, Jt_dep = [], []
	corr_pin, Jt_pin = [], []

	for rows, J_g, idx_g in zip(rows_per_group, J_per_group, idx_per_group):
		msgs = vtof[rows]                                        #(n_terms, k, 2)
		h_ia = (msgs[:, :, 1] - msgs[:, :, 0])/2/lbp_beta         #scalar cavity field

		#the surrogate pinning field carried by each (term, member) slot; same expression
		#get_geometry removes from `mag`, gathered onto the factor's member spins
		pin = final_lambda*epsilon[idx_g]*(2*s[idx_g] - 1)

		#Clipped, NOT guarded with an additive epsilon. float32 tanh saturates to exactly
		#+-1.0 at |x| >~ 9.02, so at lbp_beta = 5 any |J| >= 1.81 gives tJ == +-1, and
		#`prod` saturates just as readily; with tJ = +1 and prod = -1 both numerator and
		#denominator are 0 and corr is NaN, which atanh_saturated then propagates silently
		#into the logged stats.
		#
		#cnmc.py:217-218 handles this by adding 1e-10 to the denominator, and that does NOT
		#transfer. It works there because NumPy is float64 and evaluates the sum eagerly.
		#Here 1e-10 is far below float32 eps (1.19e-7), so XLA reassociates
		#(1 + tJ*prod) + 1e-10 into (1 + 1e-10) + tJ*prod and constant-folds the leading
		#term to exactly 1.0 - the compiled HLO contains no epsilon at all, and the guard
		#is a no-op in every jitted path. It only ever "worked" when run eagerly.
		#(Same class of trap as the 0.9999999 threshold that does not survive float32.)
		#
		#Clipping both factors one ulp off +-1 bounds |1 + tJ*prod| >= ~2*eps structurally,
		#so there is nothing for the simplifier to remove. tests/test_lbp.py pins both the
		#mechanism and the end-to-end consequence.
		tJ = clip_off_unit(jnp.tanh(lbp_beta*J_g))

		for h_eff, corr_out, Jt_out in ((h_ia - pin, corr_dep, Jt_dep),
																		(h_ia,       corr_pin, Jt_pin)):
			prod = clip_off_unit(jnp.prod(jnp.tanh(lbp_beta*h_eff), axis = 1))

			corr = (tJ + prod)/(1 + tJ*prod)

			corr_out.append(corr)
			#J_tilde is NOT arctanh(corr)/lbp_beta evaluated literally. corr = (tJ + prod)/
			#(1 + tJ*prod) IS the tanh addition formula, so arctanh(corr) = arctanh(tJ) +
			#arctanh(prod), and arctanh(tJ) = arctanh(tanh(lbp_beta*J_g)) = lbp_beta*J_g
			#EXACTLY - tanh and arctanh are inverses. The literal form computes that tanh,
			#watches it saturate for |J_g| >= 1.81 (the comment above), clips it, and then
			#inverts it to recover a number it already had. Dividing through by lbp_beta:
			#
			#    J_tilde = J_g + arctanh(prod tanh(lbp_beta*h_eff))/lbp_beta
			#
			#which removes the arctanh(1 - eps)/lbp_beta = 8.3178/lbp_beta cap entirely. That
			#cap is why per-rung lbp_beta was not viable: it shrinks as the rung gets colder,
			#so at rung beta 9.54 it sat at 0.87 and the whole theta bracket [1.0, 1.66] was
			#unreachable. Measured error against 60-digit decimal, at the five NMC rungs
			#(4.72 .. 9.54): 4.3e-08 to 1.8e-07, against 0.14 to 1.06 for the literal form.
			#
			#`corr` above is left exactly as it was: it is still returned and logged as
			#corr_k*, clip_off_unit still keeps its 0/0 finite at saturation, and the tests
			#for that guard keep their meaning. tanh(lbp_beta*J_tilde) and corr therefore
			#DISAGREE wherever corr saturates - by construction, since that is exactly where
			#corr has stopped carrying information.
			Jt_out.append(J_g + _atanh_prod_tanh(lbp_beta*h_eff)/lbp_beta)

	return corr_dep, Jt_dep, corr_pin, Jt_pin

@partial(jax.jit, static_argnames=['bp', 'FGvars', 'num_iters', 'problem_mode'])
def surrogate_LBP(bp,
									FGvars,
									s: jax.Array, 
									epsilon: jax.Array,
									lbp_beta: jax.Array,
									potential_scale: jax.Array,
									num_iters: int,
									problem_mode: str,
									tolerance_m: float,
									tolerance_d: float,
									lambdas: jax.Array) ->  Tuple[jax.Array, Dict]:
	"""
	Loopy belief propagation on the problem's factor graph and using the convexified surrogate Hamiltonian

	Parameters:
		s (jax.Array): current state of the problem
		epsilon (jax.Array): A scaling factor for every spin to account for different interactions
		lbp_beta (float): temperature of lbp
		num_iters (int): pgmax lbp number of iterations
		num_iters (int): pgmax lbp number of iterations
		tolerance_m (float): a maximum allowed bp relative diff of messages to stop
		tolerance_d (float): a maximum allowed distance from the initial state to stop
		lambdas (Tuple): control parameter of the "pinned" Loopy Belief Propagation
		potential_scale (jax.Array): multiplies the factor graph's energy table, so the graph
			can be BUILT at one temperature and RUN at another. beta enters init_factor_graph
			as one global multiply on every log potential and the wiring does not depend on it
			at all, so scaling here is exactly equivalent to rebuilding - verified to 1e-6 on a
			mixed-arity fixture, against a wrong-scale control that disagrees by 6.7 and 22.
			Pass 1.0 to keep the graph's own temperature.
	"""
	run_with_diffs = partial(bp.run_with_diffs, #type: ignore
													 num_iters = num_iters, 
													 damping = 0.0, 
													 temperature = 1.0)
	
	def map_s_distance(m, s):
		return (m != s.astype(int)).sum()/s.size
	
	if problem_mode in ["pubo", "pising"]:
		evidence_0 = (jnp.stack([1-2*s, 2*s-1], axis = 1)*epsilon[:, None]).astype(jnp.float32)*lbp_beta
	else:
		evidence_0 = (jnp.stack([-1*s, s-1], axis = 1)*epsilon[:, None]).astype(jnp.float32)*lbp_beta

	def _rescale(arr):
		#BPArrays is a plain dataclass, not a flax struct, so dataclasses.replace - not .replace
		return dataclasses.replace(arr, log_potentials = arr.log_potentials*potential_scale)

	bp_arrays = _rescale(bp.init(evidence_updates = {FGvars: evidence_0*lambdas[0]}))
	bp_arrays, msgs_eps = run_with_diffs(bp_arrays)
	map_state = infer.decode_map_states(bp.get_beliefs(bp_arrays))[FGvars]

	ms_d = map_s_distance(map_state, s)

	lbp_stats = {}

	lbp_stats["start_msgs_delta"] = msgs_eps[-1]
	lbp_stats["start_distance"] = ms_d
	lbp_stats["start_lambda"] = lambdas[0]

	# jax.debug.print("Start messages delta: {x}", x = msgs_eps[-1])
	# jax.debug.print("Start distance: {x}", x = ms_d)
	# jax.debug.print("Start lambda: {x}",  x = lambdas[0])
	
	#Setting up initial lambda (increasing) in case LBP diverged beyond the threshold
	def run_bp_lmbd_increase(x):
		lmbd = x[3]/lambdas[2]

		bp_arrays = _rescale(bp.init(evidence_updates = {FGvars: evidence_0*lmbd}))

		bp_arrays, msgs_eps = run_with_diffs(bp_arrays)

		ms_d = map_s_distance(infer.decode_map_states(bp.get_beliefs(bp_arrays))[FGvars], s)

		# jax.debug.print("Setting up: dmsgs = {x}, ms_d = {y}, lambda = {z}", x = msgs_eps[-1], y = ms_d, z = lmbd)

		return bp_arrays, msgs_eps[-1], ms_d, lmbd, x[4] + 1
	
	def set_lambda_init(bp_arr, m_eps, ms_d, lmbd):
		return lax.while_loop(lambda x: jnp.logical_or(x[1] > tolerance_m, x[2] > tolerance_d), 
													run_bp_lmbd_increase, (bp_arr, m_eps, ms_d, lmbd, 0))

	bp_arrays, m_eps, ms_d, lmbd, init_steps = \
		lax.cond(jnp.logical_or(msgs_eps[-1] > tolerance_m, ms_d > tolerance_d), 
						 set_lambda_init, 
						 lambda bp_arr, m_eps, ms_d, lmbd: (bp_arr, m_eps, ms_d, lmbd, 0), 
						 bp_arrays, msgs_eps[-1], ms_d, lambdas[0])	


	lbp_stats["init_steps"] = init_steps #number of initialization steps
	lbp_stats["init_msgs_delta"] = m_eps
	lbp_stats["init_distance"] = ms_d
	lbp_stats["init_lambda"] = lmbd

	# jax.debug.print("Init messages delta: {x}", x = m_eps)
	# jax.debug.print("Init distance: {x}", x = ms_d)
	# jax.debug.print("Init lambda: {x}",  x = lmbd)

	#Reducing lambda until LBP diverges beyond the threshold
	def run_bp_lmbd_decrease(x):
		bp_arrays_old = x[0][0]
		m_delta_old = x[0][1]
		ms_d_old = x[0][2]
		lmbd_old = x[0][3]

		lmbd = x[0][3]*lambdas[2] #decrease the value of lambda
		#initialize bp messages with the previous iteration and updated evidence
		bp_arrays = replace(x[0][0], evidence = evidence_0.flatten()*lmbd) 

		bp_arrays, msgs_eps = run_with_diffs(bp_arrays)

		#get the map state and compute distance to reference
		map_state = infer.decode_map_states(bp.get_beliefs(bp_arrays))[FGvars]
		ms_d = map_s_distance(map_state, s)

		# jax.debug.print("Adjusting: dmsgs = {x}, ms_d = {y}, lambda = {z}", x = msgs_eps[-1], y = ms_d, z = lmbd)

		return (bp_arrays, msgs_eps[-1], ms_d, lmbd), (bp_arrays_old, m_delta_old, ms_d_old, lmbd_old), x[2] + 1

	final_lbp_res, pre_final_lbp_res, final_steps  = \
		lax.while_loop(lambda x: jnp.logical_and(jnp.logical_and(x[0][1] < tolerance_m, x[0][2] < tolerance_d), x[0][3] > lambdas[1]), 
									 run_bp_lmbd_decrease, 
									 ((bp_arrays, m_eps, ms_d, lmbd), (bp_arrays, m_eps, ms_d, lmbd), 0))


	lbp_res = lax.cond(jnp.logical_or(final_lbp_res[1] > tolerance_m, final_lbp_res[2] > tolerance_d), 
										 lambda: pre_final_lbp_res, 
										 lambda: final_lbp_res)

	lbp_stats["final_steps"] = final_steps #final number of lbp steps
	lbp_stats["final_msgs_delta"] = lbp_res[1]
	lbp_stats["final_distance"] = lbp_res[2]
	lbp_stats["final_lambda"] = lbp_res[3]
	
	#total number of steps in the lbp run
	lbp_stats["total_steps"] = init_steps + final_steps + 1
	lbp_stats["total_iterations"] = lbp_stats["total_steps"]*num_iters

	# jax.debug.print("Final messages delta: {x}", x = lbp_res[1])
	# jax.debug.print("Final distance: {x}", x = lbp_res[2])
	# jax.debug.print("Final lambda: {x}",  x = lbp_res[3])

	beliefs = bp.get_beliefs(lbp_res[0])

	#return the final beliefs of the LBP convergence, the converged BPArrays (which carry
	#the messages that factor_correlations reads), and the adjusted starting lambda value
	return beliefs, lbp_res[0], lbp_stats


if __name__ == "__main__":
	pass