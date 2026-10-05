from os import path 

import sys
sys.path.append('.')

import numpy as np
import jax.numpy as jnp

from modules.nmc_types import *

from typing import Dict, Any, Tuple

def get_wcnf(instance_path: str, 
						 model_dtype: Any,
						 no_weights: bool = False):
	
	dtype = np.int32 if model_dtype == "int" else np.float32
	all_clauses = {}

	with open(instance_path, 'r') as f:
		for line in f:
			if line.startswith('c') or line.startswith('p'):
				continue

			L = np.fromstring(line, dtype = dtype, sep = ' ')
			if no_weights:
				indices = L[:-1].astype(int)
			else:
				indices = L[1:-1].astype(int)

			K = len(indices)

			if K not in all_clauses.keys():
				if no_weights:
					all_clauses[K] = np.empty((0, K), dtype = np.int32)
				else:
					all_clauses[K] = [np.empty((0,), dtype = dtype), 
											 			np.empty((0, K), dtype = np.int32)]

			if no_weights:
				all_clauses[K] = np.append(all_clauses[K], indices)

			else:
				all_clauses[K][0] = np.append(all_clauses[K][0], L[0]) #set the weight of the clause
				all_clauses[K][1] = np.append(all_clauses[K][1], indices)

	wcnf = {}
	for k, v in all_clauses.items():
		wcnf[k] = WCNF(jnp.ones((v.size//k), dtype=dtype) , jnp.array(v))

	return wcnf


def wcnf_for_jax(wcnf_dict: Dict[int, WCNF], dtype) -> Jhdata:
	J_v = []
	Jat = []

	N = 0

	for k, wcnf in sorted(wcnf_dict.items(), reverse=False):
		n = np.max(np.abs(wcnf.indices))
		if n > N:
			N = n
		if k > 1:
			J_v.append(Jsp(jnp.array(wcnf.w, dtype = dtype), 
								 -jnp.array(wcnf.indices, dtype=jnp.int32).reshape(-1, k)))

	h = jnp.zeros(N, dtype = dtype)

	for k, wcnf in sorted(wcnf_dict.items(), reverse=False):
		jat_w = [[] for _ in range(N)]
		jat_idx = [[] for _ in range(N)]

		if k == 1:
			for i, l in enumerate(wcnf.indices):
				h = h.at[abs(l)-1].set(np.sign(l)*wcnf.w[i])
		else:
			for i, l in enumerate(wcnf.indices.reshape(-1, k)):
				for ci, c in enumerate(l):
					ltmp = -np.delete(l, ci)
					ltmp = np.append(ltmp, np.sign(c))

					jat_idx[abs(c)-1].append(ltmp)
					jat_w[abs(c)-1].append(wcnf.w[i])

			maxlen = np.max([len(j) for j in jat_idx])

			data = jnp.zeros((N, maxlen), dtype = dtype)
			indices = jnp.zeros((N, maxlen, k), dtype = jnp.int32)

			for i in range(N):
				if len(jat_idx[i]) != 0:
					indices = indices.at[i, :len(jat_idx[i]), :].set(jnp.stack(jat_idx[i]))
					data = data.at[i, :len(jat_w[i])].set(jnp.array(jat_w[i]))

			Jat.append(Jsp(data, indices))

	return Jhdata(J_v, h, Jat)


def read_ising_txt(instance_path: str,
									 one_based: Any = None,
									 N: Any = None):
	"""
	Parse an arbitrary-order Ising instance in the sparse text format used by the
	Nonlocal-Monte-Carlo repo (NMC/examples/{Chimera_droplet_instances,wishart_small}).

	Each non-comment line is a term: k indices followed by one coefficient, so the
	order is inferred from the token count.

		i i v          -> bias h[i]   (the legacy 2-token-plus-value form; chimera uses it)
		i v            -> bias h[i]
		i j v   (i!=j) -> 2-body coupling
		i j k v        -> 3-body coupling
		...            -> any order

	Note this reads straight into a term list. The NMC repo's txt_to_A_droplet builds a
	dense symmetric NxN matrix instead, which is why it stores every edge twice and its
	energy carries a compensating 1/2 (cnmc.py:386). A term list must store each term
	exactly once - the files themselves already do.

	A term may be written more than once (typically as both i j v and j i v, since J is
	symmetric). Repeats must restate the same value: they are stored once, never summed.
	An unequal repeat is an asymmetric coupling matrix and raises.

	Parameters:
	- one_based: index base. None auto-detects: a minimum index of 0 proves 0-based, and
	  otherwise the base is read off the top index against N (max == N is 1-based,
	  max == N-1 is 0-based). Anything ambiguous raises rather than guessing - the
	  converse of the min-index test is FALSE, because a 0-based file whose lowest spin
	  is isolated also has a minimum index above 0.
	- N: number of spins. None infers it as max index + 1, and makes an ambiguous index
	  base unresolvable.

	Returns:
	- terms (Dict[int, Tuple[np.ndarray, np.ndarray]]): order k -> (indices (n, k), values (n,))
	- h (np.ndarray): bias per spin, shape (N,)
	- N (int)
	"""

	raw_terms = []   #(sorted index tuple, value)
	raw_bias  = []   #(index, value)
	all_idx   = []

	with open(instance_path, 'r') as f:
		for lineno, line in enumerate(f, 1):
			line = line.strip()
			if not line or line.startswith('#') or line.startswith('c'):
				continue

			tokens = line.split()
			if len(tokens) < 2:
				raise ValueError(f"{instance_path}:{lineno}: need at least one index and a value, got {line!r}")

			try:
				idx = [int(t) for t in tokens[:-1]]
				val = float(tokens[-1])
			except ValueError:
				raise ValueError(f"{instance_path}:{lineno}: could not parse {line!r}")

			all_idx.extend(idx)

			#a bias is either a single index, or the legacy repeated-index diagonal form
			if len(idx) == 1 or (len(idx) == 2 and idx[0] == idx[1]):
				raw_bias.append((idx[0], val, lineno))
				continue

			#s_i*s_i == 1 for a +-1 spin, so a repeat silently lowers the order - reject it
			if len(set(idx)) != len(idx):
				raise ValueError(
					f"{instance_path}:{lineno}: repeated index in an order-{len(idx)} term {idx}. "
					"s_i*s_i = 1 for +-1 spins, so this is a lower-order term written ambiguously."
				)

			raw_terms.append((tuple(sorted(idx)), val, lineno))

	if not all_idx:
		raise ValueError(f"{instance_path}: no terms found")
	if not raw_terms:
		raise ValueError(f"{instance_path}: only biases found, no couplings")

	#---- index base -------------------------------------------------------------
	min_idx, max_idx = min(all_idx), max(all_idx)

	if one_based is None:
		#A minimum index of 0 proves 0-based, since a 1-based file cannot contain 0. The
		#CONVERSE does not hold: a 0-based file whose lowest-numbered spin is isolated has
		#min_idx > 0. 86 of the 200 DCL_instances/{C8,C16} files are exactly that, so
		#`not (min_idx == 0)` silently renumbered every spin and lost the last one - the
		#`n_seen > N` guard below cannot catch it, because n_seen comes out too SMALL.
		#With N known the base is decidable from the top index instead; otherwise refuse.
		if min_idx == 0:
			one_based = False

		elif N is not None and max_idx == N:
			one_based = True

		elif N is not None and max_idx == N - 1:
			one_based = False

		else:
			raise ValueError(
				f"{instance_path}: cannot infer the index base (min index {min_idx}, max index "
				f"{max_idx}, N={N}). A minimum index above 0 is consistent with both a 1-based "
				f"file and a 0-based file whose lowest spin is isolated. Pass one_based explicitly."
			)

	if one_based and min_idx < 1:
		raise ValueError(f"{instance_path}: declared 1-based but contains index {min_idx}")

	shift  = 1 if one_based else 0
	n_seen = max_idx - shift + 1

	if N is None:
		N = n_seen
	elif n_seen > N:
		raise ValueError(
			f"{instance_path}: index {max_idx} exceeds N={N} under "
			f"{'1' if one_based else '0'}-based indexing. Pass one_based explicitly if the base was misdetected."
		)

	#---- accumulate; a repeat must restate the SAME value, never add to it -------
	#J_ij and J_ji are one coupling, so a term written twice is a redundant restatement
	#of a single fact, not a second interaction - summing it would invent a coupling that
	#is not in the file. Generalizes to any arity: a term's identity is its index SET,
	#since s_i*s_j*s_l is permutation-invariant. Equal values are accepted and stored
	#once; unequal values mean an asymmetric coupling matrix, which is not a valid Ising
	#problem, so they raise. On every consistent file this now agrees exactly with
	#Nonlocal-Monte-Carlo's txt_to_A_droplet, which assigns (W[i,j] = W[j,i] = v) rather
	#than accumulating; the two differ only on files this rejects outright.
	h_seen: Dict[int, Tuple[float, int]] = {}
	h = np.zeros(N, dtype = np.float64)

	for i, v, lineno in raw_bias:
		if i in h_seen:
			v_prev, lineno_prev = h_seen[i]
			if v != v_prev:
				raise ValueError(
					f"{instance_path}:{lineno}: bias on spin {i} restated as {v}, but line "
					f"{lineno_prev} gives {v_prev}. A repeated entry must restate the same value."
				)
			continue

		h_seen[i] = (v, lineno)
		h[i - shift] = v

	by_order: Dict[int, Dict[tuple, float]] = {}
	seen_at: Dict[tuple, Tuple[float, int]] = {}

	for idx, v, lineno in raw_terms:
		k = len(idx)
		key = tuple(i - shift for i in idx)

		if key in seen_at:
			v_prev, lineno_prev = seen_at[key]
			if v != v_prev:
				raise ValueError(
					f"{instance_path}:{lineno}: term {idx} restated as {v}, but line "
					f"{lineno_prev} gives {v_prev}. J is symmetric, so a repeated term must "
					f"restate the same value; unequal values are an asymmetric coupling matrix."
				)
			continue

		seen_at[key] = (v, lineno)
		by_order.setdefault(k, {})
		by_order[k][key] = v

	terms = {}
	for k in sorted(by_order):
		items = sorted(by_order[k].items())
		terms[k] = (np.array([i for i, _ in items], dtype = np.int64),
								np.array([v for _, v in items], dtype = np.float64))

	return terms, h, N


def ising_txt_for_jax(instance_path: str,
											dtype,
											one_based: Any = None,
											N: Any = None,
											negate: bool = True) -> Jhdata:
	"""
	Build a Jhdata for problem_mode "pising" from an arbitrary-order Ising text instance.

	Mirrors wcnf_for_jax's output contract, but indices are plain unsigned 0-based spin
	positions - there is no polarity to pack, so Jat carries no trailing sign column and
	its last axis is k-1 rather than k.

	SIGN CONVENTION (negate). These instance files state the Hamiltonian as
		E = + sum J*prod(s) + sum h*s
	while get_energy computes the opposite sign,
		E = - (sum J*prod(s) + sum h*s)
	which is the convention the NMC class uses. The reference driver bridges the two
	explicitly - NMC/examples/chimera_example.py:45-46 does `J = -J; h = -h  # match the
	sign of Hamiltonian` right after loading - and that step has no counterpart here, so
	without it the sampler MINIMIZES the negated Hamiltonian, i.e. it searches for the
	highest-energy state of the published problem. Verified against the shipped ground
	states: chimera128 001-004 evaluate to +210.93/+216.91/+226.43/+209.52 unnegated,
	against published values of exactly the negatives of those.

	Note this is not caught by comparing loader against loader (the two agree on the
	unnegated arrays); it only shows up against a published ground-state energy.

	Set negate = False only for a file already written in the internal -(...) convention.
	"""

	terms, h_np, N = read_ising_txt(instance_path, one_based = one_based, N = N)

	if negate:
		h_np = -h_np
		terms = {k: (idx, -val) for k, (idx, val) in terms.items()}

	J_v = []
	Jat = []

	for k in sorted(terms):
		idx, val = terms[k]

		J_v.append(Jsp(jnp.array(val, dtype = dtype),
									 jnp.array(idx, dtype = jnp.int32)))

		#per-spin gather: for each term touching i, the OTHER k-1 spins and the coefficient
		counts = np.bincount(idx.reshape(-1), minlength = N)
		maxlen = int(counts.max())

		data    = np.zeros((N, maxlen),        dtype = np.float64)
		indices = np.zeros((N, maxlen, k - 1), dtype = np.int64)
		fill    = np.zeros(N,                  dtype = np.int64)

		for t in range(idx.shape[0]):
			members = idx[t]
			for ci in range(k):
				i = members[ci]
				p = fill[i]
				indices[i, p, :] = np.delete(members, ci)
				data[i, p]       = val[t]
				fill[i]          = p + 1

		#padding slots keep data = 0: Jxs_de_pising reads them with fill_value=True, whose
		#parity product is +1, so a nonzero weight there would add a phantom term
		Jat.append(Jsp(jnp.array(data, dtype = dtype),
									 jnp.array(indices, dtype = jnp.int32)))

	return Jhdata(J_v, jnp.array(h_np, dtype = dtype), Jat)


import itertools

def test_coloring_anygraph(
		indices: np.ndarray, 
		colors: np.ndarray,
	):

	edges = []
	for factor in indices:
		for e in itertools.combinations(factor, 2):
			edges.append(e)

	edges = np.unique(np.sort(np.array(edges), axis = 1), axis = 0)

	for e in edges:
		if colors[e[0]] == colors[e[1]]:
			raise RuntimeError("Wrong graph coloring! Neigbours have same color")

def get_coloring_anygraph(
		indices: np.ndarray, 
		N: int, 
		max_colors: int, 
		time_limit: int = 60
	) -> np.ndarray:

	#OUT OF SCOPE. The Gurobi ILP solves MINIMUM graph coloring exactly - an NP-hard
	#preprocessing step for a heuristic sampler, needing a commercial license and capped
	#at a time limit it routinely hit. problems/coloring.py's greedy DSATUR replaces it:
	#polynomial, license-free, and measured at fewer colors anyway (23.64 vs 25.0 mean
	#over the 64 sf250 instances). Kept only so the existing `<instance>_col.txt` ILP
	#colorings on disk stay interpretable and can serve as a reference when validating
	#DSATUR's output - nothing in a live path may call this.
	raise NotImplementedError(
		"get_coloring_anygraph (Gurobi ILP) is out of scope - use problems.coloring."
		"get_coloring, which needs no license. Delete this raise only to regenerate an "
		"ILP baseline for comparison against DSATUR."
	)

	#imported lazily: superseded by coloring.get_coloring, and a top-level import
	#would make this whole module unimportable without a Gurobi license
	import gurobipy as gp
	from gurobipy import GRB

	assert N >= jnp.max(indices).item() and "Too many indices: > N"

	m = gp.Model()
	
	# Create variable for each node and (possible) color class
	x = m.addVars(N, max_colors, vtype = GRB.BINARY)

	# Create variable for each (possible) color class
	y = m.addVars(max_colors, vtype=GRB.BINARY)

	# Objective: minimize number of colors used
	m.setObjective(gp.quicksum(y[j] for j in range(max_colors)), GRB.MINIMIZE)

	# Constraints: suppose that only the smallest colors are used
	m.addConstrs(y[j] >= y[j+1] for j in range(max_colors - 1))

	# Constraints: must assign each vertex i to some color 
	m.addConstrs(gp.quicksum(x[i, j] for j in range(max_colors)) == 1 for i in range(N))

	edges = []
	for factor in indices:
		for e in itertools.combinations(factor, 2):
			edges.append(e)

	edges = np.unique(np.sort(np.array(edges), axis = 1), axis = 0)

	# Constraints: cannot assign endpoints {u,v} of an edge to same color
	m.addConstrs(x[u, j] + x[v, j] <= y[j] for u, v in edges for j in range(max_colors))
	
	# Constraints: cannot assign i->j when j is not a used color 
	m.addConstrs(x[i,j] <= y[j] for i in range(N) for j in range(max_colors))

	#set time-limit on Gurobi				
	m.setParam('TimeLimit', time_limit)
	# Solve
	m.optimize()

	coloring = [-1 for i in range(N)]

	for i in range(N):
		for j in range(max_colors):
				if x[i,j].x > 0.5:
						coloring[i] = j
	
	assert np.all(np.array(coloring) != -1) and "Coloring failed! Some variables unassigned"
	
	return np.array(coloring)

