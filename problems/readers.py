from os import path 

import sys
sys.path.append('.')

import numpy as np
import jax.numpy as jnp

from modules.nmc_types import *

from typing import Dict, Any

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


import gurobipy as gp
from gurobipy import GRB
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

