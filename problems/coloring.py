"""
Greedy graph coloring of the interaction graph, for the colored MCMC/NMC kernels.

Spins sharing a color are pairwise non-adjacent, so they can be flipped in
parallel by `nmc.mcmc_sampling_colored` / `nmc.nmc_sampling_colored`.

This replaces the Gurobi ILP in `readers.get_coloring_anygraph`. Measured on the
shipped instances, DSATUR returns *fewer* colors than the ILP does, because the
ILP hits its 60s time limit well short of the optimum:

	N=250   gurobi 25.0   DSATUR 23.3   largest_first 25.3
	N=500   gurobi 31.5   DSATUR 28.5   largest_first 31.0
	N=1000  gurobi 30.0   DSATUR 28.5   largest_first 32.0
	N=2000  gurobi 30.0   DSATUR 29.0   largest_first 32.5

`num_colors` is the sequential dimension of the colored sweep scan, paid on every
sweep of every run, whereas the coloring is computed once and cached -- so DSATUR
is worth its higher preprocessing cost (~24s at N=2000 vs ~0.02s).
"""

import sys
sys.path.append('.')

import itertools
import os

import networkx as nx
import numpy as np

from typing import List, Union

DEFAULT_STRATEGY = "DSATUR"


def build_graph_from_factors(
		indices: Union[np.ndarray, List[np.ndarray]],
		N: int
	) -> nx.Graph:
	"""
	Build the interaction graph by clique-expanding k-body factors.

	Each factor of arity k contributes C(k, 2) edges, matching the edge set that
	`readers.get_coloring_anygraph` builds.

	Every node 0..N-1 is added before any edge, for two reasons: networkx only
	colors nodes that are present, and `read_utils.pad_and_stack_colors` infers
	N back out of the coloring as `colors_list[0].max() + 1` -- so a variable
	that appears in no clause, if dropped here, would silently shrink N for the
	whole stacked batch.

	:param indices: Factor index array of shape (num_factors, k), 0-based, or a
		list of such arrays for a problem with mixed arity.
	:param N: Number of variables. Passed explicitly, never inferred.
	:return: An undirected networkx Graph on nodes 0..N-1.
	"""
	#only an explicit list/tuple means "several arity blocks"; anything array-like
	#(np.ndarray, jax.Array, ...) is one block
	blocks = list(indices) if isinstance(indices, (list, tuple)) else [indices]

	G = nx.Graph()
	G.add_nodes_from(range(N))  # before edges: fixes node order, keeps isolated vars

	for block in blocks:
		block = np.atleast_2d(np.asarray(block))  # a lone factor is still (1, k)
		if block.size == 0:
			continue

		if block.min() < 0:
			#a negative index means an index-convention mismatch upstream (the wcnf
			#abs()-1 applied to already-0-based indices), not an out-of-range factor.
			#Caught here because networkx would happily add a node -1 and the failure
			#would surface later as "coloring missing node N", pointing at the wrong end.
			raise ValueError(
				f"Negative factor index {block.min()}: indices must be 0-based and unsigned. "
				"The wcnf abs()-1 convention must not be applied to pising/pubo indices."
			)

		if block.max() >= N:
			raise ValueError(f"Factor index {block.max()} out of range for N = {N}")

		for factor in block:
			G.add_edges_from(itertools.combinations(sorted(factor), 2))

	return G


def greedy_coloring(G: nx.Graph, strategy: str = DEFAULT_STRATEGY) -> np.ndarray:
	"""
	Color a graph greedily, returning one label per node.

	Deterministic for the strategies used here: networkx breaks DSATUR ties with
	max() over node iteration order, so a fixed construction order gives a fixed
	coloring. A randomized strategy would need an explicit seed.

	:param G: An undirected networkx Graph on nodes 0..N-1.
	:param strategy: A networkx greedy_color strategy name.
	:return: np.ndarray of shape (N,) with contiguous labels 0..num_colors-1.
	"""
	N = G.number_of_nodes()
	coloring = nx.coloring.greedy_color(G, strategy = strategy)

	missing = [i for i in range(N) if i not in coloring]
	if missing:
		raise RuntimeError(
			f"Coloring missing {len(missing)} nodes (first: {missing[:5]}); "
			"the graph must contain every node 0..N-1"
		)

	return np.array([coloring[i] for i in range(N)], dtype = int)


def validate_coloring(G: nx.Graph, labels: np.ndarray):
	"""
	Raise if any edge joins two nodes of the same color.

	:param G: The graph that was colored.
	:param labels: Color labels of shape (N,).
	"""
	for u, v in G.edges():
		if labels[u] == labels[v]:
			raise RuntimeError("Wrong graph coloring! Neigbours have same color")


def color_classes(labels: np.ndarray) -> np.ndarray:
	"""
	Group color labels into a padded index array.

	:param labels: Color labels of shape (N,).
	:return: np.ndarray of shape (num_colors, max_class_size) holding variable
		indices, right-padded with -1 (the fill value `pad_and_stack_colors`
		expects).
	"""
	num_colors = int(labels.max()) + 1
	classes = [np.where(labels == c)[0] for c in range(num_colors)]
	max_size = max(len(c) for c in classes)

	padded = np.full((num_colors, max_size), -1, dtype = int)
	for c, idx in enumerate(classes):
		padded[c, :len(idx)] = idx

	return padded


def cache_path_for(instance_path: str, strategy: str = DEFAULT_STRATEGY) -> str:
	"""
	Cache filename for a coloring, namespaced by strategy.

	Deliberately not the bare `_col.txt` used by the Gurobi path: keeping the
	names distinct leaves the existing ILP colorings on disk intact and lets the
	two be compared.

	:param instance_path: Instance path without extension.
	:param strategy: A networkx greedy_color strategy name.
	:return: Path to the cache file.
	"""
	return f"{instance_path}_col_{strategy.lower()}.txt"


def get_coloring(
		indices: Union[np.ndarray, List[np.ndarray]],
		N: int,
		strategy: str = DEFAULT_STRATEGY,
		cache_path: Union[str, None] = None,
		validate: bool = True
	) -> np.ndarray:
	"""
	Color the interaction graph of a k-body problem, using a disk cache.

	:param indices: Factor index array of shape (num_factors, k), 0-based, or a
		list of such arrays.
	:param N: Number of variables.
	:param strategy: A networkx greedy_color strategy name.
	:param cache_path: Cache file path. None disables caching.
	:param validate: Whether to check the coloring for conflicts.
	:return: np.ndarray of shape (N,) with contiguous labels 0..num_colors-1.
	"""
	if cache_path is not None and os.path.isfile(cache_path):
		labels = np.loadtxt(cache_path).astype(int).reshape(-1)

		#cache_path_for keys on instance path + strategy only - not on N, not on the factor
		#set - so a file written when this instance parsed to a different N is reused
		#silently. That matters beyond this call: bench.py derives n_colors from the array,
		#and read_utils.pad_and_stack_colors re-infers N for the WHOLE stacked batch as
		#colors_list[0].max() + 1, so a short file drops the tail spins of every instance
		#in the batch from every colored sweep. The run then completes normally, with a
		#worse energy and no error anywhere.
		if labels.size != N:
			raise RuntimeError(
				f"Cached coloring {cache_path} has {labels.size} labels but N = {N}. "
				"It was written for a different problem size - delete it and re-color."
			)

		return labels

	G = build_graph_from_factors(indices, N)
	labels = greedy_coloring(G, strategy = strategy)

	if validate:
		validate_coloring(G, labels)

	if cache_path is not None:
		np.savetxt(cache_path, labels, fmt = '%d')

	return labels
