from os import path 

import sys
sys.path.append('.')

import numpy as np
import jax.numpy as jnp
from modules.nmc_types import *

from typing import List, Union

def pad_and_stack_Jh(Jh_list: List[Jhdata]) -> Jhdata:
	"""
	Pad the interaction matrices with zeros and stack them into a single Jhdata
	"""
	N = Jh_list[0].h.size
	assert jnp.all(jnp.array([Jh.h.size == N for Jh in Jh_list])), "Not all problems are of equal size"

	k_order = len(Jh_list[0].J)

	#max size of the sparse interaction matrices (J) of all orders
	J_sizes = [np.max([Jh.J[k].data.size for Jh in Jh_list]) for k in range(k_order)]

	#max size of the sparse interaction matrices for every index (Jat) of all orders 
	Jat_sizes = [np.max([Jh.Jat[k].data.shape[1] for Jh in Jh_list]) for k in range(k_order)]
	
	J = []
	h = jnp.stack([Jh.h for Jh in Jh_list])
	Jat = []
	for k in range(k_order):
		jsp = Jsp(jnp.stack([jax.lax.pad(Jh.J[k].data, jnp.array(0, dtype = Jh.J[k].data.dtype), 
																		 [(0, J_sizes[k] - Jh.J[k].data.shape[0], 0), ]) for Jh in Jh_list]),
							jnp.stack([jax.lax.pad(Jh.J[k].indices, 0, 
																		 [(0, J_sizes[k] - Jh.J[k].indices.shape[0], 0), (0, 0, 0)]) for Jh in Jh_list]))
		J.append(jsp)

		jat = Jsp(jnp.stack([jax.lax.pad(Jh.Jat[k].data, jnp.array(0, dtype = Jh.Jat[k].data.dtype), 
																		 [(0, 0, 0), (0, Jat_sizes[k] - Jh.Jat[k].data.shape[1], 0)]) for Jh in Jh_list]),
							jnp.stack([jax.lax.pad(Jh.Jat[k].indices, 0, 
																		 [(0, 0, 0), (0, Jat_sizes[k] - Jh.Jat[k].indices.shape[1], 0), (0, 0, 0)]) for Jh in Jh_list]))
		Jat.append(jat)

	return Jhdata(J, h, Jat)


def pad_and_concatenate_Jh(Jh_list: List[Jhdata], 
													 problem_mode) -> Jhdata:
	"""
	Pad the interaction matrices with zeros and cat them into a single Jhdata (not stacking)
	"""
	N = [Jh.h.size for Jh in Jh_list]

	p_order = len(Jh_list[0].J)

	if problem_mode in ["pubo", "pising"]:
		J = [Jsp(jnp.concatenate([Jh.J[p].data for Jh in Jh_list], axis = 0),
						 jnp.concatenate([Jh.J[p].indices + int(np.sum(N[:i])) for i, Jh in enumerate(Jh_list)], axis = 0)) 
				 for p in range(p_order)]
	else:
		J = [Jsp(jnp.concatenate([Jh.J[p].data for Jh in Jh_list], axis = 0),
						jnp.concatenate([Jh.J[p].indices + jnp.sign(Jh.J[p].indices)*int(np.sum(N[:i])) for i, Jh in enumerate(Jh_list)], axis = 0)) 
				for p in range(p_order)]

	#max size of the sparse interaction matrices for every index (Jat) of all orders 
	Jat_sizes = [np.max([Jh.Jat[p].data.shape[1] for Jh in Jh_list]) for p in range(p_order)]

	h = jnp.concatenate([Jh.h for Jh in Jh_list])
	
	Jat = []
	for p in range(p_order):
		jat_data = [jax.lax.pad(Jh.Jat[p].data, jnp.array(0, dtype = Jh.Jat[p].data.dtype), 
													  [(0, 0, 0), (0, Jat_sizes[p] - Jh.Jat[p].data.shape[1], 0)]) for Jh in Jh_list]
		jat_indices = [jax.lax.pad(Jh.Jat[p].indices, 0, 
															 [(0, 0, 0), (0, Jat_sizes[p] - Jh.Jat[p].indices.shape[1], 0), (0, 0, 0)]) for Jh in Jh_list]

		if problem_mode in ["pubo", "pising"]:
			Jat.append(Jsp(jnp.concatenate([jat_d for jat_d in jat_data], axis = 0), 
										 jnp.concatenate([jat_idx + int(np.sum(N[:i])) for i, jat_idx in enumerate(jat_indices)])))
		else:
			Jat.append(Jsp(jnp.concatenate([jat_d for jat_d in jat_data], axis = 0), 
										 jnp.concatenate([jat_idx + jnp.concatenate([jnp.sign(jat_idx[:, :, :-1])*int(np.sum(N[:i])), 
																											 					 jnp.zeros((N[i], Jat_sizes[p], 1), dtype = int)], axis = 2)
														for i, jat_idx in enumerate(jat_indices)], axis = 0)))

	return Jhdata(J, h, Jat)


def pad_and_stack_colors(colors_list: List[jax.Array]) -> jax.Array:
	#if an index in colors_list is equal to -1, then it signals the necessary padding

	#get the maximum spin index:
	N = colors_list[0].max() + 1
	assert jnp.all(jnp.array([color.max() + 1 == N for color in colors_list])) and "Not all problems are of equal size"

	max_colors = max([color.shape[0] for color in colors_list])
	max_color_size = max([color.shape[1] for color in colors_list])

	#graph coloring indices of all problems stacked, padded with N
	padded_colors = jnp.full((len(colors_list), max_colors, max_color_size), N, dtype = jnp.int32)
	for c, color in enumerate(colors_list):
		padded_colors = padded_colors.at[c, :color.shape[0], :color.shape[1]].set(color)
	
	return jnp.where(padded_colors != -1, padded_colors, N)