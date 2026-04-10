import sys

sys.path.append('.')

import jax
import jax.numpy as jnp
from typing import Tuple, Dict

from flax import nnx

class ActorCritic(nnx.Module):
	def __init__(self, 
							 N: int,
							 ac_config: Dict,
							 rngs: nnx.Rngs):
		
		self.N = N
		
		self.din = ac_config["din"] # typicaly din = 2: magnetization and state
		self.dextra = ac_config["dextra"] #typically dextra = 3: time, beta, energy

		self.dembed = ac_config["dembed"]
		self.dgru = ac_config["dgru"]
		self.dout = ac_config["dout"]

		#local per-spin embedding
		self.input_local = nnx.Sequential(
			nnx.Linear(self.din, self.dembed, rngs = rngs),
			nnx.leaky_relu,
			nnx.Linear(self.dembed, self.dgru, rngs = rngs),
			nnx.LayerNorm(self.dgru, rngs = rngs)
		)

		#global embedding
		self.input_global = nnx.Sequential(
			nnx.Linear(self.dextra, self.dembed, rngs = rngs),
			nnx.leaky_relu,
			nnx.Linear(self.dembed, self.dgru, rngs = rngs),
			nnx.LayerNorm(self.dgru, rngs = rngs)
		)

		#local per-spin memory
		self.gru_cell_local = nnx.GRUCell(self.dgru, self.dgru, rngs = rngs)
		#global per-spin memory
		self.gru_cell_global = nnx.GRUCell(self.dgru, self.dgru, rngs = rngs)

		self.output = nnx.Sequential(
			nnx.Linear(self.dgru + self.dgru, self.dout, rngs = rngs),
			nnx.leaky_relu, 
			nnx.Linear(self.dout, 1, rngs = rngs)
		)

		self.output_value = nnx.Sequential(
			nnx.Linear(self.dgru, self.dout, rngs = rngs),
			nnx.leaky_relu, 
			nnx.Linear(self.dout, 1, rngs = rngs)
		)


	def __call__(self, 
							 h: Tuple[jax.Array, jax.Array],
							 x: jax.Array, 
							 x_global: jax.Array,
							 dones: jax.Array) -> Tuple[Tuple[jax.Array, jax.Array], jax.Array, jax.Array]:

		h_local, h_global = h

		seq_length = x.shape[0]
		batch_size = x.shape[1]
		K_all = x_global.shape[2]
		
		N0 = self.N//K_all

		def gru_scan_fn_with_reset(carry, cell, ins_reset):
			ins, reset = ins_reset
			rnn_state = jnp.where(reset[:, None, None], 
														cell.initialize_carry(carry.shape, nnx.Rngs(0)), 
														carry)
			return cell(rnn_state, ins)


		y_local = self.input_local(x)
		y_local = y_local.reshape((seq_length, batch_size, K_all, N0, -1))

		y_global = self.input_global(x_global)
		
		h_local, hs_local =\
			nnx.scan(gru_scan_fn_with_reset, 
							 in_axes = (nnx.Carry, None, 0), 
							 out_axes = (nnx.Carry, 0))(h_local, 
																					self.gru_cell_local, 
																					(y_local, dones[:, :, None]))
		hs_local = hs_local.reshape(seq_length, batch_size, self.N, -1)

		h_global, hs_global =\
			nnx.scan(gru_scan_fn_with_reset, 
							 in_axes = (nnx.Carry, None, 0), 
							 out_axes = (nnx.Carry, 0))(h_global, 
																					self.gru_cell_global, 
																					(y_global, dones))
		hs_global_tiled = jnp.tile(
			hs_global[:, :, :, None, :], (1, 1, 1, N0, 1)
		).reshape(seq_length, batch_size, self.N, -1)																			 
		
		z = jnp.concatenate((hs_local, hs_global_tiled), axis = 3)
		logit = self.output(z).squeeze(axis = 3)

		value = self.output_value(hs_global)

		return (h_local, h_global), logit, value.squeeze(axis = 3)