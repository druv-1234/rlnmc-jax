import sys
sys.path.append('.')

import jax
import jax.numpy as jnp
from jax import lax

from flax import struct

from gymnax.environments import environment

from modules import nmc
from modules import lbp

from pgmax import infer

from functools import partial

from typing import Dict, Tuple, Union, NamedTuple

#NMC environment setup
class NMCSetup(NamedTuple):
	approximation: float
	
	track_stats: bool
	problem_mode: str

	coloring: bool

	energy_scale: float

	reward_scale: float

	numerical_eval_energy: bool

	use_lbp: bool

	lbp_beta: float

@struct.dataclass
class NMCParams:	
	#maximum environment steps
	episode_steps: int = 100

	#whether to use nmc jumps (running SA, if false)
	use_nmc: bool = struct.field(default = True, 
													  	 pytree_node=False)
	
	#number of sweeps of the environment reset
	nsw_reset: int = struct.field(default = 100, pytree_node=False)

	#number of nmc cycles of the nonequilibrium phase
	nmc_neq_cycles: int = struct.field(default = 10, pytree_node=False)
	#number of nmc sweeps of the nonequilibrium cycle
	nmc_nsw_neq_phase: int = struct.field(default = 1, pytree_node=False)

	#number of nmc sweeps of the equilibrium phase
	nmc_nsw_eq_phase: int = struct.field(default = 1, pytree_node=False)

	#highest temperature of the nonequilibrium sampling
	nmc_neq_beta: float = 0.1

	#SA linear inverse temperature schedule
	beta_i: float = 1.0
	beta_f: float = 8.0

	#starting temperature of the episodes
	beta_start: float = 5.0

	#loopy belief propagation setup
	# (fixed) temperature of LBP
	lbp_beta: float = struct.field(default = 1.0, pytree_node=False)
	# messages tolerance
	lbp_tolerance_m: float = 1e-3	
	# distance to reference tolerance
	lbp_tolerance_d: float = 0.01	
	# surrogate hamiltonian control of localization
	lbp_lambdas: jax.Array = struct.field(default_factory = lambda: jnp.array([0.1, 0.01, 0.5]))
	# number of iterations for each lambda
	lbp_num_iters: int = struct.field(default = 100, pytree_node=False)


@struct.dataclass
class EnvObs:
	s: jax.Array

	mag: jax.Array

	norm_time: jax.Array
	norm_best_e: jax.Array
	norm_beta: jax.Array


@struct.dataclass
class EnvState:
	time: int

	s: jax.Array
	e: jax.Array

	best_s: jax.Array
	best_e: jax.Array

	beta: jax.Array

from modules.nmc_types import *
      
	
class NMCgym(environment.Environment):
	"""
	JAX compatible NMC environment used for both NSA and RLNSA algorithms
	"""

	def __init__(self,
							 Jh_stack: Jhdata,
							 Jh_cat: Jhdata,
							 ground_states: Union[jax.Array, None], 
							 colors_stack: Union[jax.Array, None],
							 nmc_setup: NMCSetup, 
							 FG):

		super().__init__()
	
		self.Jh_stack = Jh_stack
		self.Jh_cat = Jh_cat
		if self.Jh_stack.h.ndim == 1:
			raise RuntimeError("Please provide Jh_stacked")
		
		self.n_instances = self.Jh_stack.h.shape[0]
	
		self.ground_states = ground_states
		
		#graph coloring of the problem used for parallel flipping
		self.colors_stack = colors_stack

		self.nmc_setup = nmc_setup

		self.N0 = self.Jh_stack.h[0].size
		self.N = self.Jh_cat.h.size

		#initialize the lbp of the factor graph and the localization epsilon
		if self.nmc_setup.use_lbp:
			if FG is not None:
				print("Factor graph provided")
				self.FG = FG
				
			else:
				print("Computing the factor graph...")
				self.FG = lbp.init_factor_graph(self.Jh_cat.J, 
																				self.Jh_cat.h, 
																				self.nmc_setup.lbp_beta, 
																				self.nmc_setup.problem_mode)
																			
			self.BP = infer.build_inferer(self.FG.bp_state, backend="bp")
			print("Factor graph initialized")

			self.epsilon = \
				jnp.sum(jnp.concatenate([jnp.sum(jnp.abs(j.data), axis = 1, keepdims = True) 
																	for j in self.Jh_cat.Jat], axis = 1), axis = 1)
			self.epsilon += jnp.abs(self.Jh_cat.h)
	

	@partial(jax.jit, static_argnums = (0))
	def get_geometry(self, 
									 state: jax.Array, 
									 params):

		if not self.nmc_setup.use_lbp: 
			de_vector = jax.vmap(jax.vmap(nmc.get_de, 
																		in_axes = (0, 0, 0, None, None)), 
														in_axes = (0, 0, 0, 0, None))(
				self.Jh_stack.Jat, 
				self.Jh_stack.h, 
				state.reshape(self.n_instances, -1), 
				state.reshape(self.n_instances, -1), 
				self.nmc_setup.problem_mode
			).reshape(self.n_instances, -1)

			#approximate magnetizations by simple local fields								
			magnetizations = jnp.where(de_vector <= 0, jnp.array(0, dtype = de_vector.dtype), de_vector/2)

			if self.nmc_setup.track_stats:
				geometry_stats = {
					"de": {
						"min":  jnp.min(magnetizations,  axis = 1).mean(), #type: ignore
						"max":  jnp.max(magnetizations,  axis = 1).mean(), #type: ignore
						"mean": jnp.mean(magnetizations, axis = 1).mean(), #type: ignore
						"std":  jnp.std(magnetizations,  axis = 1).mean()	#type: ignore
					}
				}
			else:
				geometry_stats = {}

		else:
			def run_surrogate_LBP(s: jax.Array):
				beliefs, lbp_stats = lbp.surrogate_LBP(self.BP, 
																							self.FG.variable_groups[0],
																							s,
																							self.epsilon, 
																							params.lbp_beta,
																							params.lbp_num_iters,
																							self.nmc_setup.problem_mode, 
																							params.lbp_tolerance_m*params.lbp_beta,
																							params.lbp_tolerance_d, 
																							params.lbp_lambdas)

				#magnetizations from the surrogate hamiltonian
				mag = (
					(beliefs[self.FG.variable_groups[0]][:, 1] - beliefs[self.FG.variable_groups[0]][:, 0])
				)/params.lbp_beta/2

				#correct the final surrogate part
				if self.nmc_setup.problem_mode in ["pubo", "pising"]:
					mag = jnp.abs(mag - lbp_stats['final_lambda']*(2*s - 1)*self.epsilon)
				else:
					mag = jnp.abs(mag - lbp_stats['final_lambda']*(s - 0.5)*self.epsilon)
				
				return mag, lbp_stats
			
			
			magnetizations, lbp_stats = run_surrogate_LBP(state)
			magnetizations = magnetizations.reshape(self.n_instances, self.N0)

			if self.nmc_setup.track_stats:
				geometry_stats = lbp_stats
			else:
				geometry_stats = {}

		return magnetizations, geometry_stats


	@partial(jax.jit, static_argnums=(0,))
	def step(self,
					 key: jax.Array,
					 state: EnvState,
					 action: jax.Array,
					 params: NMCParams):
		"""
		Performs the NMC step of the environment
		"""
		return self.step_env(key, state, action, params)

	#jit compiled by step of the parent class
	def step_env(self,
							 key,
							 state: EnvState,
							 action: jax.Array,
							 params: NMCParams):
		"""
		Performs the NMC step of the environment;
		implements the logic of the NMC cycles and the surrogate loopy belief propagation calculations
		"""
		instance_keys = jax.random.split(key, self.n_instances)

		old_best_e = state.best_e

		s = state.s.reshape(self.n_instances, self.N0)
		e = state.e

		best_s = state.best_s.reshape(self.n_instances, self.N0)
		best_e = state.best_e

		beta_range_step = (params.beta_f - params.beta_start)/params.episode_steps

		if params.use_nmc:
			action = action.reshape(self.n_instances, self.N0)
		
		total_sweeps_per_step = params.nmc_nsw_eq_phase

		if params.use_nmc:
			sweeps_per_neq = params.nmc_neq_cycles*params.nmc_nsw_neq_phase
			total_sweeps_per_step += sweeps_per_neq
			
			#the fraction of time allocated to the non-equilibrium sampling
			beta_range_neq = beta_range_step*sweeps_per_neq/(total_sweeps_per_step + 1e-9)
		
		seminkey = Seminkey(s, s, e, e, instance_keys)

		def nmc_cycle(carry, i_cycle):
			(
				seminkey, 
				best_s,
				best_e,
				beta
			) = carry
			
			def nmc_jump(
				Jh, 
				sekey, 
				cluster_phase1, 
				cluster_phase2, 
				beta_phase1, 
				beta_phase2, 
				colors
			):
				cluster_beta_phase1 = jnp.where(cluster_phase1, beta_phase1, 0.0)
				cluster_beta_phase2 = jnp.where(cluster_phase2, beta_phase2, 0.0)

				if not self.nmc_setup.coloring:
					return nmc.nmc_sampling(
						Jh, 
						sekey, 
						cluster_beta_phase1, 
						cluster_beta_phase2,

						params.nmc_nsw_neq_phase,
						params.nmc_nsw_neq_phase,
						
						self.nmc_setup.problem_mode, 
						self.nmc_setup.track_stats
					)
				else:
					return nmc.nmc_sampling_colored(
						Jh, 
						sekey, 
						cluster_beta_phase1, 
						cluster_beta_phase2,

						colors,

						params.nmc_nsw_neq_phase,
						params.nmc_nsw_neq_phase,
						
						self.nmc_setup.problem_mode, 
						self.nmc_setup.track_stats
					)

			cycle_s, cycle_e = seminkey.s, seminkey.e

			cycle_min_s = seminkey.min_s
			#reset the initial min energy to arrive at a completely new backbone steady state
			cycle_min_e = lax.select(i_cycle == 0, 
															 jnp.full_like(seminkey.min_e, jnp.inf),
															 seminkey.min_e)

			nmc_neq_beta = jnp.exp(
				jnp.log(params.nmc_neq_beta) + 
				jnp.log((state.beta + beta_range_neq)/params.nmc_neq_beta)*i_cycle/params.nmc_neq_cycles
			)

			seminkey, stats_neq = jax.vmap(nmc_jump, in_axes = (0, 0, 0, 0, None, None, 0))(
				self.Jh_stack, 

				Sekey(seminkey.s, 
							seminkey.e, 
							seminkey.key), 

				action,
				jnp.logical_not(action), 
				
				nmc_neq_beta,
				beta,
				
				self.colors_stack
			)
			
			beta = beta + beta_range_neq/params.nmc_neq_cycles

			def check_energies(seminkey,
												 cycle_min_s, cycle_min_e, 
												 best_s, best_e):
				
				min_s, min_e = lax.cond(i_cycle == 0, 
																lambda: (seminkey.s, seminkey.e), 
																lambda: (seminkey.min_s, seminkey.min_e))
				
				e_isclose = jnp.logical_or(jnp.isclose(min_e, cycle_min_e, atol = 0), 
																	 jnp.isclose(cycle_min_e, min_e, atol = 0))
				# if equal is also fine, because we need to explore
				min_s, min_e = lax.cond(jnp.logical_or(min_e < cycle_min_e, e_isclose),
																lambda: (min_s, min_e), 
																lambda: (cycle_min_s, cycle_min_e))
				
				e_isclose = jnp.logical_or(jnp.isclose(min_e, best_e, atol = 0), 
																	 jnp.isclose(best_e, min_e, atol = 0))
				best_s, best_e = lax.cond(jnp.logical_and(min_e < best_e, jnp.logical_not(e_isclose)),
																	lambda: (min_s, min_e), 
																	lambda: (best_s, best_e))
				
				return (
					Seminkey(seminkey.s, min_s, seminkey.e, min_e, seminkey.key), 
					best_s, best_e
				)

			seminkey, best_s, best_e = jax.vmap(check_energies)(
				seminkey,
				cycle_min_s, cycle_min_e,
				best_s, best_e
			)


			mc_stats = {}
			if self.nmc_setup.track_stats:
				mc_stats["neq"] = stats_neq

				mc_stats["cycle_jump"] = (cycle_s != seminkey.s).mean(axis = 1)
				mc_stats["cycle_de"] = seminkey.e - cycle_e	

				mc_stats["cycle_min_jump"] = (cycle_min_s != seminkey.min_s).mean(axis = 1)
				mc_stats["cycle_min_de"] = lax.select(i_cycle == 0, seminkey.min_e - cycle_e, seminkey.min_e - cycle_min_e)	
			
			carry = (seminkey, best_s, best_e, beta)

			return carry, mc_stats


		old_s = s
		old_e = e

		if params.use_nmc:
			#perform nmc_cycles of the nonlocal monte carlo sweeps
			(seminkey, best_s, best_e, beta), mc_stats = \
				lax.scan(nmc_cycle, (seminkey, best_s, best_e, state.beta), xs = jnp.arange(params.nmc_neq_cycles))
		else:
			beta = state.beta
			mc_stats = {}

		beta_a = beta
		beta = params.beta_start + beta_range_step*(state.time + 1)

		def eq_phase(Jh, seminkey, best_s, best_e, old_s, old_e, colors):
			seminkey_pre_eq = seminkey
			
			#when nmc is not allowed, use the s state, not min_s
			if params.use_nmc:
				mcmc_s, mcmc_min_s, mcmc_e, mcmc_min_e = (seminkey.min_s, seminkey.min_s, seminkey.min_e, seminkey.min_e)
			else:
				mcmc_s, mcmc_min_s, mcmc_e, mcmc_min_e = (seminkey.s, seminkey.min_s, seminkey.e, seminkey.min_e)

			if not self.nmc_setup.coloring:
				seminkey, eq_stats = nmc.mcmc_sampling(
					Jh, 
					Seminkey(mcmc_s, mcmc_min_s, mcmc_e, mcmc_min_e, seminkey.key), 
					jnp.array([beta_a, beta]), 
					params.nmc_nsw_eq_phase, 
					self.nmc_setup.problem_mode, 
					anneal = True, 
					totrack = self.nmc_setup.track_stats
				)

			else:
				seminkey, eq_stats = nmc.mcmc_sampling_colored(
					Jh, 
					Seminkey(mcmc_s, mcmc_min_s, mcmc_e, mcmc_min_e, seminkey.key), 
					jnp.array([beta_a, beta]), 
					colors,
					params.nmc_nsw_eq_phase, 
					self.nmc_setup.problem_mode, 
					anneal = True, 
					totrack = self.nmc_setup.track_stats
				)
				
			if self.nmc_setup.numerical_eval_energy:
				e = nmc.get_energy(Jh, seminkey.s, self.nmc_setup.problem_mode)
				min_e = nmc.get_energy(Jh, seminkey.min_s, self.nmc_setup.problem_mode)
				best_e = nmc.get_energy(Jh, best_s, self.nmc_setup.problem_mode)
			else:
				e = seminkey.e
				min_e = seminkey.min_e
		
			e_isclose = jnp.logical_or(jnp.isclose(e, min_e, atol = 0), jnp.isclose(min_e, e, atol = 0))
			min_s, min_e = lax.cond(jnp.logical_or(e < min_e, e_isclose),
															lambda: (seminkey.s, e), 
															lambda: (seminkey.min_s, min_e))
			
			e_isclose = jnp.logical_or(jnp.isclose(min_e, best_e, atol = 0), jnp.isclose(best_e, min_e, atol = 0))
			best_s, best_e = lax.cond(jnp.logical_and(min_e < best_e, jnp.logical_not(e_isclose)),
																lambda: (min_s, min_e), 
																lambda: (best_s, best_e))
			
			total_stats = {}
			if self.nmc_setup.track_stats:
				#pre-unlearning stats
				if params.use_nmc:
					total_stats["neq_jump"] = (old_s != seminkey_pre_eq.s).sum()/old_s.size
					total_stats["neq_de"] = seminkey_pre_eq.e - old_e

					total_stats["neq_min_jump"] = (old_s != seminkey_pre_eq.min_s).sum()/old_s.size
					total_stats["neq_min_de"] = seminkey_pre_eq.min_e - old_e	

				#after unlearning stats
				total_stats["jump"] = (old_s != seminkey.s).sum()/old_s.size
				total_stats["de"] = e - old_e	

				total_stats["min_jump"] = (old_s != min_s).sum()/old_s.size	
				total_stats["min_de"] = min_e - old_e	

			if params.use_nmc:
				key, subkey = jax.random.split(seminkey.key)
				log_r = jnp.log(jax.random.uniform(subkey, ()))

				#apply the Metropolis rule for stability of sampling
				min_de = min_e - old_e	
				(final_s, final_e), ar = lax.cond(log_r < -min_de*beta, 
																					lambda: ((min_s, min_e), 1.0), 
																					lambda: ((old_s, old_e), 0.0))

				if self.nmc_setup.track_stats:
					total_stats["metropolis_ar"] = ar
					total_stats["metropolis_jump"] = (old_s != final_s).sum()/old_s.size	
					total_stats["metropolis_de"] = final_e - old_e
			
			else:
				#use normal states for the case when nmc is not allowed
				final_s, final_e = seminkey.s, e
				key = seminkey.key

			return (
				Seminkey(final_s, final_s, final_e, final_e, key), 
				best_s, 
				best_e, 
				eq_stats, 
				total_stats
			)

		seminkey, best_s, best_e, eq_stats, total_stats = \
			jax.vmap(eq_phase)(
				self.Jh_stack, 
				seminkey,
				best_s, best_e,
				old_s, old_e,
				self.colors_stack
			)
		
		if self.nmc_setup.track_stats:
			if params.use_nmc:
				#average the stats over the cycles and instances
				mc_stats_cmean = jax.tree.map(lambda x: x.mean(axis = 0).mean(), mc_stats)

				#first cycle averaged over instances
				mc_stats_first = jax.tree.map(lambda x: x[0].mean(), mc_stats)

				#last cycle averaged over instances
				mc_stats_last = jax.tree.map(lambda x: x[-1].mean(), mc_stats)

				mc_stats = {}
				mc_stats["cycle_mean"] = mc_stats_cmean
				mc_stats["cycle_first"] = mc_stats_first
				mc_stats["cycle_last"] = mc_stats_last

			mc_stats["eq"] = jax.tree.map(lambda x: x.mean(), eq_stats)
			mc_stats["total"] = jax.tree.map(lambda x: x.mean(), total_stats)

		else:
			mc_stats = {}


		state = EnvState(
			time = state.time + 1,
			s = seminkey.s.reshape(-1),
			e = seminkey.e,
			best_s = best_s.reshape(-1),
			best_e = best_e, 
			beta = jnp.array(beta)
		)

		done = state.time >= params.episode_steps

		if self.ground_states is None:
			best_e_improvement = jnp.where(
				old_best_e - best_e > 0, 
				(old_best_e - best_e).astype(jnp.float32),
				jnp.zeros_like(best_e, dtype = jnp.float32)
			)

		else:
			old_best_e = (old_best_e - self.ground_states)/self.N0
			new_best_e = (best_e - self.ground_states)/self.N0

			#clip the energy by the approximation ratio
			ar = jnp.full_like(new_best_e, self.nmc_setup.approximation/self.N0)
			new_best_e_ar = jnp.stack((new_best_e, ar)).max(axis = 0)

			best_e_improvement = jnp.where(
				old_best_e - new_best_e > 0, 
				(old_best_e - jnp.min(jnp.stack((new_best_e_ar, old_best_e)), axis = 0)).astype(jnp.float32),
				jnp.zeros_like(best_e, dtype = jnp.float32)
			)
		
		reward = best_e_improvement #type: ignore

		magnetizations, geometry_stats = self.get_geometry(state.s, params)

		return (
			self.get_obs(
				state, 
				magnetizations,
				params
			),
			
			state,
			jnp.array(reward, dtype = jnp.float32)/self.nmc_setup.reward_scale,

			done,
			{
				"geometry": geometry_stats, 
				"mc": mc_stats
			}
		)


	@partial(jax.jit, static_argnums=(0,))
	def reset(self, 
					  key: jax.Array, 
						params: NMCParams) -> Tuple[EnvObs, EnvState, Dict]:
		"""
		Performs the NMC reset of the environment (higher level)
		"""				
		return self.reset_env(key, params)

	
	def reset_env(self, 
								key: jax.Array, 
								params: NMCParams) -> Tuple[EnvObs, EnvState, Dict]:
		"""
		Performs the NMC reset of the environment (lower level)
		"""
		init_key, *instance_keys = jax.random.split(key, self.n_instances + 1)

		init_s = jax.random.choice(init_key, jnp.array([False, True]), 
															 shape = (self.n_instances, self.N0))
		
		def get_seminkey(Jh, s, key, colors):
			if self.nmc_setup.problem_mode == "pubo":
				e = -nmc.Jxs_pubo(Jh.J, Jh.h, s)

			elif self.nmc_setup.problem_mode == "pising":
				e = -nmc.Jxs_pising(Jh.J, Jh.h, s)

			else:
				e = -nmc.Jxs_wcnf(Jh.J, Jh.h, s)
			
			beta_schedule = jnp.array([params.beta_i, params.beta_start])

			if not self.nmc_setup.coloring:
				seminkey, stats = \
					nmc.mcmc_sampling(
						Jh,
						Seminkey(s, s, e, e, key),
						beta_schedule, 
						params.nsw_reset, 
						self.nmc_setup.problem_mode, 
						anneal = True, 
						totrack = self.nmc_setup.track_stats
					)

			else:
				seminkey, stats = \
					nmc.mcmc_sampling_colored(
						Jh,
						Seminkey(s, s, e, e, key),
						beta_schedule, 
						colors,
						params.nsw_reset, 
						self.nmc_setup.problem_mode, 
						anneal = True, 
						totrack = self.nmc_setup.track_stats
					)
			
			return seminkey, stats
		

		#compute the energy and run the initial mcmc sweeps 
		seminkey_init, annealing_stats = jax.vmap(get_seminkey)(
			self.Jh_stack, 
			init_s, 
			jnp.stack(instance_keys),
			self.colors_stack
		)

		best_s_init = seminkey_init.min_s
		best_e_init = seminkey_init.min_e

		state = EnvState(
			time = 0,
			s = seminkey_init.s.reshape(-1),
			e = seminkey_init.e,
			best_s = best_s_init.reshape(-1),
			best_e = best_e_init,

			beta = jnp.array(params.beta_start)
		)

		magnetization, geometry_stats = self.get_geometry(state.s, params)

		return (
			self.get_obs(
				state, 
				magnetization,
				params
			), 
			state,
			{
				"annealing_stats": annealing_stats, 
				"geometry": geometry_stats
			}	
		)

	@partial(jax.jit, static_argnums = (0))
	def get_obs(self, 
						 	state: EnvState, 
							magnetization: jax.Array, 
							params: NMCParams) -> EnvObs:
		"""
		Computes observations that can potentially be passed to the recurrent policy
		"""
		
		if self.ground_states is not None:
			best_e = state.best_e - self.ground_states
		else:
			best_e = state.best_e

		return EnvObs(
			s = state.s,
			
			mag = magnetization.reshape(-1),

			norm_time = jnp.array(state.time/params.episode_steps),
			norm_best_e = best_e/self.nmc_setup.energy_scale,
			norm_beta = state.beta/params.beta_f
		)

	@property
	def name(self) -> str:
		return "RLNMC-v2"


class GymnaxWrapper(object):
  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

@struct.dataclass
class LogState:
  env_state: EnvState

  episode_returns: float
  episode_lengths: int

  returned_episode_returns: float
  returned_episode_lengths: int

  timestep: int


class LogWrapper(GymnaxWrapper):
	def __init__(self, env: NMCgym):
			super().__init__(env)

	@partial(jax.jit, static_argnums = (0,))
	def reset(self, 
						key: jax.Array, 
						params: NMCParams) -> Tuple[EnvObs, LogState, Dict]:

		obs, env_state, stats = self._env.reset(key, params)
		
		n_instances = env_state.e.size
		state = LogState(
			env_state, 
			episode_returns = jnp.zeros((n_instances,), dtype = jnp.float32), #type: ignore
			episode_lengths = 0,
			returned_episode_returns = jnp.zeros((n_instances,), dtype = jnp.float32), #type: ignore
			returned_episode_lengths = 0,
			timestep = 0
		)

		return obs, state, stats
	
	@partial(jax.jit, static_argnums = (0))
	def step(self,
					 key: jax.Array,
					 state: LogState,
					 action: jax.Array,
					 params: NMCParams) -> Tuple[EnvObs, LogState, jax.Array, jax.Array, Dict]:

		obs, env_state, reward, done, info = \
			self._env.step(key, state.env_state, action, params)
		
		n_instances = reward.size
		
		new_episode_return = state.episode_returns + reward
		new_episode_length = state.episode_lengths + 1

		done_all = jnp.full((n_instances,), done)

		state = LogState(
				env_state = env_state,

				episode_returns = new_episode_return*(1 - done),
				episode_lengths = new_episode_length*(1 - done),
				
				returned_episode_returns = state.returned_episode_returns*(1 - done)
					+ new_episode_return*done,
				returned_episode_lengths = state.returned_episode_lengths*(1 - done)
					+ new_episode_length*done,

				timestep = state.timestep + 1
		)

		info["episode_returns"] = state.episode_returns
		info["episode_lengths"] = state.episode_lengths

		info["returned_episode_returns"] = state.returned_episode_returns
		info["returned_episode_lengths"] = state.returned_episode_lengths
		info["returned_episode"] = done

		info["timestep"] = state.timestep

		#energies saved for logging
		info["energies"] = state.env_state.e
		info["best_energies"] = state.env_state.best_e

		return obs, state, reward, done, info