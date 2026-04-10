import sys
sys.path.append('.')

import jax
import jax.numpy as jnp
from jax import lax

from typing import Dict, Tuple, List
from functools import partial, reduce

from modules.nmc_types import *

#Energy functions
@jax.jit
def Jxs_pubo(J: List[Jsp], 
						 h: jax.Array,
						 s: jax.Array) -> jax.Array:
	"""
	E = Q*x_1x_2...x_k... + ... +b*x for PUBO of any order
	s is a boolean vector: s = True/False = x = 1/0
	"""

	def term_prod(v, idx: jax.Array):
		return v*(jnp.take(s, idx, fill_value = False).all())
	v_term_prod = jax.vmap(term_prod)
	
	e = jnp.array(0, dtype = J[0].data.dtype)
	for j in J:
		e += v_term_prod(j.data, j.indices).sum()
	
	return e + h@s

@jax.jit
def Jxs_pising(J: List[Jsp], 
							 h: jax.Array,
							 s: jax.Array) -> jax.Array:
	"""
	E = J*s_1s_2...s_k + ... +h*s for p-spin Ising of any order
	s is a boolean vector: s = True/False = spin = 1/-1
	"""

	def term_prod(v, idx: jax.Array):
		return v*(1 - 2*reduce(jnp.logical_xor, jnp.logical_not(jnp.take(s, idx, fill_value = True))))
	v_term_prod = jax.vmap(term_prod)
	
	e = jnp.array(0, dtype = J[0].data.dtype)
	for j in J:
		e += v_term_prod(j.data, j.indices).sum()
	
	return e + h@(2*s-1)


@jax.jit
def Jxs_wcnf(J: List[Jsp],
						 h: jax.Array,
        		 s: jax.Array) -> jax.Array:
	"""
	weigthed cnf-based computation of the energy (faster than PUBO for k-SAT)
	s is a boolean vector: s = True/False = x = 1/0
	"""

	def cnf_prod(v, idx: jax.Array):
		return v*jnp.logical_xor(jnp.take(s, jnp.abs(idx) - 1, fill_value = False), 
												   	jnp.signbit(idx)).all()
	v_cnf_prod = jax.vmap(cnf_prod)
	
	e = jnp.array(0, dtype = h.dtype)
	for j in J:
		e += v_cnf_prod(j.data, j.indices).sum()
	
	return -e - jnp.abs(h)@(jnp.logical_xor(s, jnp.logical_not(jnp.signbit(h))))

#Gradient functions
@jax.jit
def Jxs_de_pubo(Jat: List[Jsp], 
								s: jax.Array) -> jax.Array:
	"""
	dE = J*x_1x_2...dx_k + ... for PUBO of any order
	s is a boolean vector: s = True/False = x = 1/0
	"""

	def term_prod(v, idx: jax.Array):
		return v*(jnp.take(s, idx, fill_value = False).all())
	v_term_prod = jax.vmap(term_prod)
	
	de = jnp.array(0, dtype = Jat[0].data.dtype)
	for jat in Jat:
		de += v_term_prod(jat.data, jat.indices).sum()
	
	return de

@jax.jit
def Jxs_de_pising(Jat: List[Jsp], 
									s: jax.Array) -> jax.Array:
	"""
	dE = J*s_1s_2...ds_k + ... for p-spin Ising of any order
	s is a boolean vector: s = True/False = spin = 1/-1
	"""

	def term_prod(v, idx: jax.Array):
		return v*(1 - 2*reduce(jnp.logical_xor, jnp.logical_not(jnp.take(s, idx, fill_value = True))))
	v_term_prod = jax.vmap(term_prod)
	
	de = jnp.array(0, dtype = Jat[0].data.dtype)
	for jat in Jat:
		de += v_term_prod(jat.data, jat.indices).sum()
	
	return de


@jax.jit
def Jxs_de_wcnf(Jat: List[Jsp], 
        			  s: jax.Array) -> jax.Array:
	"""
	gain (gradient) of a variable = make - break for weighted cnf (faster than PUBO for weighted k-SAT)
	s is a boolean vector: s = True/False = x = 1/0
	"""
	
	def de_cnf_prod(v, idx: jax.Array):
		return v*idx[-1]*(jnp.logical_xor(jnp.take(s, jnp.abs(idx[0:-1]) - 1, fill_value = False), 
																		jnp.signbit(idx[0:-1])).all())
	v_de_cnf_prod = jax.vmap(de_cnf_prod)
	
	de = jnp.array(0, dtype = Jat[0].data.dtype)
	for jat in Jat:
		de += v_de_cnf_prod(jat.data, jat.indices).sum() #parallelized make - break
	
	return de

@partial(jax.jit, static_argnames = ["problem_mode"])
def	get_de(Jati: List[Jsp], 
					 hi: jax.Array,
        	 si: jax.Array,
					 s: jax.Array,
					 problem_mode: str) -> jax.Array:

	if problem_mode == "pubo": #PUBO
		de = (Jxs_de_pubo(Jati, s) + hi)*(2*si - 1)

	elif problem_mode == "pising": #p-spin Ising
		de = (Jxs_de_pising(Jati, s) + hi)*(4*si - 2)

	else: #weighted cnf
		de = (Jxs_de_wcnf(Jati, s) + hi)*(2*si - 1)

	return de

@partial(jax.jit, static_argnames = ["problem_mode"])
def get_energy(Jh: Jhdata,
							 s: jax.Array, 
							 problem_mode: str) -> jax.Array:

	if problem_mode == "pubo": #PUBO
		e = -Jxs_pubo(Jh.J, Jh.h, s)

	elif problem_mode == "pising": #p-spin Ising
		e = -Jxs_pising(Jh.J, Jh.h, s)

	else: #weighted cnf
		e = -Jxs_wcnf(Jh.J, Jh.h, s)

	return e

#####


#####
# NMC/MCMC sweeps logic
#####

@partial(jax.jit, static_argnames=['problem_mode'])
def mc_step(Jh: Jhdata,
						se: Seminokey,
						i: int,
						log_r: float,
						beta: float,
						problem_mode: str) -> Tuple[Seminokey, jax.Array]:
	"""
	Monte Carlo Metropolis-Hastings step at index i (with energy tracking)
	"""
	s = se.s
	e = se.e

	if problem_mode == "pubo":
		de = (Jxs_de_pubo([Jsp(jat.data[i], jat.indices[i]) for jat in Jh.Jat], s) + Jh.h[i])*(2*s[i] - 1)
	elif problem_mode == "pising":
		de = (Jxs_de_pising([Jsp(jat.data[i], jat.indices[i]) for jat in Jh.Jat], s) + Jh.h[i])*(4*s[i] - 2)
	else:
		de = (Jxs_de_wcnf([Jsp(jat.data[i], jat.indices[i]) for jat in Jh.Jat], s) + Jh.h[i])*(2*s[i] - 1)
	
	# stat: (x, de), where x = 1 is flipped, 0 is not flipped, -1 is nmc
	s, e, stat = lax.cond(log_r < -de*beta, 
												lambda: (s.at[i].set(jnp.logical_not(s[i])), e + de, 
																 jnp.array([1, de])), 
												lambda: (s, e, 
																 jnp.array([0, 0], dtype = de.dtype)))

	min_e, min_s = lax.cond(e < se.min_e,
													lambda: (e, s), 
													lambda: (se.min_e, se.min_s))
	
	return Seminokey(s, min_s, e, min_e), stat

@partial(jax.jit, static_argnames=['problem_mode'])
def mc_step_colored(Jh: Jhdata,
									  s: jax.Array,
									  i: int,
									  log_r: float,
									  beta: float,
									  problem_mode: str) -> Tuple[jax.Array, jax.Array, jax.Array]:
	"""
	Monte Carlo Metropolis-Hastings step at index i
	"""

	if problem_mode == "pubo":
		de = (Jxs_de_pubo([Jsp(jat.data[i], jat.indices[i]) for jat in Jh.Jat], s) + Jh.h[i])*(2*s[i] - 1)
	elif problem_mode == "pising":
		de = (Jxs_de_pising([Jsp(jat.data[i], jat.indices[i]) for jat in Jh.Jat], s) + Jh.h[i])*(4*s[i] - 2)
	else:
		de = (Jxs_de_wcnf([Jsp(jat.data[i], jat.indices[i]) for jat in Jh.Jat], s) + Jh.h[i])*(2*s[i] - 1)

	#track the negative energy change (even if not flipped)
	min_s_i, neg_de = lax.cond(de < 0, 
														 lambda: (jnp.logical_not(s[i]), de), 
														 lambda: (s[i], jnp.array(0, dtype = de.dtype)))

	# stat: (x, de), where x = 1 is flipped, 0 is not flipped, -1 is ncm
	s_i, stat = lax.cond(log_r < -de*beta, 
											lambda: (jnp.logical_not(s[i]), jnp.array([1, de, neg_de])), 
											lambda: (s[i], jnp.array([0, 0, neg_de])))

	return s_i, min_s_i, stat


@partial(jax.jit, static_argnames=['problem_mode'])
def c_step(Jh: Jhdata,
					 se: Seminokey,
					 is_c: bool,
					 i: int,
					 log_r: float,
					 beta: float,
					 problem_mode: str) -> Tuple[Seminokey, jax.Array]:
	"""
	Monte Carlo Metropolis-Hastings cluster step at index i
	"""
	s = se.s

	if problem_mode == "pubo":
		de = (Jxs_de_pubo([Jsp(jat.data[i], jat.indices[i]) for jat in Jh.Jat], s) + Jh.h[i])*(2*s[i] - 1)

	elif problem_mode == "pising":
		de = (Jxs_de_pising([Jsp(jat.data[i], jat.indices[i]) for jat in Jh.Jat], s) + Jh.h[i])*(4*s[i] - 2)

	else:
		de = (Jxs_de_wcnf([Jsp(jat.data[i], jat.indices[i]) for jat in Jh.Jat], s) + Jh.h[i])*(2*s[i] - 1)
	
	s, e, stat = lax.cond(jnp.logical_and(is_c, log_r < -de*beta), 
												lambda: (s.at[i].set(jnp.logical_not(s[i])), se.e + de, 
																 jnp.array([1, de])), 
												lambda: (s, se.e, 
																 jnp.array([0 - jnp.logical_not(is_c), 0], dtype = de.dtype)))

	min_e, min_s = lax.cond(e < se.min_e,
												 lambda: (e, s), 
												 lambda: (se.min_e, se.min_s))																 
	
	return Seminokey(s, min_s, e, min_e), stat


@partial(jax.jit, static_argnames=['problem_mode'])
def c_step_colored(Jh: Jhdata,
									 s: jax.Array,
									 is_c: bool,
									 i: int,
									 log_r: float,
									 beta: float,
									 problem_mode: str) -> Tuple[jax.Array, jax.Array, jax.Array]:
	"""
	Colored Monte Carlo Metropolis-Hastings cluster step at index i
	"""

	if problem_mode == "pubo":
		de = (Jxs_de_pubo([Jsp(jat.data[i], jat.indices[i]) for jat in Jh.Jat], s) + Jh.h[i])*(2*s[i] - 1)

	elif problem_mode == "pising":
		de = (Jxs_de_pising([Jsp(jat.data[i], jat.indices[i]) for jat in Jh.Jat], s) + Jh.h[i])*(4*s[i] - 2)

	else:
		de = (Jxs_de_wcnf([Jsp(jat.data[i], jat.indices[i]) for jat in Jh.Jat], s) + Jh.h[i])*(2*s[i] - 1)

	#track the negative energy change (even if not flipped)
	min_s_i, neg_de = lax.cond(jnp.logical_and(is_c, de < 0), 
														 lambda: (jnp.logical_not(s[i]), de), 
														 lambda: (s[i], jnp.array(0, dtype = de.dtype)))

	s_i, stat = lax.cond(jnp.logical_and(is_c, log_r < -de*beta), 
											 lambda: (jnp.logical_not(s[i]), jnp.array([1, de, neg_de])), 
											 lambda: (s[i], jnp.array([0 - jnp.logical_not(is_c), 0, neg_de], dtype = de.dtype)))

	return s_i, min_s_i, stat


@partial(jax.jit, static_argnames=['nsw_all', 'problem_mode', 'anneal', 'totrack'])
def mcmc_sampling(Jh: Jhdata,
								  seminkey: Seminkey,
								  beta: jax.Array,
								  nsw_all: int,
								  problem_mode: str = "wcnf", 
								  anneal: bool = False, 
								  totrack: bool = True) -> Tuple[Seminkey, Dict]:
	
	s_old = seminkey.s
	e_old = seminkey.e
	min_s_old = seminkey.min_s
	min_e_old = seminkey.min_e
	
	N = seminkey.s.size

	#linear schedule of the inverse temperature beta
	if anneal:
		beta_schedule = jnp.linspace(beta[0], beta[1], nsw_all, endpoint = False) 
	else:
		beta_schedule = jnp.full((nsw_all,), beta)
	
	def mc_sweep_call(seminkey: Seminkey, beta: jax.Array):
		#MCMC sweep with a random permutation of flips
		
		def mc_step_call(se: Seminokey, i_mc_r):
			i = i_mc_r[0]
			mcr = i_mc_r[1]

			return mc_step(Jh, se, i, mcr, beta, problem_mode)

		key, subkey1, subkey2 = jax.random.split(seminkey.key, 3)

		perm = jax.random.permutation(subkey1, N) 				 #generate permutation of the sweep

		#generate random numbers for the sweep all at once
		mc_r = jnp.log(jax.random.uniform(subkey2, (N,)))
		
		se = Seminokey(seminkey.s, seminkey.min_s, seminkey.e, seminkey.min_e)

		se, stats = lax.scan(mc_step_call, se, xs = (perm, mc_r)) #type: ignore
		
		if totrack:
			sweep_accepted_flips = jnp.where(stats[:, 0] > 0, 1.0, 0.0).sum()
			sweep_total_flips = 	 jnp.where(stats[:, 0] >= 0, 1.0, 0.0).sum()

			sweep_accepted_climb_flips = jnp.where(stats[:, 1] > 0, 1.0, 0).sum()
			sweep_de_avg_climb = 	 			 jnp.where(stats[:, 1] > 0, stats[:, 1], 0).sum()

			#sometimes the total number of flips at small temperature is zero
			sweep_stats = jnp.array([sweep_accepted_flips/(sweep_total_flips + 1e-9),
															 sweep_de_avg_climb/(sweep_accepted_climb_flips + 1e-9)])
		else:
			sweep_stats = jnp.empty(0)
		
		return Seminkey(se.s, se.min_s, se.e, se.min_e, key), sweep_stats

	#perform a for loop implementing Nsw MC sweeps
	seminkey, sweep_stats_batch = lax.scan(mc_sweep_call, seminkey, xs = beta_schedule)
	
	#record the sampling statistics
	if totrack:
		sampling_stats = {
			"avg_ar": 	 sweep_stats_batch[:, 0].mean(),   # average acceptance rate of flips
			"avg_climb": sweep_stats_batch[:, 1].mean(),	 # average energy climb is flips accepted with de > 0

			"jump":	 (s_old != seminkey.s).sum()/N,    # jump distance
			"de":		 seminkey.e - e_old,							 # energy change

			"min_jump":	 (min_s_old != seminkey.min_s).sum()/N,    # jump distance
			"min_de":		 seminkey.min_e - min_e_old								 # energy change
		}
	else:
		sampling_stats = {}
	
	return seminkey, sampling_stats


@partial(jax.jit, static_argnames=['nsw_all', 'problem_mode', 'anneal', 'totrack'])
def mcmc_sampling_colored(
	Jh: Jhdata,
	seminkey: Seminkey,
	beta: jax.Array,
	colors: jax.Array,
	nsw_all: int,
	problem_mode: str = "wcnf", 
	anneal: bool = False, 
	totrack: bool = True
) -> Tuple[Seminkey, Dict]:
	
	s_old = seminkey.s
	e_old = seminkey.e
	min_s_old = seminkey.min_s
	min_e_old = seminkey.min_e
	
	N = seminkey.s.size
	n_colors = colors.shape[0]

	#linear schedule of the inverse temperature beta
	if anneal:
		beta_schedule = jnp.linspace(beta[0], beta[1], nsw_all, endpoint = True) 
	else:
		beta_schedule = jnp.full((nsw_all,), beta)
	
	def mc_sweep_call_colored(seminkey: Seminkey, beta: jax.Array):
		#MCMC sweep with a random permutation of flips
		
		def step_call_colored(carry, color):
			#monte-carlo random numbers:
			mc_r_color = carry[0].at[color].get(
				mode = 'fill', 
				fill_value = 0
			)
			# s = carry[1]
			# min_s = carry[2]
			# de = carry[3]
			# min_de = carry[4]

			s, min_s, stat = \
				jax.vmap(mc_step_colored, 
								 in_axes=(None, None, 0, 0, None, None), 
								 out_axes=(0, 0, 1))(
				Jh, 
				carry[1], #state s
				color, 
				mc_r_color, 
				beta, 
				problem_mode
			)
			
			s = carry[1].at[color].set(s, mode = 'drop')
			min_s = carry[1].at[color].set(min_s, mode = 'drop') #carry[1] instead of [2] on purporse, not a mistake!

			# tracking the colored flips
			flips = jnp.where(color != N, stat[0], -1)

			# tracking the energy change of each flip
			de_all = jnp.where(color != N, stat[1], 0)
			carry_de = carry[3] + de_all.sum() #type: ignore

			# tracking the energy reduction of each potential flip
			min_de_all = jnp.where(color != N, stat[2], 0)
			carry_min_de = carry[3] + min_de_all.sum() #type: ignore
			
			min_s, carry_min_de = \
				lax.cond(carry_min_de < carry[4], 
								 lambda: (min_s, carry_min_de), 
								 lambda: (carry[2], carry[4]))

			return (carry[0], s, min_s, carry_de, carry_min_de), (flips, de_all)

		key, subkey1, subkey2 = jax.random.split(seminkey.key, 3)

		#used for permutation of colors
		perm = jax.random.permutation(subkey1, n_colors)

		#generate random numbers for the sweep all at once
		mc_r = jnp.log(jax.random.uniform(subkey2, (N,)))
		
		(_, s, min_s, carry_de, carry_min_de), stats = \
			lax.scan(step_call_colored, 
							 (mc_r, 
								seminkey.s, 
								seminkey.min_s, 
								jnp.array(0, dtype = seminkey.e.dtype), 
								seminkey.min_e - seminkey.e), xs = colors[perm])

		flips_all = stats[0].reshape(-1) #type: ignore
		de_all = stats[1].reshape(-1) #type: ignore

		e = seminkey.e + carry_de
		min_e = seminkey.e + carry_min_de
		
		if totrack:
			sweep_accepted_flips = jnp.where(flips_all > 0, 1.0, 0.0).sum()
			sweep_total_flips = 	 jnp.where(flips_all >= 0, 1.0, 0.0).sum()

			sweep_accepted_climb_flips = jnp.where(de_all > 0, 1.0, 0).sum()  #type: ignore
			sweep_de_avg_climb = 	 			 jnp.where(de_all > 0, de_all, 0).sum() #type: ignore

			#sometimes the total number of flips at small temperature is zero
			sweep_stats = jnp.array([sweep_accepted_flips/(sweep_total_flips + 1e-9),
															 sweep_de_avg_climb/(sweep_accepted_climb_flips + 1e-9)])
		else:
			sweep_stats = jnp.empty(0)
		
		return Seminkey(s, min_s, e, min_e, key), sweep_stats

	#perform a for loop implementing Nsw MC sweeps
	seminkey, sweep_stats_batch = lax.scan(mc_sweep_call_colored, seminkey, xs = beta_schedule)
	
	#record the sampling statistics
	if totrack:
		sampling_stats = {
			"avg_ar": 	 sweep_stats_batch[:, 0].mean(),   # average acceptance rate of flips
			"avg_climb": sweep_stats_batch[:, 1].mean(),	 # average energy climb is flips accepted with de > 0

			"jump":	 (s_old != seminkey.s).sum()/N,    # jump distance
			"de":		 seminkey.e - e_old,							 # energy change

			"min_jump":	 (min_s_old != seminkey.min_s).sum()/N,    # jump distance
			"min_de":		 seminkey.min_e - min_e_old								 # energy change
		}
	else:
		sampling_stats = {}
	
	return seminkey, sampling_stats

@partial(jax.jit, static_argnames=['nsw_phase1', 'nsw_phase2', 'problem_mode', 'totrack'])
def nmc_sampling(
	Jh: Jhdata,
	sekey: Sekey,

	beta_phase1: jax.Array,
	beta_phase2: jax.Array,

	nsw_phase1: int,
	nsw_phase2: int,

	problem_mode: str = "wcnf",
	totrack: bool = True
) -> Tuple[Seminkey, Dict]:

	N = sekey.s.size
	
	key, subkey = jax.random.split(sekey.key, 2)
	perm = jax.random.permutation(subkey, N)	# generate permutation of the sweep
	
	def cluster_step_call(se: Seminokey, xs):
		beta = xs[0]
		is_cluster = xs[0].astype(bool)
		i = xs[1]
		mcr = xs[2]

		#beta == 0.0 means the full rejection-free flip of the cluster
		return c_step(Jh, se, is_cluster, i, mcr, beta, problem_mode)


	def sweep_call(runner_state, i_cycle):
		#Flop sweep with a random permutation of flips

		seminkey = runner_state[0]
		perm = runner_state[1]
		beta_start = runner_state[2]
		beta_end = runner_state[3]

		# # uncomment if annealing schedule is needed:
		# beta = beta_start + (i_cycle/(nsw_phase1 - 1 + 1e-9))*(beta_end - beta_start)
		beta = beta_start

		key, subkey1, subkey2 = jax.random.split(seminkey.key, 3)

		mc_r = jnp.log(jax.random.uniform(subkey1, (N,)))

		#Perform the cluster move sweep
		se, stats = \
			lax.scan(cluster_step_call, 
							Seminokey(seminkey.s, seminkey.min_s, seminkey.e, seminkey.min_e), 
							xs = (beta[perm], perm, mc_r)) #type: ignore

		if totrack:
			accepted_flips = jnp.where(stats[:, 0] > 0, 1.0, 0.0).sum()
			total_flips = jnp.where(stats[:, 0] >= 0, 1.0, 0.0).sum()

			accepted_climb_flips = jnp.where(stats[:, 1] > 0, 1.0, 0).sum()
			de_avg_climb = jnp.where(stats[:, 1] > 0, stats[:, 1], 0).sum() #type: ignore

			#sometimes the total number of flips at small temperature is zero
			stats = jnp.array([accepted_flips/(total_flips + 1e-9),
															de_avg_climb/(accepted_climb_flips + 1e-9)])

		else:
			stats = jnp.empty(0)
		
		perm = jax.random.permutation(subkey2, N) #generate new permutation of the sweep
		
		return (Seminkey(se.s, se.min_s, se.e, se.min_e, key), perm, beta_start, beta_end), stats


	seminkey = Seminkey(sekey.s, sekey.s, sekey.e, sekey.e, key)

	runner_state = (seminkey, perm, beta_phase1, beta_phase1)
	#perform a for loop implementing nsw_phase1 MC sweeps
	(seminkey_phase1, perm, _, _), phase1_stats = \
		lax.scan(sweep_call, runner_state, xs = jnp.arange(nsw_phase1))

	runner_state = (seminkey_phase1, perm, beta_phase2, beta_phase2)
	#perform a for loop implementing nsw_phase2 MC sweeps
	(seminkey, _, _, _), phase2_stats = \
		lax.scan(sweep_call, runner_state, xs = jnp.arange(nsw_phase2))

	if totrack:
		stats_nmc = {

			"phase1_ar": 	  phase1_stats[:, 0].mean(), # average acceptance rate of flips
			"phase1_climb": phase1_stats[:, 1].mean(),	# average energy climb if flips accepted with de > 0

			"phase1_jump":	 (seminkey_phase1.s != sekey.s).sum()/N,
			"phase1_de":		  seminkey_phase1.e - sekey.e,

			"phase2_ar": 	  phase2_stats[:, 0].mean(), # average acceptance rate of flips
			"phase2_climb":  phase2_stats[:, 1].mean(), # average energy climb if flips accepted with de > 0

			"phase2_jump":	 (seminkey.s != seminkey_phase1.s).sum()/N,
			"phase2_de":		  seminkey.e - seminkey_phase1.e,

			"jump":	 (seminkey.s != sekey.s).sum()/N, 	  #jump distance
			"de":		  seminkey.e - sekey.e								#energy change
		}

	else:
		stats_nmc = {}

	return seminkey, stats_nmc


@partial(jax.jit, static_argnames=['nsw_phase1', 'nsw_phase2', 'problem_mode', 'totrack'])
def nmc_sampling_colored(
	Jh: Jhdata,
	sekey: Sekey,

	beta_phase1: jax.Array,
	beta_phase2: jax.Array,

	colors: jax.Array, 

	nsw_phase1: int,
	nsw_phase2: int,

	problem_mode: str = "wcnf",
	totrack: bool = True

) -> Tuple[Seminkey, Dict]:

	N = sekey.s.size
	n_colors = colors.shape[0]
	
	key, subkey = jax.random.split(sekey.key, 2)
	col_perm = jax.random.permutation(subkey, n_colors) #generate permutation of the colors

	def cluster_step_call_colored(carry, color):
		#monte-carlo random numbers:
		mc_r_color = carry[0].at[color].get(
			mode = 'fill', 
			fill_value = 0
		)
		# s = carry[1]
		# min_s = carry[2]
		# carry_de = carry[3]
		# carry_min_de = carry[4]

		beta_color = carry[5].at[color].get(
			mode = 'fill', 
			fill_value = 0
		)
		#beta = 0 (False) means that the variable is frozen in its state
		is_cluster_color = beta_color.astype(bool)

		s, min_s, stat = \
			jax.vmap(c_step_colored,
							 in_axes=(None, None, 0, 0, 0, 0, None), 
							 out_axes=(0, 0, 1))(
			Jh, 
			carry[1], 
			is_cluster_color, 
			color, 
			mc_r_color, 
			beta_color, 
			problem_mode
		)

		s = carry[1].at[color].set(s, mode = 'drop')
		min_s = carry[1].at[color].set(min_s, mode = 'drop') #carry[1] instead of [2] on purporse, not a mistake!

		# tracking the colored flips
		flips = jnp.where(color != N, stat[0], -1)

		# tracking the energy change of each flip
		de_all = jnp.where(color != N, stat[1], 0)
		carry_de = carry[3] + de_all.sum() #type: ignore

		# tracking the energy reduction of each potential flip
		min_de_all = jnp.where(color != N, stat[2], 0)
		carry_min_de = carry[3] + min_de_all.sum() #type: ignore
		
		min_s, carry_min_de = \
			lax.cond(carry_min_de < carry[4], 
							lambda: (min_s, carry_min_de), 
							lambda: (carry[2], carry[4]))

		return (carry[0], s, min_s, carry_de, carry_min_de, carry[5]), (flips, de_all)


	def sweep_call_colored(runner_state, i_cycle):
		seminkey = runner_state[0]
		col_perm = runner_state[1]
		beta_start = runner_state[2]
		beta_end = runner_state[3]

		## uncomment if annealing schedule is needed:
		# beta = beta_start + (i_cycle/(nsw_phase1 - 1 + 1e-9))*(beta_end - beta_start)
		beta = beta_start

		key, subkey1, subkey2 = jax.random.split(seminkey.key, 3)

		mc_r = jnp.log(jax.random.uniform(subkey1, (N,)))

		#Perform the cluster move sweep
		(_, s, min_s, carry_de, carry_min_de, _), stats = \
			lax.scan(cluster_step_call_colored, 
							(mc_r, 
							 seminkey.s, 
							 seminkey.min_s, 
							 jnp.array(0, dtype = seminkey.e.dtype),
							 seminkey.min_e - seminkey.e,
							 beta), 
							 xs = colors[col_perm]) #type: ignore

		flips_all = stats[0].reshape(-1) #type: ignore
		de_all = stats[1].reshape(-1) #type: ignore

		e = seminkey.e + carry_de
		min_e = seminkey.e + carry_min_de

		if totrack:
			sweep_accepted_flips = jnp.where(flips_all > 0, 1.0, 0.0).sum()
			sweep_total_flips = jnp.where(flips_all >= 0, 1.0, 0.0).sum()

			sweep_accepted_climb_flips = jnp.where(de_all > 0, 1.0, 0).sum()
			sweep_de_avg_climb = jnp.where(de_all > 0, de_all, 0).sum()

			#sometimes the total number of flips at small temperature is zero
			sweep_stats = jnp.array([sweep_accepted_flips/(sweep_total_flips + 1e-9),
															sweep_de_avg_climb/(sweep_accepted_climb_flips + 1e-9)])

		else:
			sweep_stats = jnp.empty(0)
		
		col_perm = jax.random.permutation(subkey2, n_colors)
		
		return (Seminkey(s, min_s, e, min_e, key), col_perm, beta_start, beta_end), sweep_stats

	seminkey = Seminkey(sekey.s, sekey.s, sekey.e, sekey.e, key)

	runner_state = (seminkey, col_perm, beta_phase1, beta_phase1)
	#perform a for loop implementing nsw_phase1 MC sweeps
	(seminkey_phase1, col_perm, _, _), phase1_stats = \
		lax.scan(sweep_call_colored, runner_state, xs = jnp.arange(nsw_phase1))

	runner_state = (seminkey_phase1, col_perm, beta_phase2, beta_phase2)
	#perform a for loop implementing nsw_phase2 MC sweeps
	(seminkey, _, _, _), phase2_stats = \
		lax.scan(sweep_call_colored, runner_state, xs = jnp.arange(nsw_phase2))

	if totrack:
		stats_nmc = {

			"phase1_ar": 	  phase1_stats[:, 0].mean(), # average acceptance rate of flips
			"phase1_climb": phase1_stats[:, 1].mean(),	# average energy climb if flips accepted with de > 0

			"phase1_jump":	 (seminkey_phase1.s != sekey.s).sum()/N,
			"phase1_de":		  seminkey_phase1.e - sekey.e,

			"phase2_ar": 	  phase2_stats[:, 0].mean(), # average acceptance rate of flips
			"phase2_climb":  phase2_stats[:, 1].mean(), # average energy climb if flips accepted with de > 0

			"phase2_jump":	 (seminkey.s != seminkey_phase1.s).sum()/N,
			"phase2_de":		  seminkey.e - seminkey_phase1.e,

			"jump":	 (seminkey.s != sekey.s).sum()/N, 	  #jump distance
			"de":		  seminkey.e - sekey.e								#energy change
		}

	else:
		stats_nmc = {}

	return seminkey, stats_nmc