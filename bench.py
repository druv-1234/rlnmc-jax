import os
# MacOS multi-CPU setup
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=8"

import sys
import time
import json
import argparse
import wandb

sys.path.append('.')

from modules.nmc_types import *
from modules.utils import bench_wandb_logging, tts_bootstrap
from modules.diversity import get_diversity
from modules.clustering import *
from modules.environment.nmcgym import *

import jax
import jax.numpy as jnp
from typing import Dict, Union, List

from flax import nnx
import distrax
from modules.policies.nmc_policy import *
import orbax.checkpoint as ocp

from problems.readers import *
from problems.read_utils import pad_and_concatenate_Jh, pad_and_stack_Jh, pad_and_stack_colors

import pickle

def make_bench_nmc(Jh_list: List[Jhdata], 
									 ground_states: Union[jax.Array, None],
									 colors_list: List[jax.Array],

									 nmc_setup: NMCSetup,

									 algo_setup: Dict,

									 params: NMCParams,
									 instance_names: List[str],
									 save_config: Dict,

									 diversity_r: List[float],
									 magnetizations_bins: List[float],

									 n_replicas: int,

									 devices,
									 ac_config, 
									 seed):
	
	project_dir = os.path.join(save_config["save_path"], save_config["project_name"])
	os.makedirs(project_dir, exist_ok = True)

	save_data_dir = os.path.join(project_dir, save_config["save_data"]["name"])
	os.makedirs(save_data_dir, exist_ok = True)

	if save_config["save_data"]["energy"]:
		for i_name in instance_names:
			with open(os.path.join(save_data_dir, i_name + "_e.dat"), mode = 'w') as _:
				pass

	if save_config["save_data"]["diversity"]:
		for i_name in instance_names:
			with open(os.path.join(save_data_dir, i_name + "_div.dat"), mode = 'w') as _:
				pass

	if save_config["save_data"]["magnetizations"] and (algo_setup["algo"] in ["rlnmc", "nmc"]):
		with open(os.path.join(save_data_dir, "mag_action_probabilities_avg.dat"), mode = 'w') as _:
			pass
		with open(os.path.join(save_data_dir, "mag_action_probabilities_std.dat"), mode = 'w') as _:
			pass
	
	n_devices = len(devices)
	n_instances = len(Jh_list)
	N0 = Jh_list[0].h.size

	assert n_replicas%n_devices == 0, "Can't distribute equally across devices!"
	n_replicas_per_device = n_replicas//n_devices

	if nmc_setup.coloring:
		if colors_list is None:
			raise RuntimeError("Please provide graph coloring!")

		#stack and pad the graph colorings of each instance
		colors_stack = pad_and_stack_colors(colors_list)
		print(f"Stacked number of colors: {colors_stack.shape[1]}")

	else:
		colors_stack = None

	if n_instances > 1:
		Jh_stacked = pad_and_stack_Jh(Jh_list) #type: ignore
		Jh_cat = pad_and_concatenate_Jh(Jh_list, nmc_setup.problem_mode) #type: ignore

	else:
		Jh_cat = Jhdata(Jh_list[0].J, Jh_list[0].h, Jh_list[0].Jat)

		Jh_stacked = Jhdata(
			J = [Jsp(jnp.expand_dims(j.data, 0), jnp.expand_dims(j.indices, 0)) for j in Jh_list[0].J], 
			h = jnp.expand_dims(Jh_list[0].h, 0), 
			Jat = [Jsp(jnp.expand_dims(jat.data, 0), jnp.expand_dims(jat.indices, 0)) for jat in Jh_list[0].Jat]
		)

	if n_instances > 1:
		save_fg_name = f"{instance_names[0]}-{instance_names[-1]}_{nmc_setup.lbp_beta}_fg.pkl"
	else:
		save_fg_name = f"{instance_names[0]}-{instance_names[0]}_{nmc_setup.lbp_beta}_fg.pkl"

	if not (save_config["save_data"]["factor_graph"] == "load" and os.path.isfile(os.path.join(project_dir, save_fg_name))):
		#Initialize the environment
		nmc_env = NMCgym(Jh_stack = Jh_stacked, 
										 Jh_cat = Jh_cat,
										 ground_states = ground_states, 
										 colors_stack = colors_stack,
										 nmc_setup = nmc_setup, 
										 FG = None)
		
		print("Initialized env")
		if nmc_setup.use_lbp:
			if save_config["save_data"]["factor_graph"] in ["load", "save"]:
				with open(os.path.join(project_dir, save_fg_name), 'wb') as f:	
					pickle.dump(nmc_env.FG, f)
					print("Saved factor graph")

	else:
		if nmc_setup.use_lbp:
			with open(os.path.join(project_dir, save_fg_name), 'rb') as f:	
				FG = pickle.load(f)
				print("Initializing env with precomputed factor graph")
		else:
			FG = None

		nmc_env = NMCgym(Jh_stack = Jh_stacked, 
										Jh_cat = Jh_cat,
										ground_states = ground_states, 
										colors_stack = colors_stack,
										nmc_setup = nmc_setup, 
										FG = FG)


	if algo_setup["algo"] == "rlnmc":
		model = ActorCritic(Jh_cat.h.size, ac_config, rngs = nnx.Rngs(0))
		model_graph, model_state_rnd, other_vars = nnx.split(model, nnx.Param, ...)

		h_init_local = model.gru_cell_local.initialize_carry(
			(n_devices, n_replicas_per_device, n_instances, N0, -1), 
			nnx.Rngs(0)
		)
		h_init_global = model.gru_cell_global.initialize_carry(
			(n_devices, n_replicas_per_device, n_instances, -1), 
			nnx.Rngs(0)
		)
		h_init = (h_init_local, h_init_global) #per spin and per instance carry hidden GRU vectors

	else:
		h_init = jnp.empty((n_devices, n_replicas_per_device, n_instances))
	
	sweeps_per_step = params.nmc_nsw_eq_phase
	if params.use_nmc:
		sweeps_per_step += params.nmc_neq_cycles*params.nmc_nsw_neq_phase

	if sweeps_per_step == 0:
		STEPS_PER_LOG = 1
	else:
		if save_config["log"]["sweeps_per_log"] < sweeps_per_step:
			STEPS_PER_LOG = 1
		else:
			STEPS_PER_LOG = save_config["log"]["sweeps_per_log"]//sweeps_per_step

	STEPS_PER_LOG = min(params.episode_steps, STEPS_PER_LOG)
	N_LOGGING_STEPS = params.episode_steps//STEPS_PER_LOG

	if len(diversity_r) > 0:
		track_diversity = True
		set_of_all_solutions = [set() for _ in range(n_instances)]

	else:
		track_diversity = False


	def bench_nmc(key, model_state_pretrained) -> None:

		if algo_setup["algo"] == "rlnmc":
			if model_state_pretrained is not None:
				model_state = model_state_pretrained
			else:
				model_state = model_state_rnd

		key, reset_key = jax.random.split(key)
		
		reset_keys = jax.random.split(reset_key, n_replicas).reshape(n_devices, n_replicas_per_device)

		timer = 0
		time_a = time.time()
		if n_instances > 1:
			if nmc_setup.use_lbp:
				save_reset_name = f"{instance_names[0]}-{instance_names[-1]}_{nmc_setup.lbp_beta}_reset_{n_replicas}_{seed}_{algo_setup['algo']}.pkl"
			else:
				save_reset_name = f"{instance_names[0]}-{instance_names[-1]}_de_reset_{n_replicas}_{seed}_{algo_setup['algo']}.pkl"
		else:
			if nmc_setup.use_lbp:
				save_reset_name = f"{instance_names[0]}-{instance_names[0]}_{nmc_setup.lbp_beta}_reset_{n_replicas}_{seed}_{algo_setup['algo']}.pkl"
			else:
				save_reset_name = f"{instance_names[0]}-{instance_names[0]}_de_reset_{n_replicas}_{seed}_{algo_setup['algo']}.pkl"

		if not (save_config["save_data"]["reset"] == "load" and os.path.isfile(os.path.join(project_dir, save_reset_name))):
			
			print("Resetting environments...")
			nmc_obs, nmc_state, reset_stats = jax.pmap(
				jax.vmap(nmc_env.reset, in_axes = (0, None)), 
				in_axes = (0, None) #type: ignore
			)(reset_keys, params)  
								
			jax.tree.map(lambda x: x.block_until_ready(), (nmc_obs, nmc_state))
			timer += time.time() - time_a
			
			if save_config["save_data"]["reset"] in ["load", "save"]:
				with open(os.path.join(project_dir, save_reset_name), 'wb') as f:	
					pickle.dump((nmc_obs, nmc_state, reset_stats), f)
				print("Saved reset data")

		else:
			with open(os.path.join(project_dir, save_reset_name), 'rb') as f:	
				nmc_obs, nmc_state, reset_stats = pickle.load(f)
			
			nmc_state = jax.tree.map(lambda x: x.reshape([n_devices, n_replicas_per_device] + list(x.shape[2:])), nmc_state)
			nmc_obs = jax.tree.map(lambda x: x.reshape([n_devices, n_replicas_per_device] + list(x.shape[2:])), nmc_obs)

			print("Loaded precomputed reset data")

		print("Reset complete, running environment steps...")

		def ncm_per_device(env, runner_state):
			def _env_step(runner_state, unused):
				nmc_state, last_obs, key, h = runner_state
										
				key, subkey_1, subkey_2, subkey_3 = jax.random.split(key, 4)
		
				if algo_setup["algo"] == "sa":
					action_step = jnp.empty((n_replicas_per_device,))

					if params.use_nmc:
						raise RuntimeError("Can't use nmc for SA algorithm!")

				elif algo_setup["algo"] == "nmc":
					action_cand = last_obs.mag >= algo_setup["nmc"]["threshold"]

				elif algo_setup["algo"] == "rlnmc":
					model = nnx.merge(model_graph, model_state, other_vars)

					local_obs = jnp.stack((last_obs.s, last_obs.mag), axis = 2)[None, :]
					global_obs = jnp.concatenate(
						(
							jnp.tile(last_obs.norm_time[:, None, None], (1, n_instances, 1)), 
							jnp.tile(last_obs.norm_beta[:, None, None], (1, n_instances, 1)), 
						 	last_obs.norm_best_e[:, :, None]
					), axis = 2)[None, :]

					h, pi, _ = model(
						h,  
						local_obs, 
						global_obs, 
						jnp.zeros(n_replicas_per_device, dtype = bool)[None, :]
					) #type:ignore

					pi = pi.squeeze(axis = 0)
					
					pi_bernoulli = distrax.Bernoulli(logits = pi)

					action_cand = pi_bernoulli.sample(seed = subkey_1)

				else:
					raise RuntimeError("wrong algo!")

				if algo_setup["algo"] in ["nmc", "rlnmc"]:
					cl_subkeys = jax.random.split(subkey_2, n_replicas_per_device*n_instances)

					action_seeds = jax.vmap(choose_a_cluster_seed_nmc, in_axes=(0, 0, None))(
						cl_subkeys, 
						action_cand.reshape(-1, N0), 
						algo_setup["nmc"]["n_seeds"]
					).reshape(n_replicas_per_device, n_instances, N0)

					action_step = jax.vmap(jax.vmap(get_clusters_anygraph, 
																					in_axes=(0, None, 0, 0)), 
																	in_axes = (None, None, 0, 0))(
						Jh_stacked, 
						nmc_setup.problem_mode,
						action_seeds,
						action_cand.reshape(n_replicas_per_device, n_instances, N0)
					).reshape(n_replicas_per_device, -1)

				step_keys = jax.random.split(subkey_3, n_replicas_per_device)

				nmc_obs, nmc_state, _, _, info = jax.vmap(env.step, in_axes = (0, 0, 0, None))(
					step_keys, 
					nmc_state, 
					action_step, 
					params
				)


				if nmc_setup.track_stats:
					if algo_setup["algo"] in ["nmc", "rlnmc"]:
						
						#for tracking the histogram of magnetizations
						range_min_values = jnp.array(magnetizations_bins)
						range_max_values = jnp.concatenate(
							(jnp.array(magnetizations_bins[1:]), jnp.array([jnp.inf])),
							axis = 0
						)

						def get_mag_stats(mag, act):
							#accumulate m populations according to the given resolution
							mag_action = jax.vmap(
								lambda x, y, a, b: jnp.where(jnp.where(x >= a, x, b) < b, y, 0).sum(), #type: ignore
								in_axes = (None, None, 0, 0)
							)(mag, act, range_min_values, range_max_values)

							mag_dist = jax.vmap(
								lambda x, a, b: jnp.where(jnp.where(x >= a, x, b) < b, 1, 0).sum(), #type: ignore
								in_axes = (None, 0, 0)
							)(mag, range_min_values, range_max_values)

							return mag_action/(mag_dist + 1e-9)

						# get the probability for spins with a given magnetization
						# to be in the backbone subgraph
						mag_stats = jax.vmap(get_mag_stats)(
							last_obs.mag.reshape(n_replicas_per_device*n_instances, -1),
							action_step.reshape(n_replicas_per_device*n_instances, -1)
						).reshape(n_replicas_per_device, n_instances, -1)

						mag_stats_count = jnp.where(mag_stats != 0, 1, 0).sum(axis = 1)

						info["mag_action_prob"] = (
							mag_stats.sum(axis = 1)/(mag_stats_count + 1e-9)
						)

						#log the backbone statistics (averaged over instances)
						info['mc']['backbone_size'] = {}
						info['mc']['backbone_cand'] = {}
						info['mc']['backbone_seed_size'] = {}
						info['mc']['backbone_rel_size'] = {}
						info['mc']['backbone_seed_rel_size'] = {}

						for p in [0, 20, 50, 80, 100]:
							info['mc']['backbone_size'][f"{p}"] = jnp.percentile(
								action_step.reshape(n_replicas_per_device, n_instances, -1).mean(axis=2),
								p, axis = 1
							)
							info['mc']['backbone_cand'][f"{p}"] = jnp.percentile(
								action_cand.reshape(n_replicas_per_device, n_instances, -1).mean(axis=2),
								p, axis = 1
							)
							info['mc']['backbone_seed_size'][f"{p}"] = jnp.percentile(
								action_seeds.reshape(n_replicas_per_device, n_instances, -1).mean(axis=2),
								p, axis = 1
							)
							
							def masked_perc_jit(x, mask, perc):
								#gets the needed percentile of a masked array
								x = jnp.where(mask, x, jnp.inf)
								sorted_x = jnp.sort(x) #type: ignore

								n = jnp.sum(mask)
								
								idx = jnp.where(n == 0, mask.size, (perc*(n - 1)/100).astype(int))
								return sorted_x.at[idx].get(mode = 'fill', fill_value = 0)

							#average relative cluster size (not couning zeros)
							backbone_rel_size = (
								action_step.reshape(n_replicas_per_device, n_instances, -1).mean(axis=2)/
								(action_cand.reshape(n_replicas_per_device, n_instances, -1).mean(axis=2) + 1e-9)
							)
							info['mc']['backbone_rel_size'][f"{p}"] = jax.vmap(masked_perc_jit, in_axes=(0, 0, None))(
								backbone_rel_size,
								jnp.where(backbone_rel_size, True, False), 
								p
							)

							#average relative seeds size (not counting zeros)
							backbone_seed_rel_size = (
								action_seeds.reshape(n_replicas_per_device, n_instances, -1).mean(axis=2)/
								(action_cand.reshape(n_replicas_per_device, n_instances, -1).mean(axis=2) + 1e-9)
							)
							info['mc']['backbone_seed_rel_size'][f"{p}"] = jax.vmap(masked_perc_jit, in_axes=(0, 0, None))(
								backbone_seed_rel_size,
								jnp.where(backbone_seed_rel_size, True, False),
								p
							)

					else:
						info["mag_action_prob"] = None

				info["energies"] = nmc_state.e
				info["best_energies"] = nmc_state.best_e

				runner_state = (nmc_state, nmc_obs, key, h)
				
				return runner_state, info
			
			(nmc_state, nmc_obs, key, h), info = \
				jax.lax.scan(_env_step, runner_state, None, STEPS_PER_LOG)

			runner_state = (
				nmc_state,
				nmc_obs,
				key,
				h
			)
			return runner_state, info
		
		key, *subkeys = jax.random.split(key, n_devices + 1)
		runner_state = (
			nmc_state,
			nmc_obs,			
			jnp.stack(subkeys), 
			h_init
		)

		for log_step in range(N_LOGGING_STEPS):
			timer_old = timer
			time_a = time.time()
			
			a, b = jax.pmap(ncm_per_device, in_axes = (None, 0), #type: ignore
											static_broadcasted_argnums = (0))(nmc_env, runner_state)

			runner_state, traj_info = jax.tree.map(lambda x: x.block_until_ready(), (a, b))
			timer += time.time() - time_a

			traj_info["energies"] \
				= traj_info['energies'].swapaxes(0, 1).reshape(STEPS_PER_LOG, 
																											n_replicas, 
																											n_instances)
			traj_info["best_energies"] \
				= traj_info['best_energies'].swapaxes(0, 1).reshape(STEPS_PER_LOG, 
																														n_replicas, 
																														n_instances)

			final_best_energies = traj_info["best_energies"][-1]

			if ground_states is not None:
				final_best_energies = final_best_energies - ground_states[None, :]
			
			metric = {}
			
			#current number of sweeps in each of the replicas
			LOG_SWEEP = (log_step + 1)*STEPS_PER_LOG*sweeps_per_step + params.nsw_reset

			if track_diversity:
				if ground_states is None:
					raise RuntimeError("Need to know ground states (approximation) to compute diversity")

				time_diversity = 0
				time_div = time.time()

				best_states = np.array(runner_state[0].best_s.reshape(n_replicas, n_instances, -1))

				for i in range(n_instances):
					for j in range(n_replicas):
						if final_best_energies[j][i] <= nmc_setup.approximation:
							set_of_all_solutions[i].add(tuple(best_states[j][i]))

				time_diversity += time.time() - time_div
			
			tts_results = tts_bootstrap(
				final_best_energies.swapaxes(0, 1), 
				jnp.full((n_instances,), LOG_SWEEP),
				jnp.full((n_instances,), nmc_setup.approximation),
				jax.random.key(0),
				jnp.array(save_config["log"]["percentiles"]), 
				save_config["log"]["bootstrap"]
			)

			for i, p in enumerate(save_config["log"]["percentiles"]):
				metric[f"tts_{p}"] = tts_results["tts_p_means"][i]
				metric[f"s_tts_{p}"] = tts_results["tts_p_stds"][i]

			if save_config["log"]["wandb"]:
				if n_instances > 1:
					metric["pos"] = wandb.Histogram(tts_results["successes"]/(tts_results["successes"] + tts_results["failures"]))
					metric["tts_means"] = wandb.Histogram(tts_results["tts_means"])
				else:
					metric["pos"] = tts_results["successes"]/(tts_results["successes"] + tts_results["failures"])
					metric["tts_means"] = tts_results["tts_means"]

			else:
				if n_instances > 1:
					metric["pos"] = jnp.histogram(tts_results["successes"]/(tts_results["successes"] + tts_results["failures"]))
					metric["tts_means"] = jnp.histogram(tts_results["tts_means"])
				else:
					metric["pos"] = tts_results["successes"]/(tts_results["successes"] + tts_results["failures"])
					metric["tts_means"] = tts_results["tts_means"]

			if track_diversity:
				diversity = {r: {} for r in diversity_r}
				diversity[-1] = {}

				time_div = time.time()
				for i_name, solutions in zip(instance_names, set_of_all_solutions):
					div_int, div_r = get_diversity(solutions, diversity_r)

					for ri, r in enumerate(diversity_r):
						diversity[r][i_name] = div_r[ri]

					diversity[-1][i_name] = div_int

				for r in diversity_r:
					diversity_values = np.array(list(diversity[r].values()))
					# metric[f"div_median/{r}"] = np.median(diversity_values)
					metric[f"div_mean/{r}"] = np.mean(diversity_values)
					# metric[f"div_min/{r}"] = np.min(diversity_values)
					# metric[f"div_max/{r}"] = np.max(diversity_values)
				
				# metric[f"div_median/int"] = np.median(np.array(list(diversity[-1].values())))
				metric[f"div_mean/int"] = np.mean(np.array(list(diversity[-1].values())))
				# metric[f"div_min/int"] = np.min(np.array(list(diversity[-1].values())))
				# metric[f"div_max/int"] = np.max(np.array(list(diversity[-1].values())))

				num_solutions = {}
				for i_name, solutions in zip(instance_names, set_of_all_solutions):
					num_solutions[i_name] = len(solutions)
				num_sol_values = np.array(list(num_solutions.values()))

				metric["num_sol_median"] = np.median(num_sol_values)
				metric["num_sol_mean"] = np.mean(num_sol_values)
				metric["num_sol_min"] = np.min(num_sol_values)
				metric["num_sol_max"] = np.max(num_sol_values)

				time_diversity += time.time() - time_div


			best_energies = {}
			for i_name, beste in zip(instance_names, final_best_energies.swapaxes(0, 1)):
				best_energies[i_name] = beste

			if save_config["save_data"]["energy"]:
				for i_name in instance_names:
					with open(os.path.join(save_data_dir, i_name + "_e.dat"), mode = 'a') as f:
						np.savetxt(f, np.concatenate([[LOG_SWEEP], best_energies[i_name].astype(int)], dtype = int), 
											newline = " ", fmt = "%i")

			if save_config["save_data"]["diversity"]:
				for i_name in instance_names:
					with open(os.path.join(save_data_dir, i_name + "_div.dat"), mode = 'a') as f:
						np.savetxt(f, np.concatenate([[LOG_SWEEP], 
																					[diversity[r][i_name] for r in diversity_r], 
																					[diversity[-1][i_name]]], dtype = float), 
											newline = " ", fmt = "%.7g")
						
			energy_stats = {}
			if ground_states is not None:
				energy_stats["best_e"] 	= traj_info["best_energies"] - ground_states[None, None, :]
				e_all_close = jnp.isclose(traj_info["best_energies"], ground_states[None, None, :], atol = 0)
				energy_stats["best_e"] = energy_stats["best_e"]*jnp.logical_not(e_all_close)

			else:
				energy_stats["best_e"] = traj_info["best_energies"]

			if nmc_setup.track_stats:
				landscape_stats = {}
				landscape_stats["mc"] = \
					jax.tree.map(lambda x: x.swapaxes(0, 1).reshape(STEPS_PER_LOG, n_replicas), 
											 traj_info["mc"])

				landscape_stats["geometry"] = \
					jax.tree.map(lambda x: x.swapaxes(0, 1).reshape(STEPS_PER_LOG, n_replicas), 
											 traj_info["geometry"])
				
				# probability of spins with a given magnetization to be in a backbone
				# mean and std over replicas saved to a file in the run directory
				if traj_info["mag_action_prob"] is not None:
					if save_config["save_data"]["magnetizations"] and (algo_setup["algo"] in ["rlnmc", "nmc"]):
						mag_action_prob = traj_info["mag_action_prob"].swapaxes(0, 1).reshape(
							STEPS_PER_LOG, 
							n_replicas,
							len(bench_config["magnetizations_bins"])
						)

						with open(os.path.join(save_data_dir, "mag_action_probabilities_avg.dat"), mode = 'a') as f:
							for line in np.array(mag_action_prob.mean(axis = 1)):
								np.savetxt(f, line[None, :], fmt = "%.5g")

						with open(os.path.join(save_data_dir, "mag_action_probabilities_std.dat"), mode = 'a') as f:
							for line in np.array(mag_action_prob.std(axis = 1)):
								np.savetxt(f, line[None, :], fmt = "%.5g")

			else:
				landscape_stats = None

			other = {}
			other["seconds_per_log"] = jnp.array(timer - timer_old)
			other["sweeps_per_second_per_repetition"] = (
				jnp.array(STEPS_PER_LOG*sweeps_per_step)/other["seconds_per_log"]
			)

			if track_diversity:
				other["seconds_per_diversity"] = time_diversity

			if save_config["log"]["wandb"]:
				print("Logging to wandb at sweep: ", LOG_SWEEP)
				bench_wandb_logging(LOG_SWEEP,
														sweeps_per_step,
														energy_stats,
														landscape_stats,
														metric,
														other)
			else:
				print("Finished sweeps: ", LOG_SWEEP)

		#save magnetizations statistics to file
		if save_config["save_data"]["magnetizations"]:
			if nmc_setup.use_lbp:
				mag_name = f"{instance_names[0]}-{instance_names[-1]}_{nmc_setup.lbp_beta}_mag.txt"
			else:
				mag_name = f"{instance_names[0]}-{instance_names[-1]}_de_mag.txt"

			with open(os.path.join(save_data_dir, mag_name), 'w') as f:	
				np.savetxt(f, np.array(runner_state[1].mag.reshape(-1)), newline = " ", fmt = "%.7g")												

			print("Saved magnetizations to file")
		#bench nmc function complete

	return bench_nmc




##################################################################
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-bench_config", "--bench_config", 
											help = "bench_config filename", 
											default = "config_bench_uf500")

	parser.add_argument("-problem_config", "--problem_config", 
											help = "problem_config filename", 
											default = "config_500")

	parser.add_argument("-id", "--id", 
										help = "instance seeds id", 
										default = "default")
	
	parser.add_argument("-n_devices", "--n_devices", 
											help = "n_devices", 
											default = "default")
	
	parser.add_argument("-n_replicas", "--n_replicas", 
											help = "n_replicas", 
											default = "default")
	
	parser.add_argument("-seed", "--seed", 
											help = "seed", 
											default = "default")

	parser.add_argument("-project_name", "--project_name", 
											help = "project_name", 
											default = "default")

	parser.add_argument("-load_project_name", "--load_project_name", 
											help = "load_project_name", 
											default = "default")

	parser.add_argument("-load_name", "--load_name", 
											help = "load_name", 
											default = "default")
	
	parser.add_argument("-save_name", "--save_name", 
											help = "save_name", 
											default = "default")
	
	parser.add_argument("-load_run_id", "--load_run_id", 
											help = "load_run_id", 
											default = "default")

	parser.add_argument("-gen_only", "--gen_only", 
											help = "gen_only", 
											default = "0")


	parser.add_argument("-algo", "--algo", 
											help = "algo", 
											default = "default")
	
	args = parser.parse_args()

	with open(os.path.join("configs", f"{args.bench_config}.json"), 'r') as f:
		bench_config = json.load(f)

	with open(os.path.join("problems", 
												 bench_config["problem_class"], 
												 f"{args.problem_config}.json"), 'r') as f:
		problem_config = json.load(f)



	## bench config custom params
	bench_config["n_devices"] = bench_config['n_devices'] \
		if args.n_devices == "default" else int(args.n_devices)

	bench_config['seed'] = bench_config['seed'] \
		if args.seed == "default" else int(args.seed)

	bench_config["n_replicas"] = bench_config['n_replicas'] \
		if args.n_replicas == "default" else int(args.n_replicas)

	bench_config["save"]["save_data"]["name"] = bench_config["save"]["save_data"]["name"] \
		if args.save_name == "default" else args.save_name

	bench_config["load"]["run_id"] = bench_config["load"]["run_id"] \
		if args.load_run_id == "default" else args.load_run_id
	
	bench_config["save"]["project_name"] = bench_config["save"]["project_name"] \
		if args.project_name == "default" else args.project_name
	
	bench_config["load"]["project_name"] = bench_config["load"]["project_name"] \
		if args.load_project_name == "default" else args.load_project_name
	
	bench_config["load"]["name"] = bench_config["load"]["name"] \
		if args.load_name == "default" else args.load_name

	bench_config["algo_setup"]["algo"] = bench_config["algo_setup"]["algo"] \
		if args.algo == "default" else args.algo
	
	
	if args.id == "default":
		bench_config["instances"] = \
			list(range(bench_config["instances"][0][0], bench_config["instances"][0][1]+1)) if len(bench_config["instances"][0]) == 2 else \
			bench_config["instances"][0]
	else:
		bench_config["instances"] = \
			list(range(bench_config["instances"][int(args.id)][0], bench_config["instances"][int(args.id)][1]+1)) if len(bench_config["instances"][int(args.id)]) == 2 else \
			bench_config["instances"][int(args.id)]


	Jh_list = []
	colors_list = []

	N = problem_config["problem_setup"]["N"]
	M = problem_config["problem_setup"]["M"]

	instances_path = os.path.join("problems", bench_config["problem_class"], "instances")

	for iseed in bench_config["instances"]:
		instance_file_name = (
			f"{bench_config['problem_class']}_n{N}_m{M}_{iseed}"
		)
		file_path = os.path.join(instances_path, instance_file_name)
		
		if os.path.isfile(file_path + ".pkl"):
			with open(file_path + ".pkl", 'rb') as f:	
				Jh = pickle.load(f)

			print(f"Loaded {instance_file_name}")

		else:
			print(f"Reading {instance_file_name} for jax")
			wcnf = get_wcnf(file_path + ".cnf", 
											"float", no_weights = True)

			Jh = wcnf_for_jax(wcnf, dtype = jnp.float32)

			with open(file_path + ".pkl", 'wb') as f:
				pickle.dump(Jh, f)
				
			print(f"Saved {instance_file_name}")

		Jh_list.append(Jh)

		if bench_config["nmc_setup"]["coloring"]:
			if os.path.isfile(file_path + "_col" + ".txt"):
				coloring_labels = np.loadtxt(file_path + "_col" + ".txt").astype(int)
				# test_coloring_anygraph(np.abs(Jh.J[0].indices)-1, coloring_labels)

			else:
				coloring_labels = get_coloring_anygraph(
					np.abs(Jh.J[0].indices)-1, 
					Jh.h.size, 
					# Jh.Jat[0].data.shape[1],
					33,
					time_limit = 60
				)
				test_coloring_anygraph(np.abs(Jh.J[0].indices)-1, coloring_labels)
				
				np.savetxt(file_path + "_col" + ".txt", coloring_labels)
			
			#number of used colors:
			n_colors = max(coloring_labels) + 1
			#maximum variables of single color:
			max_color_size = max([np.where(coloring_labels == c)[0].shape[0] for c in range(n_colors)])

			print(f"{n_colors} colors, max {max_color_size} vars per color")

			coloring = jnp.full((n_colors, max_color_size), -1, dtype = jnp.int32)

			for c in range(n_colors):
				c_idx = jnp.where(coloring_labels == c)[0]
				coloring = coloring.at[c, :c_idx.size].set(c_idx)
			
			colors_list.append(coloring)

		else:
			colors_list.append(None)
	
	if bench_config["nmc_setup"]["coloring"]:
		color_numbers = [c.shape[0] for c in colors_list]
		print("Average coloring:", np.mean(color_numbers))
		print("Min coloring:", np.min(color_numbers))
		print("Max coloring:", np.max(color_numbers))

	if int(args.gen_only):
		exit()

	nmc_setup =  NMCSetup(
		approximation = bench_config["nmc_setup"]["approximation"],
		
		track_stats = bench_config["nmc_setup"]["track_stats"],
		problem_mode = bench_config["nmc_setup"]["problem_mode"],

		coloring = bench_config["nmc_setup"]["coloring"],

		energy_scale = bench_config["nmc_setup"]["energy_scale"],

		reward_scale = 1.0,

		numerical_eval_energy = bench_config["nmc_setup"]["numerical_eval_energy"],
		use_lbp = bench_config["nmc_setup"]["use_lbp"],

		lbp_beta = problem_config["NMC"]["lbp_beta"]
	)

	if bench_config["algo_setup"]["algo"] in ["nmc", "rlnmc"]:
		params = NMCParams(
			episode_steps = problem_config["NMC"]["episode_steps"],
			use_nmc = True,

			nsw_reset = problem_config["NMC"]["nsw_reset"],

			nmc_neq_cycles = problem_config["NMC"]["nmc_neq_cycles"],
			nmc_nsw_neq_phase = problem_config["NMC"]["nmc_nsw_neq_phase"],

			nmc_nsw_eq_phase = problem_config["NMC"]["nmc_nsw_eq_phase"],

			nmc_neq_beta = problem_config["NMC"]["nmc_neq_beta"],

			beta_i = problem_config["NMC"]["beta_i"],
			beta_f = problem_config["NMC"]["beta_f"],

			beta_start = problem_config["NMC"]["beta_start"],

			lbp_beta = problem_config["NMC"]["lbp_beta"],
			lbp_tolerance_m = problem_config["NMC"]["lbp_tolerance_m"],
			lbp_tolerance_d = problem_config["NMC"]["lbp_tolerance_d"],
			lbp_lambdas = jnp.array(problem_config["NMC"]["lbp_lambdas"]),
			lbp_num_iters = problem_config["NMC"]["lbp_num_iters"]
		)
	else:
		params = NMCParams(
			episode_steps = problem_config["SA"]["episode_steps"],
			
			use_nmc = False,

			nsw_reset = problem_config["SA"]["nsw_reset"],

			nmc_nsw_eq_phase = problem_config["SA"]["nmc_nsw_eq_phase"],

			beta_i = problem_config["SA"]["beta_i"],
			beta_f = problem_config["SA"]["beta_f"],

			beta_start = problem_config["SA"]["beta_start"]
		)


	bench_config["ground_states"] = \
		bench_config["ground_states"][0] if args.id == "default" else bench_config["ground_states"][int(args.id)]

	if len(bench_config["ground_states"]) > 1:
		assert len(bench_config["ground_states"]) == len(bench_config["instances"]) \
			and "Ground states length is not equal to instance seeds"
	
		print("Ground states:", bench_config["ground_states"])
		bench_config["ground_states"] = \
			jnp.array(bench_config["ground_states"], dtype = Jh_list[0].h.dtype)

	elif len(bench_config["ground_states"]) == 1:

		print("Ground state for all instances:", bench_config["ground_states"][0])

		bench_config["ground_states"] = jnp.full(
			(len(bench_config["instances"]),), 
			bench_config["ground_states"][0], 
			dtype = Jh_list[0].h.dtype
		)
			
	else:
		print("No ground states given")
		bench_config["ground_states"] = None



	with open(os.path.join("modules", "policies", "nmc_ac.json"), 'r') as f:
		ac_config = json.load(f)

	if bench_config["load"]["trained"] and (bench_config["algo_setup"]["algo"] == "rlnmc"):
		checkpointer_model = ocp.StandardCheckpointer()

		# load a pre-trained model for benchmarking
		load_path = os.path.join(bench_config["load"]["save_path"], 
														 bench_config["load"]["project_name"], 
														 "MODELS",
														 f"model_{bench_config['load']['run_id']}")

		load_file_path = os.path.join(load_path, bench_config["load"]["name"])

		if bench_config["load"]["type"] == "json":
			# reconstruct a model from a json file

			with open(load_file_path + ".json", 'r') as f:
				model_json = json.load(f)
			
			def convert_leaves_to_jax_array(d):
				if isinstance(d, dict):
						return {k: convert_leaves_to_jax_array(v) for k, v in d.items()}
				elif isinstance(d, list):
						return jnp.array(d, dtype = jnp.float32)
				else:
						return d
			model_dict = convert_leaves_to_jax_array(model_json)
			
			abstract_model = nnx.eval_shape(lambda: ActorCritic(
				Jh_list[0].h.size,
				ac_config, 
				rngs = nnx.Rngs(0))
			)

			graphdef, model_state, abstract_other = nnx.split(abstract_model, nnx.Param, ...)
			model_state.replace_by_pure_dict(model_dict) #type: ignore  

			print(f"Pre-trained model_state {load_file_path}, loaded via json")

		elif bench_config["load"]["type"] == "orbax":
			#load the models from a checkpointer

			abstract_model = nnx.eval_shape(lambda: ActorCritic(
				Jh_list[0].h.size,
				ac_config, 
				rngs = nnx.Rngs(0))
			)
			
			graphdef, abstract_state, abstract_other = nnx.split(abstract_model, nnx.Param, ...)
			model_state = checkpointer_model.restore(os.path.abspath(load_file_path), abstract_state)

			print(f"Trained model_state {load_file_path} loaded via orbax")

		else:
			raise RuntimeError("Loading error: invalid type")

	else:
		print(f"No pretraining used")
		model_state = None

	if bench_config["save"]["log"]["wandb"]:
		wandb_run = wandb.init(
			project = bench_config["save"]["project_name"],

			config = {
				"instances": bench_config["instances"], 
				"ground_states": bench_config["ground_states"],
				"problem_setup": problem_config["problem_setup"],

				"problem_class": bench_config["problem_class"],
				"n_replicas": bench_config["n_replicas"],
				"n_devices": bench_config["n_devices"],
				"seed": bench_config["seed"],

				"diversity_r": bench_config["diversity_r"],

				"nmc_setup": {
					"approximation": nmc_setup.approximation,
					"track_stats": nmc_setup.track_stats,
					"problem_mode": nmc_setup.problem_mode,
					"coloring": nmc_setup.coloring,
					"energy_scale": nmc_setup.energy_scale,
					"reward_scale": nmc_setup.reward_scale,
					"numerical_eval_energy": nmc_setup.numerical_eval_energy,
					"use_lbp": nmc_setup.use_lbp,
					"lbp_beta": nmc_setup.lbp_beta
				},
				
				"params": params,

				"algo_setup": bench_config["algo_setup"],

				"load_model": bench_config["load"] if bench_config["load"]["trained"] else None,
				"ac_config": ac_config,
				
				"save": bench_config["save"]
			}
		)
		bench_config["save"]["run_id"] = wandb_run.id  #type: ignore

	print("Using", jax.devices()[:bench_config["n_devices"]])
	bench_nmc = make_bench_nmc(
		Jh_list = Jh_list,
		ground_states = bench_config["ground_states"],
		colors_list = colors_list,

		nmc_setup = nmc_setup,
		params = params,
		
		instance_names = [f"{i}" for i in bench_config["instances"]],

		save_config = bench_config["save"],

		diversity_r = bench_config["diversity_r"],
		magnetizations_bins = bench_config["magnetizations_bins"],

		algo_setup = bench_config["algo_setup"],

		n_replicas =  bench_config["n_replicas"],
		ac_config = ac_config,

		devices = jax.devices()[:bench_config["n_devices"]], 
		seed = bench_config["seed"]
	)

	bench_nmc(jax.random.key(bench_config["seed"]), model_state)
