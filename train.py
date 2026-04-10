import os
# MacOS multi-CPU setup
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=8"

import json
import sys
import time
import argparse

sys.path.append('.')

from modules.nmc_types import Jhdata
from modules.environment.nmcgym import *
from modules.utils import train_wandb_logging

from modules.policies.nmc_policy import *
from modules.clustering import *

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Dict

from problems.read_utils import *

from problems.readers import *

from flax import nnx
from flax.training import train_state

import optax
import distrax

from pprint import pprint
import pickle
import orbax.checkpoint as ocp

import wandb

class Transition(NamedTuple):
	done: jnp.ndarray
	action_cand: jnp.ndarray
	value: jnp.ndarray
	reward: jnp.ndarray
	log_prob: jnp.ndarray
	obs: jnp.ndarray
	info: jnp.ndarray

class GaeTransition(NamedTuple):
	done: jnp.ndarray
	value: jnp.ndarray
	reward: jnp.ndarray
	

def make_train(Jh_stacked: Jhdata,
							 Jh_cat: Jhdata, 
							 ground_states: Union[jax.Array, None],
							 colors_stack: Union[jax.Array, None],

							 nmc_setup: NMCSetup,
							 params: NMCParams,

							 n_replicas: int,
							 train_setup: Dict,

							 project_dir: str,
							 i_names: List[str],

							 devices, 
							 seed):

	N_DEVICES = len(devices)
	N_TRIALS = train_setup["trials_per_log"]

	assert train_setup["episodes_per_trial"]%train_setup["episodes_per_update"] == 0

	assert (n_replicas%train_setup["num_minibatches"] == 0) and \
		"please use correct number of repetitions for minibatches"
	
	N0 = Jh_stacked.h[0].size
	N = Jh_cat.h.size
	n_instances = Jh_stacked.h.shape[0]

	if n_instances > 1:
		save_fg_name = f"{i_names[0]}-{i_names[-1]}_{nmc_setup.lbp_beta}_fg.pkl"
	else:
		save_fg_name = f"{i_names[0]}-{i_names[0]}_{nmc_setup.lbp_beta}_fg.pkl"

	#precomputed factor graphs are saved to a file for each beta of LBP
	os.makedirs(os.path.join(project_dir, "FACTOR_GRAPHS"), exist_ok = True) 

	#reset data is saved for each trial so that can be re-used in future training runs
	os.makedirs(os.path.join(project_dir, "RESET_DATA"), exist_ok = True) 

	os.makedirs(os.path.join(project_dir, "MODELS"), exist_ok = True) 

	if not os.path.isfile(os.path.join(project_dir, "FACTOR_GRAPHS", save_fg_name)):
		#Initialize the environment
		nmc_env = NMCgym(Jh_stack = Jh_stacked, 
										 Jh_cat = Jh_cat,
										 ground_states = ground_states, 
										 colors_stack = colors_stack,
										 nmc_setup = nmc_setup, 
										 FG = None)

		print("Initialized env")
		if nmc_setup.use_lbp:
			with open(os.path.join(project_dir, "FACTOR_GRAPHS", save_fg_name), 'wb') as f:	
				pickle.dump(nmc_env.FG, f)
				print("Saved factor graph")

	else:
		if nmc_setup.use_lbp:
			with open(os.path.join(project_dir, "FACTOR_GRAPHS", save_fg_name), 'rb') as f:	
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

	nmc_env = LogWrapper(nmc_env)

	def _calculate_gae(traj_batch, last_val, last_done):
		def _get_advantages(carry, transition):
			gae, next_value, next_done = carry
			done, value, reward = (
				transition.done, 
				transition.value, 
				transition.reward
			)
			delta = reward + train_setup["gamma"] * next_value * (1 - next_done) - value
			gae = delta + train_setup["gamma"] * train_setup["gae_lambda"] * (1 - next_done) * gae
			
			return (gae, value, done), gae
		_, advantages = jax.lax.scan(_get_advantages, 
																(jnp.zeros_like(last_val), last_val, last_done), 
																traj_batch, 
																reverse = True, 
																unroll = 16)
		return advantages, advantages + traj_batch.value

	def train_function(key: jax.Array, t_state, train_step_id: int):

		train_info_per_trial = []
		for trial_id in range(N_TRIALS):

			if nmc_setup.use_lbp:
				save_reset_name = (
					f"{i_names[0]}-{i_names[-1]}_{nmc_setup.lbp_beta}_reset_step{train_step_id}_trial{trial_id}_devices{N_DEVICES}_nreps{n_replicas}_{seed}.pkl"
				)
			else:
				save_reset_name = (
					f"{i_names[0]}-{i_names[-1]}_de_reset_step{train_step_id}_trial{trial_id}_devices{N_DEVICES}_nreps{n_replicas}_{seed}.pkl"
				)
			
			key, subkey = jax.random.split(key)
			rngs = nnx.Rngs(subkey)

			h_reset_local = model.gru_cell_local.initialize_carry(
				(n_replicas, n_instances, N0, -1), 
				rngs
			)

			h_reset_global = model.gru_cell_global.initialize_carry(
				(n_replicas, n_instances, -1), 
				rngs
			)
			h_reset = (h_reset_local, h_reset_global) #per spin and per instance carry hidden GRU vectors

			#episode done flags
			dones_reset = jnp.zeros(n_replicas, dtype = bool)

			key, reset_key = jax.random.split(key)

			print(f"Resetting environment for trial...")
			reset_keys = jax.random.split(reset_key, n_replicas).reshape(N_DEVICES, -1)
			
			if not (os.path.isfile(os.path.join(project_dir, "RESET_DATA", save_reset_name))):
				obs, log_state, _ = \
					jax.pmap(jax.vmap(nmc_env.reset, 
														in_axes = (0, None)),
									in_axes=(0, None))(reset_keys, params)  #type: ignore

				jax.tree.map(lambda x: x.block_until_ready(), (obs, log_state))
				
				with open(os.path.join(project_dir, "RESET_DATA", save_reset_name), 'wb') as f:	
					pickle.dump((obs, log_state), f)
				print(f"Saved reset data at step {train_step_id}, trial {trial_id}")

			else:
				with open(os.path.join(project_dir, "RESET_DATA", save_reset_name), 'rb') as f:	
					obs, log_state = pickle.load(f)

				print(f"Loaded precomputed reset data at step {train_step_id}, trial {trial_id}")

			print("Reset complete")

			#collapse the axes from parallel devices
			obs_reset = jax.tree.map(lambda x: lax.collapse(x, 0, 2), obs)
			log_state_reset = jax.tree.map(lambda x: lax.collapse(x, 0, 2), log_state)
			# reset_stats = jax.tree.map(lambda x: lax.collapse(x, 0, 2), reset_stats)
			
			N_UPDATES = train_setup["episodes_per_trial"]//train_setup["episodes_per_update"]
			print(f"Running {N_UPDATES} RL update steps...")

			# TRAIN LOOP
			def _update_step(runner_state, unused):
				def env_scan(runner_state):
					def _env_episode(runner_state, unused):
						def _env_step(runner_state, unused):
							(
								t_state, 
								last_log_state, 
								last_obs, 
								last_done, 
								h, 
								key
							) = runner_state

							local_obs = jnp.stack((last_obs.s, last_obs.mag), axis = 2)[None, :]

							global_obs = jnp.concatenate(
								(
									jnp.tile(last_obs.norm_time[:, None, None], (1, n_instances, 1)), 
									jnp.tile(last_obs.norm_beta[:, None, None], (1, n_instances, 1)), 
									last_obs.norm_best_e[:, :, None]
							), axis = 2)[None, :]

							(h, pi, value), (_, _) = \
								t_state.apply_fn(t_state.params, t_state.other_vars)(
									h,  
									local_obs, 
									global_obs, 
									last_done[None, :]
								)
							
							#policy output predicting the clusters
							pi = pi.squeeze(axis = 0)
							#value function predicting returns
							value = value.squeeze(axis = 0)

							if not (train_setup["pretrain_mode"] in ["action", "value", "action_and_value"]):
								#pure data from the policy, no supervision from the NMC thresholding								

								pi_bernoulli = distrax.Bernoulli(logits = pi)
								key, subkey = jax.random.split(key)
								
								action_cand = pi_bernoulli.sample(seed = subkey)

								log_prob = pi_bernoulli.log_prob(action_cand)

							else:
								#supervise policy pre-training using the data from standard NMC thresholding

								action_cand = last_obs.mag >= train_setup["nmc"]["threshold"]
								log_prob = None #actor loss will not be used

							key, *cl_subkeys = jax.random.split(key, n_replicas*n_instances + 1)
							action_seeds = jax.vmap(choose_a_cluster_seed_nmc, in_axes=(0, 0, None))(
								jnp.stack(cl_subkeys), 
								action_cand.reshape(n_replicas*n_instances, N0), 
								train_setup["nmc"]["n_seeds"]
							).reshape(n_replicas, n_instances, N0)

							action_step = jax.vmap(jax.vmap(get_clusters_anygraph, 
																							in_axes=(0, None, 0, 0)), 
																		in_axes = (None, None, 0, 0))(
								Jh_stacked, 
								nmc_setup.problem_mode,
								action_seeds,
								action_cand.reshape(n_replicas, n_instances, N0)
							).reshape(n_replicas, -1)
		
							# STEP ENV
							key, *step_keys = jax.random.split(key, n_replicas + 1)

							obs, log_state, reward, done, info = \
								jax.vmap(nmc_env.step, in_axes = (0, 0, 0, None))(
									jnp.stack(step_keys), 
									last_log_state, 
									action_step,
									params
								)
							
							# jax.debug.print("reward = {x}", x = reward.mean())
							
							info['mc']['backbone_size'] = {}
							info['mc']['backbone_cand'] = {}
							info['mc']['backbone_seed_size'] = {}
							info['mc']['backbone_rel_size'] = {}
							info['mc']['backbone_seed_rel_size'] = {}

							for p in [0, 20, 50, 80, 100]:
								info['mc']['backbone_size'][f"{p}"] = jnp.percentile(
									action_step.reshape(n_replicas, n_instances, -1).mean(axis=2),
									p, axis = 1
								)
								info['mc']['backbone_cand'][f"{p}"] = jnp.percentile(
									action_cand.reshape(n_replicas, n_instances, -1).mean(axis=2),
									p, axis = 1
								)
								info['mc']['backbone_seed_size'][f"{p}"] = jnp.percentile(
									action_seeds.reshape(n_replicas, n_instances, -1).mean(axis=2),
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
									action_step.reshape(n_replicas, n_instances, -1).mean(axis=2)/
									(action_cand.reshape(n_replicas, n_instances, -1).mean(axis=2) + 1e-9)
								)
								info['mc']['backbone_rel_size'][f"{p}"] = jax.vmap(masked_perc_jit, in_axes=(0, 0, None))(
									backbone_rel_size,
									jnp.where(backbone_rel_size, True, False), 
									p
								)

								#average relative seeds size (not counting zeros)
								backbone_seed_rel_size = (
									action_seeds.reshape(n_replicas, n_instances, -1).mean(axis=2)/
									(action_cand.reshape(n_replicas, n_instances, -1).mean(axis=2) + 1e-9)
								)
								info['mc']['backbone_seed_rel_size'][f"{p}"] = jax.vmap(masked_perc_jit, in_axes=(0, 0, None))(
									backbone_seed_rel_size,
									jnp.where(backbone_seed_rel_size, True, False),
									p
								)

							transition = Transition(
								last_done, 
								action_cand, 
								value, 
								reward, 
								log_prob, #type: ignore
								last_obs, 
								info
							)
							runner_state = (t_state, log_state, obs, done, h, key)

							return runner_state, transition
						
						runner_state, traj_batch = \
							jax.lax.scan(_env_step, runner_state, length = params.episode_steps)
					
						(
							t_state, 
							_, #replaced by reset
							_, #replaced by reset
							done, 
							_, #replaced by reset, 
							key
						) = runner_state
						
						runner_state = (
							t_state, 
							log_state_reset, 
							obs_reset, 
							jnp.full_like(done, True), 
							h_reset,
							key
						)

						return runner_state, traj_batch
					
					runner_state, traj_batch = \
							jax.lax.scan(_env_episode, runner_state, length = train_setup["episodes_per_update"])
					
					#collapse the several episodes onto a single axis
					traj_batch = jax.tree.map(lambda x: lax.collapse(x, 0, 2), traj_batch)
					
					return runner_state, traj_batch

				h_save = runner_state[-2] #save h for update state below

				# Collect trajectories
				runner_state, traj_batch = env_scan(runner_state)

				#last value is always set to zero because the episode is reset
				last_value = jnp.zeros_like(traj_batch.value[-1])
				last_done = jnp.full_like(traj_batch.done[-1], True)
				
				gae_traj_batch = GaeTransition(
					done = traj_batch.done,
					value = traj_batch.value,
					reward = traj_batch.reward
				)

				#calculate gae for every instance individually
				advantages, targets = jax.vmap(_calculate_gae,
																	 		 in_axes = (GaeTransition(None, 2, 2), 1, None), #type:ignore
																			 out_axes = (2, 2))(
					gae_traj_batch, 
					last_value, 
					last_done
				)

				## UPDATE NETWORK
				def _update_epoch(update_state, epoch_id):
					def _update_minbatch(t_state, batch):

						h, traj_batch, advantages, targets = batch

						def _loss_fn(params, other_vars, h, traj_batch, gae, targets):

							traj_length = gae.shape[0]
							minibatch_size = gae.shape[1]
							n_instances = gae.shape[2]

							local_obs = jnp.stack((traj_batch.obs.s, traj_batch.obs.mag), axis = 3)

							global_obs = jnp.concatenate(
								(
									jnp.tile(traj_batch.obs.norm_time[:, :, None, None], (1, 1, n_instances, 1)), 
									jnp.tile(traj_batch.obs.norm_beta[:, :, None, None], (1, 1, n_instances, 1)), 
									traj_batch.obs.norm_best_e[:, :, :, None]
							), axis = 3)
							
							(_, pi, value), (_, _) = \
								t_state.apply_fn(params, other_vars)(
									jax.tree.map(lambda x: x[0], h),  
									local_obs, 
									global_obs, 
									traj_batch.done
								)

							#VALUE LOSS
							value_losses = jnp.square(value - targets)

							if not (train_setup["pretrain_mode"] in ["action", "value", "action_and_value"]):
								if train_setup["clip_eps_vf"] != 0:
									#value clipping
									value_pred_clipped = traj_batch.value + (
										value - traj_batch.value
									).clip(-train_setup["clip_eps_vf"], train_setup["clip_eps_vf"])

									value_losses_clipped = jnp.square(value_pred_clipped - targets)

									value_losses_max = jnp.maximum(value_losses, value_losses_clipped)

									#log the clipping rate of the value function
									clipping_rate_vf = (value_losses != value_losses_max).reshape(-1, n_instances).mean(axis = 0)
								
									value_loss = value_losses_max.reshape(-1, n_instances).mean(axis = 0)

								else:
									#no value clipping
									value_loss = value_losses.reshape(-1, n_instances).mean(axis = 0)
									clipping_rate_vf = jnp.zeros_like(value_loss, dtype = value_loss.dtype)
								
							else:
								#no value clipping
								value_loss = value_losses.reshape(-1, n_instances).mean(axis = 0)
								clipping_rate_vf = jnp.zeros_like(value_loss, dtype = value_loss.dtype)

							value_loss_rel = jnp.sqrt(value_loss)/(targets.std() + 1e-8)

							#ACTOR LOSS
							pi_bernoulli = distrax.Bernoulli(logits = pi)								

							entropy_loss = (
								pi_bernoulli.entropy().reshape(-1, n_instances, N0).mean(axis = 2) 
							).mean(axis = 0)

							### all-ones and all-zeros penalty
							all_ones_log_prob = (
								pi_bernoulli.log_prob(jnp.full_like(traj_batch.action_cand, 1, dtype = pi.dtype))
							).reshape(traj_length, minibatch_size, n_instances, N0).sum(axis = 3)
							all_zeros_log_prob = (
								pi_bernoulli.log_prob(jnp.full_like(traj_batch.action_cand, 0, dtype = pi.dtype))
							).reshape(traj_length, minibatch_size, n_instances, N0).sum(axis = 3)

							all_same_penalty = (
									jnp.exp(all_ones_log_prob) 
								+ jnp.exp(all_zeros_log_prob)
							).reshape(-1, n_instances).mean(axis = 0)

							###

							if not (train_setup["pretrain_mode"] in ["action", "value", "action_and_value"]):
								traj_log_prob = traj_batch.log_prob.reshape(
									traj_length, minibatch_size, n_instances, N0
								).sum(axis = 3)

								log_prob = pi_bernoulli.log_prob(jnp.array(traj_batch.action_cand, dtype = pi.dtype))
								log_prob = log_prob.reshape(traj_length, minibatch_size, n_instances, N0).sum(axis = 3)

								#get the ration of probabolity per replica
								ratio = jnp.exp(log_prob - traj_log_prob)

								if train_setup["clip_eps"] != 0:
									clipped_ratio = jnp.clip(
										ratio,
										1.0 - train_setup["clip_eps"],
										1.0 + train_setup["clip_eps"]
									)
									#track the clipping rate of the policy
									clipping_rate = (ratio != clipped_ratio).reshape(-1, n_instances).mean(axis = 0)
					
								else:
									clipping_rate = jnp.zeros((n_instances,), dtype = ratio.dtype)

								# normalize the GAE values
								gae_mean = gae.reshape(-1, n_instances).mean(axis = 0)
								gae_std = gae.reshape(-1, n_instances).std(axis = 0)

								gae = (gae - gae_mean[None, None, :]) / (gae_std[None, None, :] + 1e-8)

								loss_actor1 = ratio*gae
								if train_setup["clip_eps"] != 0:
									#policy clipping
									loss_actor2 = clipped_ratio*gae
									actor_loss = (
										-jnp.minimum(loss_actor1, loss_actor2).reshape(-1, n_instances).mean(axis = 0)
									)

								else:
									#no policy clipping
									actor_loss = -loss_actor1.reshape(-1, n_instances).mean(axis = 0)

								total_loss = (
									actor_loss 
									+ train_setup["value_coef"]*value_loss 
									- train_setup["entropy_coef"]*entropy_loss
									+ train_setup["all_same_coef"]*all_same_penalty
								)

							else:
								clipping_rate = jnp.zeros((n_instances,), dtype = pi.dtype)

								if train_setup["pretrain_mode"] == "action":
									#cross entropy loss between the supervision actions and the policy outputs
									actor_loss = optax.sigmoid_binary_cross_entropy(pi, traj_batch.action_cand)
									actor_loss = actor_loss.reshape(-1, n_instances, N0).mean(axis = 2).mean(axis = 0)
									
									total_loss = actor_loss

								elif train_setup["pretrain_mode"] == "value":
									#actor is not trained (action given by supervision)
									actor_loss = jnp.zeros((n_instances,), dtype = pi.dtype)

									total_loss = train_setup["value_coef"]*value_loss

								elif train_setup["pretrain_mode"] == "action_and_value":
									#cross entropy loss between the supervision actions and the policy outputs
									actor_loss = optax.sigmoid_binary_cross_entropy(pi, traj_batch.action_cand)
									actor_loss = actor_loss.reshape(-1, n_instances, N0).mean(axis = 2).mean(axis = 0)

									total_loss = (
										actor_loss +
										train_setup["value_coef"]*value_loss
									)


							return (
								total_loss.mean(), #total loss is averaged over all instances
								( #per-instance losses are saved for logging as well
									total_loss,
									actor_loss,
									train_setup["value_coef"]*value_loss,
									-train_setup["entropy_coef"]*entropy_loss,
									train_setup["all_same_coef"]*all_same_penalty,
									value_loss_rel,
									clipping_rate,
									clipping_rate_vf
								)
							)
						
						grad_fn = jax.value_and_grad(_loss_fn, has_aux = True)
						total_loss, grads = grad_fn(t_state.params, t_state.other_vars,
																				h, traj_batch, advantages, targets)
						
						return t_state.apply_gradients(grads = grads), total_loss

					(
						t_state,
						h,
						traj_batch,
						advantages,
						targets,
						key,
					) = update_state

					key, subkey = jax.random.split(key)

					permutation = jax.random.permutation(subkey, n_replicas)

					shuffled_h = jax.tree_util.tree_map(
						lambda x: jnp.take(x[None, :], permutation, axis = 1), h
					)

					shuffled_traj_batch = jax.tree_util.tree_map(
						lambda x: jnp.take(x, permutation, axis = 1), traj_batch
					)

					shuffled_advantages = jax.tree_util.tree_map(
						lambda x: jnp.take(x, permutation, axis = 1), advantages
					)
					
					shuffled_targets = jax.tree_util.tree_map(
						lambda x: jnp.take(x, permutation, axis = 1), targets
					)

					shuffled_batch = (
						shuffled_h, 
						shuffled_traj_batch, 
						shuffled_advantages, 
						shuffled_targets
					)

					minibatches = jax.tree_util.tree_map(
						lambda x: jnp.swapaxes(
							jnp.reshape(x, [x.shape[0], train_setup["num_minibatches"], -1] + list(x.shape[2:])),
							1, 0,
						),
						shuffled_batch,
					)

					t_state, loss_stats = jax.lax.scan(_update_minbatch, t_state, minibatches)

					update_state = (
						t_state,
						h,
						traj_batch,
						advantages,
						targets,
						key
					)
					return update_state, loss_stats
				
				
				update_state = (
					runner_state[0], #t_state
					h_save, #take h from the runner state above
					traj_batch,
					advantages,
					targets,
					runner_state[-1] #key
				)

				update_state, loss_info = \
					jax.lax.scan(_update_epoch, update_state, xs = jnp.arange(train_setup["update_epochs"]))
				
				t_state = update_state[0]
				key = update_state[-1]
				
				runner_state = (
					t_state, 
					runner_state[1],  #log_state
					runner_state[2],	#obs
					runner_state[3],	#dones
					runner_state[4],	#h
					key
				)
				
				return runner_state, (traj_batch.info, loss_info)

			runner_state = (
				t_state,
				log_state_reset,
				obs_reset,
				dones_reset,
				h_reset,
				key
			)

			runner_state, running_info = \
				jax.lax.scan(
					_update_step, 
					runner_state, 
					length = N_UPDATES
				)
			
			t_state = runner_state[0]
			key = runner_state[-1]

			train_info_per_trial.append(
				{
					"traj_info": running_info[0], 
					"loss_info": running_info[1]
				}
			)
		
		train_info = {
			"traj_info": jax.tree.map(lambda *x: jnp.concatenate(x, axis = 0), 
																*[info["traj_info"] for info in train_info_per_trial]),
			"loss_info": jax.tree.map(lambda *x: jnp.concatenate(x, axis = 0), 
																*[info["loss_info"] for info in train_info_per_trial]),
		}

		#return the final model parameters, random key and the information about the trajectory and loss
		return t_state, key, train_info

	return train_function






if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	#add arguments that could be customly changed

	#PROJECT SETUP
	parser.add_argument("-train_config", "--train_config", 
											help = "experiment run train_config filename", 
											default = "config_train_sf250")

	parser.add_argument("-problem_config", "--problem_config", 
											help = "problem_config filename", 
											default = "config_250")
	
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

	parser.add_argument("-gen_instances_only", "--gen_instances_only", 
											help = "gen_instances_only", 
											default = "0")

	#SAVE AND LOAD
	parser.add_argument("-project_name", "--project_name", 
										help = "project_name", 
										default = "default")

	parser.add_argument("-load_project_name", "--load_project_name", 
										help = "load_project_name", 
										default = "default")

	parser.add_argument("-load_run_id", "--load_run_id", 
										help = "load_run_id", 
										default = "default")

	parser.add_argument("-load_name", "--load_name", 
										help = "load_name", 
										default = "default")

	parser.add_argument("-load_trained", "--load_trained", 
											help = "load_trained", 
											default = "default")

	parser.add_argument("-save_name", "--save_name", 
											help = "save_name", 
											default = "default")

	parser.add_argument("-save_run_id", "--save_run_id", 
										help = "save_run_id", 
										default = "default")

	#TRAIN SETUP
	parser.add_argument("-total_trials", "--total_trials", 
											help = "total_trials", 
											default = "default")

	parser.add_argument("-episodes_per_trial", "--episodes_per_trial", 
											help = "episodes_per_trial", 
											default = "default")

	parser.add_argument("-update_epochs", "--update_epochs", 
											help = "update_epochs", 
											default = "default")

	parser.add_argument("-num_minibatches", "--num_minibatches", 
											help = "num_minibatches", 
											default = "default")

	parser.add_argument("-lr_init", "--lr_init", 
											help = "lr_init", 
											default = "default")

	parser.add_argument("-lr_final", "--lr_final", 
											help = "lr_final", 
											default = "default")

	parser.add_argument("-pretrain_mode", "--pretrain_mode", 
											help = "pretrain_mode", 
											default = "default")

	#ENV SETUP
	parser.add_argument("-nmc_threshold", "--nmc_threshold", 
										help = "nmc_threshold", 
										default = "default")

	parser.add_argument("-nmc_n_seeds", "--nmc_n_seeds", 
											help = "nmc_n_seeds", 
											default = "default")

	#NMC SETUP
	parser.add_argument("-use_lbp", "--use_lbp", 
											help = "use_lbp", 
											default = "default")
	
	###NMC PARAMS
	parser.add_argument("-lbp_beta", "--lbp_beta", 
											help = "lbp_beta", 
											default = "default")

	parser.add_argument("-lbp_num_iters", "--lbp_num_iters", 
											help = "lbp_num_iters", 
											default = "default")

	parser.add_argument("-lbp_lambda_init", "--lbp_lambda_init", 
											help = "lbp_lambda_init", 
											default = "default")

	parser.add_argument("-lbp_lambda_final", "--lbp_lambda_final", 
											help = "lbp_lambda_final", 
											default = "default")

	parser.add_argument("-lbp_lambda_step", "--lbp_lambda_step", 
											help = "lbp_lambda_step", 
											default = "default")

	parser.add_argument("-lbp_tolerance_d", "--lbp_tolerance_d", 
											help = "lbp_tolerance_d", 
											default = "default")

	parser.add_argument("-nmc_neq_beta", "--nmc_neq_beta", 
											help = "nmc_neq_beta", 
											default = "default")
	args = parser.parse_args()
	

	#PROJECT SETUP
	with open(os.path.join("configs", f"{args.train_config}.json"), 'r') as f:
		train_config = json.load(f)

	with open(os.path.join("problems", 
												 train_config["problem_class"], 
												 f"{args.problem_config}.json"), 'r') as f:
		problem_config = json.load(f)
	
	train_config['seed'] = train_config['seed'] \
		if args.seed == "default" else int(args.seed)

	train_config["n_devices"] = train_config["n_devices"] \
		if args.n_devices == "default" else int(args.n_devices)

	train_config["n_replicas"] = train_config["n_replicas"] \
		if args.n_replicas == "default" else int(args.n_replicas)

	train_config["instances"] = \
		list(range(train_config["instances"][0][0], train_config["instances"][0][1]+1)) if args.id == "default" else \
		list(range(train_config["instances"][int(args.id)][0], train_config["instances"][int(args.id)][1]+1))

	
	#SAVE AND LOAD
	train_config["save"]["project_name"] = train_config["save"]["project_name"] \
		if args.project_name == "default" else args.project_name

	train_config["load"]["project_name"] = train_config["load"]["project_name"] \
		if args.load_project_name == "default" else args.load_project_name

	train_config["load"]["run_id"] = train_config["load"]["run_id"] \
		if args.load_run_id == "default" else args.load_run_id

	train_config["save"]["run_id"] = train_config["save"]["run_id"] \
		if args.save_run_id == "default" else args.save_run_id

	train_config["load"]["name"] = train_config["load"]["name"] \
		if args.load_name == "default" else args.load_name

	train_config["save"]["name"] = train_config["save"]["name"] \
		if args.save_name == "default" else args.save_name

	train_config["load"]["trained"] = train_config["load"]["trained"] \
		if args.load_trained == "default" else bool(int(args.load_trained))


	#ENV SETUP
	train_config["setup"]["nmc"]["threshold"] = train_config["setup"]["nmc"]["threshold"] \
		if args.nmc_threshold == "default" else float(args.nmc_threshold)

	train_config["setup"]["nmc"]["n_seeds"] = train_config["setup"]["nmc"]["n_seeds"] \
		if args.nmc_n_seeds == "default" else int(args.nmc_n_seeds)

	#NMC SETUP
	train_config["nmc_setup"]["use_lbp"] = train_config["nmc_setup"]["use_lbp"] \
		if args.use_lbp == "default" else bool(int(args.use_lbp))
	

	#TRAIN SETUP
	train_config["setup"]["num_minibatches"] = train_config["setup"]["num_minibatches"] \
		if args.num_minibatches == "default" else int(args.num_minibatches)

	train_config["setup"]["update_epochs"] = train_config["setup"]["update_epochs"] \
		if args.update_epochs == "default" else int(args.update_epochs)

	train_config["setup"]["total_trials"] = train_config["setup"]["total_trials"] \
		if args.total_trials == "default" else int(args.total_trials)

	train_config["setup"]["episodes_per_trial"] = train_config["setup"]["episodes_per_trial"] \
		if args.episodes_per_trial == "default" else int(args.episodes_per_trial)

	train_config["setup"]["lr_init"] = train_config["setup"]["lr_init"] \
		if args.lr_init == "default" else float(args.lr_init)

	train_config["setup"]["lr_final"] = train_config["setup"]["lr_final"] \
		if args.lr_final == "default" else float(args.lr_final)

	train_config["setup"]["pretrain_mode"] = train_config["setup"]["pretrain_mode"] \
		if args.pretrain_mode == "default" else args.pretrain_mode


	#NMC PARAMS
	problem_config["NMC"]["lbp_beta"] = problem_config["NMC"]["lbp_beta"] \
		if args.lbp_beta == "default" else float(args.lbp_beta)

	problem_config["NMC"]["lbp_num_iters"] = problem_config["NMC"]["lbp_num_iters"] \
		if args.lbp_num_iters == "default" else int(args.lbp_num_iters)

	problem_config["NMC"]["lbp_tolerance_d"] = problem_config["NMC"]["lbp_tolerance_d"] \
		if args.lbp_tolerance_d == "default" else float(args.lbp_tolerance_d)
	
	problem_config["NMC"]["nmc_neq_beta"] = problem_config["NMC"]["nmc_neq_beta"] \
		if args.nmc_neq_beta == "default" else float(args.nmc_neq_beta)

	lbp_lambdas = problem_config["NMC"]["lbp_lambdas"]
	if args.lbp_lambda_init != "default":
		lbp_lambdas[0] = float(args.lbp_lambda_init)
	if args.lbp_lambda_final != "default":
		lbp_lambdas[1] = float(args.lbp_lambda_final)
	if args.lbp_lambda_step != "default":
		lbp_lambdas[2] = float(args.lbp_lambda_step)
	problem_config["NMC"]["lbp_lambdas"] = lbp_lambdas


	print("Using devices:", jax.devices()[:train_config["n_devices"]])

	Jh_list = []
	colors_list = []

	N = problem_config["problem_setup"]["N"]
	M = problem_config["problem_setup"]["M"]

	problem_setup = None

	instances_path = os.path.join("problems", train_config["problem_class"], "instances")

	for iseed in train_config["instances"]:
		instance_file_name = (
			f"{train_config['problem_class']}_n{N}_m{M}_{iseed}"
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

		if train_config["nmc_setup"]["coloring"]:
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

	if train_config["nmc_setup"]["coloring"]:
		color_numbers = [c.shape[0] for c in colors_list]
		print("Average coloring:", np.mean(color_numbers))
		print("Min coloring:", np.min(color_numbers))
		print("Max coloring:", np.max(color_numbers))

	#exit when instances are generated
	if int(args.gen_instances_only):
		exit()

	train_config["ground_states"] = \
		train_config["ground_states"][0] if args.id == "default" else train_config["ground_states"][int(args.id)]
	
	if len(train_config["ground_states"]) > 1:
		assert len(train_config["ground_states"]) == len(train_config["instances"]) \
			and "Ground states length is not equal to instance seeds"
	
		print("Ground states:", train_config["ground_states"])
		train_config["ground_states"] = \
			jnp.array(train_config["ground_states"], dtype = Jh_list[0].h.dtype)

	elif len(train_config["ground_states"]) == 1:

		print("Ground state for all instances:", train_config["ground_states"][0])

		train_config["ground_states"] = jnp.full(
			(len(train_config["instances"]),), 
			train_config["ground_states"][0], 
			dtype = Jh_list[0].h.dtype
		)
			
	else:
		print("No ground states given")
		train_config["ground_states"] = None


	nmc_setup =  NMCSetup(
		approximation = train_config["nmc_setup"]["approximation"],
		
		track_stats = train_config["nmc_setup"]["track_stats"],
		problem_mode = train_config["nmc_setup"]["problem_mode"],

		coloring = train_config["nmc_setup"]["coloring"],

		energy_scale = train_config["nmc_setup"]["energy_scale"],

		reward_scale = train_config["nmc_setup"]["reward_scale"],

		numerical_eval_energy = train_config["nmc_setup"]["numerical_eval_energy"],

		use_lbp = train_config["nmc_setup"]["use_lbp"],

		lbp_beta = problem_config["NMC"]["lbp_beta"]
	)

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

	with open(os.path.join("modules", "policies", "nmc_ac.json"), 'r') as f:
		ac_config = json.load(f)
	
	checkpointer_model = ocp.StandardCheckpointer()
	
	if train_config["load"]["trained"]:
		# load a pre-trained model for benchmarking
		load_path = os.path.join(train_config["load"]["save_path"], 
														 train_config["load"]["project_name"], 
														 "MODELS",
														 f"model_{train_config['load']['run_id']}")

		load_file_path = os.path.join(load_path, train_config["load"]["name"])


		abstract_model = nnx.eval_shape(lambda: ActorCritic(
			Jh_list[0].h.size,
			ac_config, 
			rngs = nnx.Rngs(0))
		)
		graphdef, abstract_state, other_vars = nnx.split(abstract_model, nnx.Param, ...)

		if train_config["load"]["type"] == "json":
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
			
			abstract_state.replace_by_pure_dict(model_dict) #type: ignore  
			model_state = abstract_state

			print(f"Pre-trained model_state {load_file_path}, loaded via json")

		elif train_config["load"]["type"] == "orbax":
			#load the models from a checkpointer

			model_state = checkpointer_model.restore(os.path.abspath(load_file_path), abstract_state)

			print(f"Trained model_state {load_file_path} loaded via orbax")

		else:
			raise RuntimeError("Loading error: invalid type")

	else:
		print(f"No pretraining, a random model is used with seed {train_config['seed']}")

		model =	ActorCritic(
			Jh_list[0].h.size, ac_config, rngs = nnx.Rngs(train_config['seed'])
		)
		graphdef, model_state, other_vars = nnx.split(model, nnx.Param, ...)
	
	if train_config["save"]["log"]["wandb"]:
		wandb_run = wandb.init(
			project = train_config["save"]["project_name"],

			config = {
				"instances": train_config["instances"], 
				"ground_states": train_config["ground_states"],
				"problem_setup": problem_config["problem_setup"],

				"problem_class": train_config["problem_class"],
				"n_replicas": train_config["n_replicas"],
				"n_devices": train_config["n_devices"],
				"seed": train_config["seed"],

				"train_setup": train_config["setup"],

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

				"load_model": train_config["load"] if train_config["load"]["trained"] else None,
				"ac_config": ac_config,
				
				"save": train_config["save"]
			}
		)
		if args.save_run_id == "default":
			train_config["save"]["run_id"] = wandb_run.id  #type: ignore

	n_instances = len(Jh_list)
	n_instances_step = (n_instances//4 + 1) if n_instances > 4 else 1 #for saving data in logging

	if n_instances > 1:
		Jh_stacked = pad_and_stack_Jh(Jh_list) #type: ignore
		Jh_cat = pad_and_concatenate_Jh(Jh_list, nmc_setup.problem_mode)  #type: ignore

	else:
		Jh_cat = Jhdata(Jh_list[0].J, Jh_list[0].h, Jh_list[0].Jat)

		Jh_stacked = Jhdata(
			J = [Jsp(jnp.expand_dims(j.data, 0), jnp.expand_dims(j.indices, 0)) for j in Jh_list[0].J], 
			h = jnp.expand_dims(Jh_list[0].h, 0), 
			Jat = [Jsp(jnp.expand_dims(jat.data, 0), jnp.expand_dims(jat.indices, 0)) for jat in Jh_list[0].Jat]
		)

	if nmc_setup.coloring:
		if colors_list is None:
			raise RuntimeError("Please provide graph coloring!")
			
		colors_stack = pad_and_stack_colors(colors_list)

	else:
		colors_stack = None

	train_function = make_train(
		Jh_stacked = Jh_stacked,
		Jh_cat = Jh_cat,
		ground_states = train_config["ground_states"],
		colors_stack = colors_stack,
		nmc_setup = nmc_setup,
		params = params,
		n_replicas = train_config["n_replicas"],
		train_setup = train_config["setup"], 
		devices = jax.devices()[:train_config["n_devices"]], 
		project_dir = os.path.join(train_config["save"]["save_path"], train_config["save"]["project_name"]), 
		i_names = [f"{i}" for i in train_config["instances"]],
		seed = train_config["seed"]
	)
	

	TOTAL_TRAIN_STEPS = train_config["setup"]["total_trials"]//train_config["setup"]["trials_per_log"]

	EPISODES_PER_LOG = train_config["setup"]["trials_per_log"]*train_config["setup"]["episodes_per_trial"]
	NUM_UPDATES_PER_LOG = EPISODES_PER_LOG//train_config["setup"]["episodes_per_update"]

	NUM_STEPS_PER_UPDATE = train_config["setup"]["episodes_per_update"]*params.episode_steps

	def lr_schedule(counter):
		total_steps = (
			train_config["setup"]["total_trials"]*
			(train_config["setup"]["episodes_per_trial"]//train_config["setup"]["episodes_per_update"])*
			train_config["setup"]["num_minibatches"]*train_config["setup"]["update_epochs"]
		)
		ratio = jnp.log(train_config["setup"]["lr_final"]/train_config["setup"]["lr_init"])
		loglr = jnp.log(train_config["setup"]["lr_init"]) + (counter/total_steps)*ratio

		return jnp.exp(loglr)

	tx = optax.chain(
		optax.clip_by_global_norm(train_config["setup"]["max_grad_norm"]),
		optax.adam(learning_rate = lr_schedule)
	)

	model =	ActorCritic(Jh_cat.h.size, ac_config, rngs = nnx.Rngs(0))
	graphdef, _, other_vars = nnx.split(model, nnx.Param, ...)

	#create the running train state
	class TrainState(train_state.TrainState):
		other_vars: nnx.State

	t_state = TrainState.create(
		apply_fn = graphdef.apply, #type: ignore
		params = model_state,
		other_vars = other_vars,
		tx = tx
	)

	timer = 0
	time_total = time.time()

	key = jax.random.key(train_config['seed'])

	for log_step in range(TOTAL_TRAIN_STEPS):

		total_trials = train_config["setup"]["total_trials"]
		trials_per_log = train_config["setup"]["trials_per_log"]
		
		print(
			"Training trials:",
			f"{log_step*trials_per_log + 1}-{(log_step + 1)*trials_per_log} (out of {total_trials})..."
		)

		timer_old = timer
		time_a = time.time()

		t_state, key, train_info = train_function(key, t_state, log_step)

		train_info = jax.tree.map(lambda x: x.block_until_ready(), train_info)
		timer += time.time() - time_a
		print(f"Training complete in {time.time() - time_a} seconds, logging...")

		# to save the model state
		model_state = t_state.params
		
		if train_config["ground_states"] is not None:
			best_energies = (
				train_info["traj_info"]["best_energies"] -
				train_config["ground_states"][None, None, None, :]
			)

			energy_stats = {
				"best_energies": best_energies
			}

		else:

			energy_stats = {
				"best_energies": train_info["traj_info"]["best_energies"]
			}

		episode_returns = {}
		for i in range(0, n_instances):
			#track returns for each instance individually
			episode_returns[f"{i}"] = (
				train_info["traj_info"]["returned_episode_returns"][:, :, :, i][
					train_info["traj_info"]["returned_episode"]
				]
			)

		if nmc_setup.track_stats:
			landscape_stats = {
				"geometry": jax.tree.map(
					lambda x: x.reshape(NUM_UPDATES_PER_LOG, 
															NUM_STEPS_PER_UPDATE,
															train_config["n_replicas"]), 
					train_info["traj_info"]["geometry"]
				),

				"mc": jax.tree.map(
					lambda x: x.reshape(NUM_UPDATES_PER_LOG, 
															NUM_STEPS_PER_UPDATE,
															train_config["n_replicas"]), 
					train_info["traj_info"]["mc"]
				)
			} 
		else: 
			landscape_stats = None
		

		loss_info = {}
		for i in range(0, n_instances, n_instances_step):
			#track returns for each instance individually

			loss_info[f"{i}"] = {
				"total": train_info["loss_info"][1][0][:, :, :, i],

				"actor": train_info["loss_info"][1][1][:, :, :, i],
				"value": train_info["loss_info"][1][2][:, :, :, i],
				"entropy": train_info["loss_info"][1][3][:, :, :, i], 
				"allsame": train_info["loss_info"][1][4][:, :, :, i], 

				"rel_value": train_info["loss_info"][1][5][:, :, :, i],

				"clipping_rate": train_info["loss_info"][1][6][:, :, :, i], 
				"clipping_rate_vf": train_info["loss_info"][1][7][:, :, :, i]
			}

		#average losses over all instances
		loss_info["all"] = {
			"total": train_info["loss_info"][0],

			"actor": jnp.mean(train_info["loss_info"][1][1], axis = 3),
			"value": jnp.mean(train_info["loss_info"][1][2], axis = 3),
			"entropy": jnp.mean(train_info["loss_info"][1][3], axis = 3),
			"allsame": jnp.mean(train_info["loss_info"][1][4], axis = 3),

			"rel_value": jnp.mean(train_info["loss_info"][1][5], axis = 3),

			"clipping_rate": jnp.mean(train_info["loss_info"][1][6], axis = 3), 
			"clipping_rate_vf": jnp.mean(train_info["loss_info"][1][7], axis = 3)
		}

		def log_stats(update_step, 
									energy_stats, 
									landscape_stats, 
									loss_info):
			
			if train_config["save"]["log"]["wandb"]:
				train_wandb_logging(update_step = update_step, 
														energy_stats = energy_stats,
														landscape_stats = landscape_stats, 
														loss_info = loss_info)
			else:
				pass

		for update in range(NUM_UPDATES_PER_LOG):
			log_stats(update_step = (log_step*NUM_UPDATES_PER_LOG + update)*NUM_STEPS_PER_UPDATE, 
								
								energy_stats = jax.tree.map(lambda x: x[update], energy_stats),
								landscape_stats = jax.tree.map(lambda x: x[update], landscape_stats), 

								loss_info = jax.tree.map(lambda x: x[update], loss_info))

	 
		wandb_stats = {}
		for i in range(0, n_instances, n_instances_step):
			wandb_stats[f"episode/{i}/avg_returns"] = (
				jnp.mean(episode_returns[f"{i}"])
			)
			
			wandb_stats[f"episode/{i}/std_returns"] = (
				jnp.std(episode_returns[f"{i}"])
			)

			wandb_stats[f"episode/{i}/rel_std_returns"] = (
				wandb_stats[f"episode/{i}/std_returns"]/(wandb_stats[f"episode/{i}/avg_returns"] + 1e-9)
			)

			wandb_stats[f"episode/{i}/min_returns"] = (
				jnp.min(episode_returns[f"{i}"])
			)

			wandb_stats[f"episode/{i}/max_returns"] = (
				jnp.max(episode_returns[f"{i}"])
			)

		#compute average of the average across problem instances
		wandb_stats[f"episode/all_avg/avg_returns"] = jnp.mean(
			jnp.stack([wandb_stats[f"episode/{i}/avg_returns"] for i in range(0, n_instances, n_instances_step)])
		)
		wandb_stats[f"episode/all_avg/median_returns"] = jnp.median(
			jnp.stack([wandb_stats[f"episode/{i}/avg_returns"] for i in range(0, n_instances, n_instances_step)])
		)
		wandb_stats[f"episode/all_avg/std_returns"] = jnp.std(
			jnp.stack([wandb_stats[f"episode/{i}/avg_returns"] for i in range(0, n_instances, n_instances_step)])
		)
			

		wandb_stats["seconds_per_log"] = timer - timer_old
		wandb_stats["total_progress"] = (log_step+1)/TOTAL_TRAIN_STEPS

		current_step = (log_step+1)*NUM_STEPS_PER_UPDATE*NUM_UPDATES_PER_LOG

		if train_config["save"]["log"]["wandb"]:
			wandb.log(wandb_stats, step = current_step)

		print(f"Episode stats at step {current_step}:")
		pprint(wandb_stats)
		

		if train_config["save"]["save_data"]["model_every_log"]:
			save_path = (
				ocp.test_utils.erase_and_create_empty(
					os.path.abspath(
						os.path.join(
							train_config["save"]["save_path"], 
							train_config["save"]["project_name"],
							"MODELS",
							f"model_{train_config['save']['run_id']}"
						)
					)
				)
			)

		if train_config["save"]["save_data"]["model_every_log"]:
			checkpointer_model.save(save_path / f"{train_config['save']['name']}", model_state) #type: ignore
			checkpointer_model.wait_until_finished()
			print(f"Saved model at {time.time() - time_total} seconds | {log_step} log step")
					
	if train_config["save"]["save_data"]["model_final"]:
		save_path = (
			ocp.test_utils.erase_and_create_empty(
					os.path.abspath(
						os.path.join(
							train_config["save"]["save_path"], 
							train_config["save"]["project_name"], 
							"MODELS",
							f"model_{train_config['save']['run_id']}"
						)
					)
				)
		)

		#save the final state of the model
		checkpointer_model.save(save_path / f"{train_config['save']['name']}", model_state) #type: ignore
		checkpointer_model.wait_until_finished()
		print(f"Saved final model at {time.time() - time_total} seconds")
		
	print("FINISHED TRAINING")
