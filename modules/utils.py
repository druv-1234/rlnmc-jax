import jax
import jax.numpy as jnp
import numpy as np
import wandb

from typing import Dict, Any, Union, List

from functools import partial

@partial(jax.jit, static_argnames = ["bootstrap_size"])
def tts_bootstrap(energies: jax.Array, 
                  total_times: jax.Array,
                  approximation: jax.Array, 
                  key: jax.Array,
                  percentiles: jax.Array = jnp.array([10, 20, 50, 80, 90]),
                  bootstrap_size: int = 1000) -> Dict[Any, jax.Array]:
    """
    TTS bootstrapping from the parallel replica stats

    Parameters:
      energies (jax.Array): energies relative to the ground states for each instances
      total_times (jax.Array): lengths of runs for each instance
      approximation: (float) approximation ratios for each instance
      key (jax.Array): jax random key
      bootstrap_size: size of the bootstrap samples
      percentiles: bootstrap tts percentiles
    """
    k = energies.shape[0]

    def get_success(e, approx):
      succ = (e <= approx)
      # check if numerically close too:
      e_is_close = jnp.logical_or(jnp.isclose(e, approx, atol = 0), jnp.isclose(approx, e, atol = 0))
      succ = jnp.logical_or(succ, e_is_close).sum()
      
      fail = e.size - succ
      return succ, fail
    successes, failures = jax.vmap(get_success)(energies, approximation)
    
    keys = jax.random.split(key, bootstrap_size)
    def sample_tts(key: jax.Array):
      subkey1, subkey2 = jax.random.split(key)
      new_I = jax.random.choice(subkey1, jnp.arange(0, k), (k,), replace = True)
      keys = jax.random.split(subkey2, k)

      def get_tts(i, key):
        pos = jax.random.beta(key, successes[i] + 0.5, failures[i] + 0.5)
        r = jnp.log(0.01)/jnp.log(1.0 - jnp.clip(pos, 1e-15, 1.0 - 1e-15))
        tts = jnp.where(r < 1, total_times[i], total_times[i]*r)
        return tts
      return jax.vmap(get_tts)(new_I, keys)
    
    all_tts_samples = jax.vmap(sample_tts)(keys)

    tts_means = jnp.mean(all_tts_samples, axis = 0)
    tts_stds = jnp.std(all_tts_samples, axis = 0)

    tts_percentiles = jnp.array([jnp.percentile(all_tts_samples, p, axis = 1) for p in percentiles])
    tts_p_means = jnp.mean(tts_percentiles, axis = 1)
    tts_p_stds  = jnp.std(tts_percentiles, axis = 1)
        
    return {
       "successes": successes,
       "failures": failures,
       "tts_means": tts_means,
       "tts_stds": tts_stds,
       "tts_p_means": tts_p_means,
       "tts_p_stds": tts_p_stds
    }

def get_stats(func, data: Union[Dict, jax.Array], axis = 0):
  stats = {}
  if func is None:
    stats["mean"] = jax.tree.map(lambda x: jnp.mean(x, axis = axis), data)
    # stats["std"]    = jax.tree.map(lambda x: jnp.std(x, axis = axis), data)

    # stats["min"]  = jax.tree.map(lambda x: jnp.min(x, axis = axis), data)  
    # stats["max"]  = jax.tree.map(lambda x: jnp.max(x, axis = axis), data)

  else:
    stats["mean"] = jax.tree.map(lambda x: func(jnp.mean(x, axis = axis)), data)
    # # stats["std"]    = jax.tree.map(lambda x: func(jnp.std(x, axis = axis)), data)
    
    # stats["min"]  = jax.tree.map(lambda x: func(jnp.min(x, axis = axis)), data)  
    # stats["max"]  = jax.tree.map(lambda x: func(jnp.max(x, axis = axis)), data)

  return stats


def train_wandb_logging(update_step: int, 
                        energy_stats: Dict,
                        landscape_stats: Dict, 
                        loss_info: Dict):
  wandb_stats = {}

  steps_per_update = jax.tree_leaves(energy_stats)[0].shape[0]
  n_instances = jax.tree_leaves(energy_stats)[0].shape[2]
  
  #don't log all steps, too much data
  log_n_steps = steps_per_update if steps_per_update < 3 else 3

  for i in np.linspace(0, steps_per_update, log_n_steps, True, dtype = int):
    if i == steps_per_update:
      step_i = i.item() - 1
    else:
      step_i = i.item()

    if landscape_stats is not None:
      if landscape_stats["mc"] is not None:
        wandb_stats[f"stats/mc"] = \
          get_stats(None, 
                    jax.tree.map(lambda x: x[step_i, :], landscape_stats['mc']), 
                    axis = 0)
        
      if landscape_stats["geometry"] is not None:
        wandb_stats[f"stats/geometry"] = \
          get_stats(None, 
                    jax.tree.map(lambda x: x[step_i, :], landscape_stats['geometry']), 
                    axis = 0)
      
    for inst_i in range(0, n_instances, 4):
      wandb_stats[f"energy/{inst_i}"] = \
        get_stats(None, jax.tree.map(lambda x: x[step_i, :, inst_i], energy_stats), axis = 0)

    wandb_stats[f"energy/avg"] = \
      get_stats(None, jax.tree.map(lambda x: jnp.mean(x[step_i], axis = 0), energy_stats), axis = 0)

    wandb_stats[f"energy/rep_min_inst_avg"] = \
      get_stats(None, jax.tree.map(lambda x: jnp.min(x[step_i], axis = 0), energy_stats), axis = 0)
    wandb_stats[f"energy/inst_min_rep_avg"] = \
      get_stats(None, jax.tree.map(lambda x: jnp.min(x[step_i], axis = 1), energy_stats), axis = 0)

    wandb.log(data = wandb_stats, step = update_step + step_i + 1)

  wandb_stats = {}
  if loss_info is not None:
    
    #rl loss stats averaged over instances
    wandb_stats[f"loss_epoch_mean_all/"] = get_stats(jnp.mean, loss_info["all"], axis = 1)
    wandb_stats[f"loss_epoch_min_all/"] = get_stats(jnp.min, loss_info["all"], axis = 1)

    #instance-wise rl loss stats
    # for k, v in loss_info.items():
    #   if k != "all":
    #     wandb_stats[f"loss_epoch_mean_iw/{k}"] = get_stats(jnp.mean, v, axis = 1)
    #     wandb_stats[f"loss_epoch_min_iw/{k}"] = get_stats(jnp.min, v, axis = 1)

  log_step = update_step + steps_per_update
  wandb.log(data = wandb_stats, step = log_step)


def bench_wandb_logging(sweep: int, 
                        sweeps_per_step: int,
                        energy_stats: Dict,
                        landscape_stats: Union[Dict, None],
                        metric: Union[Dict, None],
                        other: Union[Dict, None]):
  
  log_shape = jax.tree_leaves(energy_stats)[0].shape
  n_steps = log_shape[0]

  #don't log all steps, too much data
  log_n_steps = n_steps if n_steps < 10 else 10

  for i in np.linspace(0, n_steps, log_n_steps, True, dtype = int):
    if i == n_steps:
      step_i = i.item() - 1
    else:
      step_i = i.item()

    wandb_stats = {}
    
    if landscape_stats is not None:
      if landscape_stats["mc"] is not None:
        wandb_stats[f"stats/mc"] = \
          get_stats(None, jax.tree.map(lambda x: x[step_i], landscape_stats["mc"]), axis = 0)

      if landscape_stats["geometry"] is not None:
        wandb_stats[f"stats/geometry"] = \
          get_stats(None, jax.tree.map(lambda x: x[step_i], landscape_stats["geometry"]), axis = 0)
    
    # for j, i_name in enumerate(instance_names):      
    #   wandb_stats[f"{i_name}/energy"] = \
    #     get_stats(None, jax.tree.map(lambda x: x[step_i, :, j], energy_stats["best_e"]), axis = 0)

    wandb_stats[f"energy/avg"] = \
      get_stats(None, jax.tree.map(lambda x: jnp.mean(x[step_i],  axis = 0), energy_stats["best_e"]), axis = 0)
      
    wandb_stats[f"energy/rep_min_inst_avg"] = \
      get_stats(None, jax.tree.map(lambda x: jnp.mean(x[step_i], axis = 0), energy_stats["best_e"]), axis = 0)
    wandb_stats[f"energy/inst_min_rep_avg"] = \
      get_stats(None, jax.tree.map(lambda x: jnp.min(x[step_i],  axis = 1), energy_stats["best_e"]), axis = 0)

    wandb.log(data = wandb_stats, step = sweep + (step_i - n_steps + 1)*sweeps_per_step)

  wandb_stats = {}
        
  if metric is not None:
    for key, value in metric.items():
      wandb_stats[f"metric/{key}"] = value

  if other is not None:
    for key, value in other.items():
      wandb_stats[f"other/{key}"] = value
		
  wandb.log(data = wandb_stats, step = sweep)