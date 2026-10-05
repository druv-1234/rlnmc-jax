import sys
sys.path.append('.')

import jax
import jax.numpy as jnp
from jax import lax

from flax import struct

from gymnax.environments import environment

from modules import nmc
from modules import lbp
from modules import mc_corr

#Fold constant for the geometry PRNG key. Any fixed value works - what matters is that it
#is DERIVED from the step/reset key by fold_in rather than taken from a split, so adding a
#sampling geometry source does not shift the stream positions every other consumer reads
#from and the Hi/LBP paths stay bit-for-bit identical.
_GEOM_KEY_FOLD = 0x6D63      #"mc"

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

	#divide J, h and Jat by max|J| (Nonlocal-Monte-Carlo/NMC/cnmc.py:471-476).
	#recorded here for logging/reproducibility - applied in bench.py at load time
	normalize_weights: bool = False

	#which form of the LBP quantities feeds backbone selection. The surrogate pins every
	#spin to its current value with a lambda*epsilon*s field; "depinned" removes that
	#artifact from both `mag` and J_tilde, "pinned" keeps it (the Nonlocal-Monte-Carlo
	#convention). Only meaningful when use_lbp is true - the Hi heuristic has no surrogate
	#and no pin. Both forms are always logged; this only selects which one is thresholded.
	lbp_pinning: str = "depinned"

	#greedy_color strategy used to build the color classes. Recorded so a run's config
	#identifies the coloring it actually ran under, and so the reset cache can key on it.
	coloring_strategy: str = "DSATUR"

	#WHICH QUANTITY selects the backbone:
	#  "hi"       - the local field / energy gradient heuristic  (per VARIABLE)
	#  "lbp_mag"  - LBP marginals of the surrogate               (per VARIABLE)
	#  "lbp_corr" - effective couplings J_tilde, Eq. (15)        (per FACTOR)
	#  "mc_corr"  - the SAME J_tilde, estimated by direct MC     (per FACTOR)
	#
	#The first two are what use_lbp already selected between; bench.py derives one of the
	#pair from the other so no shipped config changes meaning. "lbp_corr" is the NMC paper's
	#Sec. V rule and is a different KIND of threshold, not a retuning: |J_tilde| lives on a
	#factor, so a k-body term is rigid or not as a unit and all k members enter together.
	#Requires use_lbp and problem_mode "pising" - Eq. (C3) presumes an Ising k-body factor.
	#
	#"mc_corr" estimates the IDENTICAL quantity as "lbp_corr" - the k-local factor
	#correlation of Eq. (13)/(C3) and its J_tilde - by sampling the surrogate directly
	#(modules/mc_corr.py) instead of reading it off converged LBP messages. It is a
	#second SOURCE, not a second rule: `clustering.factor_seed_mask`/`factor_clustering`
	#consume its output unchanged. It needs NO factor graph, so it requires use_lbp
	#FALSE - the two are mutually exclusive arms and a comparison runs them separately.
	#
	#None means "derive from use_lbp" (False -> "hi", True -> "lbp_mag"), which is what
	#every caller that predates this field gets. NMCgym resolves it once at construction, so
	#self.nmc_setup.backbone_source is always a concrete string.
	backbone_source: Union[str, None] = None

	#MC correlation source settings, read only when backbone_source is "mc_corr".
	#
	#Every one of these is STATIC: mc_corr.MCCfg's fields become `lax.scan` trip counts
	#and a `lax.while_loop` bound, and the thresholds are compile-time constants. That is
	#why they live here and not in NMCParams - `bench_apt.py`'s `_axes_for` builds its
	#vmap axes by tree-mapping NMCParams to None and would have to learn about any new
	#traced leaf. Nothing here is per-replica.
	#
	#mc_beta is the SAMPLING temperature and is deliberately NOT the replica's rung. MC
	#estimates <prod r_i> by counting how often the product is -1; at an NMC rung's beta
	#(9.6-48.5 on chimera128) that costs e^(-2*beta*h) per flip, so the chain freezes,
	#every factor reports |corr| = 1 and J_tilde collapses to the bare |J_a|. Measured on
	#normalised chimera128: 15% of factors saturated at beta 9.64 and 51% at 48.49, and at
	#9.64 the ladder has no safe rung at all. LBP is immune because it never samples - it
	#evaluates the cavity product in log space. The usable window there is beta ~ 2.5-4.5;
	#below it the surrogate is paramagnetic (p = q1 = q0 = 0, nothing to detect) and above
	#it saturates. See `mc_corr.corr_to_Jtilde`.
	#
	#CONSEQUENCE, accepted deliberately: the MC backbone describes rigidity AT mc_beta,
	#not at the rung. That is a different physical question from the LBP arm's, and
	#whether it makes a better move proposal is the empirical question the comparison
	#answers. It also means theta_seed/theta_cut must be re-swept for this arm.
	mc_beta: float = 3.0

	#"sample" divides the surrogate term by mc_beta - Eq. (15) exactly. "rung" divides by
	#the replica's own beta instead, expressing J_tilde in the rung's units. "rung" is
	#available but not recommended: it makes the surrogate contribution O(1/beta_rung), so
	#it SHRINKS with the rung rather than converging to the cavity-field limit LBP reaches.
	#It rescales; it does not recover the information.
	mc_beta_div: str = "sample"

	#Overrides lbp_lambdas[2] (the geometric ratio) for the MC ladder. The START is derived
	#by mc_corr's increase phase from mc_cfg.lambda_seed, not configured here.
	#None keeps lbp_lambdas[2].
	mc_lambda_ratio: Union[float, None] = None

	#Overrides lbp_lambdas[1] (the floor) for the MC ladder. The LBP floor of 0.001 is far
	#below anything meaningful here: measured on normalised chimera128, at lam = 0.03 the
	#overlap p is <= 0.35 at EVERY beta from 1.5 to 48.5, i.e. the surrogate is definitively
	#not localised and its correlations have gone state-INDEPENDENT - the degenerate regime,
	#not an operating point. Descending past it only burns monitor phases before reporting
	#`floor_reached`. None keeps lbp_lambdas[1].
	mc_lambda_min: Union[float, None] = None

	#Stop sensitivity. Measured plateau on chimera128: pq_sigma anywhere in [3, 12] picks
	#the identical operating lambda, and split_ratio is inert from 2 to 25 - the directed
	#p-vs-q1 test dominates. Run-to-run PRNG noise moves the operating lambda more than a
	#4x sweep of pq_sigma does, which is why these are fixed here rather than searched.
	mc_pq_sigma: float = 6.0
	mc_split_ratio: float = 5.0

	#Sweep budget and clone count, as an mc_corr.MCCfg. None means MCCfg's own defaults,
	#which are calibrated (see its docstring - monitor_block_len 50 is a floor, not a
	#preference). Carried as the concrete NamedTuple so it reaches surrogate_MC as one
	#static argument.
	mc_cfg: Union[object, None] = None

	#Backbone temperature convention. None keeps the shipped schedule, which ramps from the
	#ABSOLUTE nmc_neq_beta up to the base beta inside every env step - so the boost ratio
	#drifts across a run (38x -> 60x on chimera128 with the shipped config). A float instead
	#pins the hot end to beta/backbone_temp_x, the constant-multiplier convention of
	#Nonlocal-Monte-Carlo (cnmc.py:371, temp_x = 20 in all five shipped examples); the NMC
	#paper quotes 2-10x (Sec. VII). A multiplier and an absolute floor cannot both be the
	#default, which is why this is its own switch.
	backbone_temp_x: Union[float, None] = None

	#DOES THE BACKBONE BOOST FADE WITHIN AN ENV STEP? True (default) keeps the shipped
	#behaviour: nmc_neq_beta is ramped geometrically from the hot end up to the base beta
	#across the nmc_neq_cycles, so the boost decays inside every step - measured on the
	#uf1000 ladder, 10x at cycle 0 down to 1.26x at cycle 9 (beta_range_neq is exactly 0
	#under APT, where beta_start = beta_f, so the ramp reduces to beta*10**(-i/cycles)).
	#False holds the boost CONSTANT at beta/backbone_temp_x for every cycle, which is what
	#the reference does - cnmc.py/npt.py divide J and h by a fixed temp_x once, outside the
	#cycle loop, and never re-scale.
	neq_beta_anneal: bool = True

	#LBP AT THE RUNG'S OWN TEMPERATURE. False (default) runs the surrogate at the fixed
	#nmc_setup.lbp_beta for every replica, which is what every result before 2026-09-17 used.
	#True makes the surrogate temperature FOLLOW THE LADDER: the effective beta is the
	#replica's own rung (params.beta_start, which bench_apt.py sets equal to beta_f per rung),
	#so J_tilde describes the distribution that replica is actually sampling rather than one
	#fixed reference temperature.
	#
	#This is free because temperature enters the factor graph as ONE GLOBAL MULTIPLY on the
	#energy table while the wiring is temperature-free - so the graph is built once at
	#nmc_setup.lbp_beta and its table is rescaled per replica at run time. No rebuild, and the
	#five NMC rungs still go through a single vmapped call. The scale applied is the RATIO
	#rung_beta/nmc_setup.lbp_beta, so the graph's own build temperature cancels and the
	#EXISTING cached factor graphs are reused unchanged - nmc_setup.lbp_beta comes from
	#problem_config["NMC"]["lbp_beta"] (5.0 on uf1000), not from the bench config, and nothing
	#needs rebuilding at 1.0.
	#
	#MEASURED CONSEQUENCES, both accepted deliberately: the pinning field is epsilon*s*beta*lmbd
	#and lmbd floors at 0.02 for beta >= 2, so the COLDEST rung ends up the most pinned
	#(0.095 at beta 4.76 vs 0.197 at beta 9.85); and at fixed theta a colder rung clears more
	#factors, so the backbone GROWS toward the cold end rather than shrinking.
	lbp_beta_from_rung: bool = False

	#WHAT LEAVES AN NMC ENV STEP.
	#  "argmin"   - the shipped behaviour: Metropolis-test the minimum-energy state seen
	#               anywhere in the step (the NMC cycles AND the equilibrium phase) against
	#               the state the step began from, and carry the winner.
	#  "endpoint" - carry the equilibrium phase's ENDPOINT, which is what the reference hands
	#               its swap (npt.py:647, m_start = M[:, -1]).
	#
	#The reference DOES restart each NMC phase from an argmin (np.argmin(energies) between
	#phases), so "argmin" is faithful INSIDE the subroutine; what differs is the state that
	#leaves it. Since min_e <= old_e almost always at cold temperatures the Metropolis gate is
	#near-always open and the walker's energy is effectively monotone. Under APT that matters
	#at one specific place - the Gibbs -> NMC rung boundary, where an NMC rung holding a
	#ratcheted-low energy makes (beta_hi - beta_lo)(E_hi - E_lo) strongly negative, so good
	#configurations found in the Gibbs region cannot cross into the NMC region.
	#
	#Default "argmin" so bench.py, the sf250_fork_1e5 data on disk and the RLNMC checkpoints
	#keep their exact meaning. bench_apt.py opts in to "endpoint". best_s/best_e track the
	#running minimum separately and are unaffected either way.
	nmc_carry: str = "argmin"

	#WHERE the LBP surrogate runs. True keeps it as the last statement of every env step,
	#which is what bench.py wants. False makes step_env return a zeroed `mag` and jt = None
	#and leaves the caller to compute geometry itself.
	#
	#bench_apt.py sets False because under APT the step is split across a ladder and only the
	#coldest few rungs select a backbone - the rest are plain Gibbs and discard it. On uf1000
	#that is 16 of 21 rungs running a full LBP solve per step for nothing, and LBP is ~22x the
	#cost of a sweep lane. It also lets the caller compute geometry AFTER the replica
	#exchange, so an NMC rung derives its backbone from the configuration it actually ends the
	#step holding rather than the one it started with.
	geometry_in_step: bool = True

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

	#THE UNLEARNING PHASE, per NMC cycle. 0 (default) reproduces the previous behaviour
	#exactly: the only unhomogeneous-free relaxation is the single equilibrium phase at the
	#end of the env step. A positive value runs that many FULL-system sweeps at the rung's
	#own beta, nothing frozen, after each cycle's two inhomogeneous phases - the reference's
	#third phase (cnmc.py:420-431 / npt.py, gated there by full_update_frequency, which is 1
	#in every shipped example, i.e. every cycle).
	#
	#The paper, Sec. VIII: "we invoke unlearning phases in which we apply standard (local and
	#homogeneous) MCMC frequently (after each application of the inhomogeneous MCMC) to
	#mitigate the inductive bias in our model for the backbones".
	#
	#Static, like the other sweep counts: it is a lax.scan trip count.
	nmc_nsw_unlearn: int = struct.field(default = 0, pytree_node=False)

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

	#Effective couplings J_tilde (Eq. 15) for the CURRENT state, flat over all factors in
	#np.concatenate(Jt) order - i.e. factor ids from 0, with NO bias factors, which is not
	#init_correlation_index's numbering (that starts at N). None unless backbone_source is
	#"lbp_corr". Defaulted so the field is additive: it costs nothing on the other paths and
	#does not invalidate a pickled warm-up written before it existed.
	jt: Union[jax.Array, None] = None


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

		#backbone_source is what actually selects the backbone; use_lbp only says whether the
		#surrogate runs at all. An unset source derives from use_lbp, which is what callers
		#predating the field (train.py, older tests) get - so adding it broke nothing.
		if self.nmc_setup.backbone_source is None:
			self.nmc_setup = self.nmc_setup._replace(
				backbone_source = "lbp_mag" if self.nmc_setup.use_lbp else "hi"
			)

		#The rest fire loudly rather than producing an all-False backbone at run time, which
		#is what a missing J_tilde would silently look like.
		if self.nmc_setup.backbone_source not in ("hi", "lbp_mag", "lbp_corr", "mc_corr"):
			raise ValueError(
				f"unknown backbone_source {self.nmc_setup.backbone_source!r}; expected one of "
				f"'hi', 'lbp_mag', 'lbp_corr', 'mc_corr'"
			)

		#"mc_corr" is the one source that needs NO factor graph, so it is excluded here and
		#required to be use_lbp FALSE just below. Building the graph for it would cost a
		#~400 s pgmax construction whose output is never read.
		if (self.nmc_setup.backbone_source not in ("hi", "mc_corr")
				and not self.nmc_setup.use_lbp):
			raise ValueError(
				f"backbone_source {self.nmc_setup.backbone_source!r} needs the LBP surrogate, "
				f"but use_lbp is False. Set use_lbp true, or use backbone_source 'hi'."
			)

		if self.nmc_setup.backbone_source == "hi" and self.nmc_setup.use_lbp:
			raise ValueError(
				"backbone_source 'hi' with use_lbp true would run LBP and then throw its "
				"result away. Set use_lbp false, or use backbone_source 'lbp_mag'."
			)

		if self.nmc_setup.backbone_source == "mc_corr" and self.nmc_setup.use_lbp:
			raise ValueError(
				"backbone_source 'mc_corr' with use_lbp true would build the pgmax factor "
				"graph and never read it. The MC and LBP correlation sources are mutually "
				"exclusive arms - run them as separate benchmarks and compare the results. "
				"Set use_lbp false."
			)

		if (self.nmc_setup.backbone_source in ("lbp_corr", "mc_corr")
				and self.nmc_setup.problem_mode != "pising"):
			raise ValueError(
				f"backbone_source {self.nmc_setup.backbone_source!r} is only defined for "
				f"pising, got {self.nmc_setup.problem_mode!r}. Eq. (C3) presumes an Ising "
				f"k-body factor J*prod(s); a wcnf clause indicator is a different object, so "
				f"the effective couplings would be meaningless rather than merely inaccurate."
			)

		if self.nmc_setup.backbone_source == "mc_corr":
			#The sweep is a CHROMATIC Gibbs scan - spins of one colour share no factor, so a
			#whole colour class resamples in parallel. Without a colouring there is no scan.
			if not self.nmc_setup.coloring:
				raise ValueError(
					"backbone_source 'mc_corr' requires coloring true: mc_corr samples the "
					"surrogate with a chromatic Gibbs sweep, which has no uncoloured form here."
				)

			#De-pinning is a CAVITY-FIELD operation. MC observes configurations, not messages,
			#and the only exact removal is an importance reweighting whose variance is exactly
			#what the pin exists to avoid. Raising rather than ignoring, because every shipped
			#chimera128 config sets "depinned" and a silent downgrade to "pinned" would make
			#an MC arm quietly incomparable with the LBP arm it is being benchmarked against.
			if self.nmc_setup.lbp_pinning != "pinned":
				raise ValueError(
					f"backbone_source 'mc_corr' requires lbp_pinning 'pinned', got "
					f"{self.nmc_setup.lbp_pinning!r}. De-pinning subtracts lambda*eps*s from "
					f"each CAVITY field, which has no Monte Carlo analogue - MC observes "
					f"configurations, not messages. Set lbp_pinning 'pinned' here, and compare "
					f"against an LBP arm that is also 'pinned' or the two measure different "
					f"things."
				)

			#mc_beta is explicitly NOT the rung (see the field's comment): at an NMC rung's
			#beta the chain freezes and every factor saturates. Letting this flag through
			#would silently produce a J_tilde equal to the bare |J_a|.
			if self.nmc_setup.lbp_beta_from_rung:
				raise ValueError(
					"backbone_source 'mc_corr' cannot use lbp_beta_from_rung: it would sample "
					"at the replica's own rung (up to beta 48 on chimera128), where a spin flip "
					"costs e^-48, the chain never moves, every factor reports |corr| = 1 and "
					"J_tilde collapses to the bare |J_a|. Set it false and use nmc_setup.mc_beta "
					"(order 1-5); nmc_setup.mc_beta_div 'rung' expresses J_tilde in rung units "
					"if that is what was wanted."
				)

			if self.nmc_setup.mc_beta_div not in ("sample", "rung"):
				raise ValueError(
					f"mc_beta_div must be 'sample' or 'rung', got "
					f"{self.nmc_setup.mc_beta_div!r}"
				)

			if self.nmc_setup.mc_beta <= 0:
				raise ValueError(f"mc_beta must be positive, got {self.nmc_setup.mc_beta}")

			#q0 is the pairwise CLONE overlap (Eq. B4) and does not exist at R = 1. Checked
			#here rather than left to surrogate_MC's own assert, which fires at trace time
			#inside a pmap and reports a much less useful location.
			_cfg = self.nmc_setup.mc_cfg
			if _cfg is not None and _cfg.replicas < 2:
				raise ValueError(
					f"mc_cfg.replicas must be >= 2, got {_cfg.replicas}: q0 is the overlap "
					f"between two DIFFERENT clones (Eq. B4) and has no single-chain substitute "
					f"- a block-based one tracks q1 all the way down and detects nothing."
				)

		if (self.nmc_setup.backbone_temp_x is not None
				and self.nmc_setup.backbone_temp_x <= 0):
			raise ValueError(
				f"backbone_temp_x must be positive (it divides beta), got "
				f"{self.nmc_setup.backbone_temp_x}"
			)

		self.N0 = self.Jh_stack.h[0].size
		self.N = self.Jh_cat.h.size

		#The surrogate's localisation scale and the per-factor member indices. HOISTED out of
		#the use_lbp branch below: both are plain array reductions over Jh_cat with no pgmax
		#in them, and backbone_source "mc_corr" needs both while building no factor graph at
		#all. Computing them unconditionally costs one pass over the couplings.
		if self.nmc_setup.problem_mode == "pising":
			#member spin indices per factor, (n_terms, k) per order group: needed both to
			#gather the surrogate pinning field onto each factor slot and to label the
			#stats by arity rather than by list position
			self.corr_idx = [j.indices for j in self.Jh_cat.J]
			self.corr_J = [j.data for j in self.Jh_cat.J]
			#the same couplings flattened into `concatenate(Jt)` order - the ordering
			#clustering.py indexes with, and what mc_corr.corr_to_Jtilde needs
			self.J_flat_cat = jnp.concatenate([j.data for j in self.Jh_cat.J])
		else:
			self.corr_idx, self.corr_J, self.J_flat_cat = None, None, None

		self.epsilon = \
			jnp.sum(jnp.concatenate([jnp.sum(jnp.abs(j.data), axis = 1, keepdims = True)
																for j in self.Jh_cat.Jat], axis = 1), axis = 1)
		self.epsilon += jnp.abs(self.Jh_cat.h)

		#CAT-SPACE COLOUR CLASSES, for the mc_corr sweep only.
		#
		#colors_stack is (n_instances, n_colors, max_class) in INSTANCE-LOCAL index space:
		#entries run 0..N0-1 and padding is N0. mc_corr samples Jh_cat, where instance i's
		#spins are offset by i*N0 (read_utils.pad_and_concatenate_Jh), so passing a raw
		#colors_stack[i] would (a) only ever touch the first N0 spins and (b) write to spin
		#N0 - a VALID cat index - wherever the pad appears, because `.at[].set(mode='drop')`
		#only drops out-of-range indices. Remap the pad to self.N explicitly.
		#
		#Colour class c of two different instances share no factor (they share no spins at
		#all), so they merge into one class and the class count is unchanged.
		if self.nmc_setup.backbone_source == "mc_corr" and self.colors_stack is not None:
			self.colors_cat = jnp.concatenate(
				[jnp.where(self.colors_stack[i] == self.N0,
									 self.N,
									 self.colors_stack[i] + i*self.N0)
				 for i in range(self.n_instances)],
				axis = 1
			)
		else:
			self.colors_cat = None

		#initialize the lbp of the factor graph
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

			#static index arrays for reading k-local correlations off the converged messages.
			#Only defined for pising - Eq. (C3) presumes an Ising k-body factor, so it is
			#meaningless (not merely inaccurate) for wcnf clause-indicator factors.
			if self.nmc_setup.problem_mode == "pising":
				self.corr_vsfe0, self.corr_rows = lbp.init_correlation_index(
					self.FG, self.Jh_cat.J, self.N, self.nmc_setup.problem_mode
				)
				print("Correlation index initialized")
			else:
				self.corr_vsfe0, self.corr_rows = None, None
		else:
			self.corr_vsfe0, self.corr_rows = None, None


	@partial(jax.jit, static_argnums = (0))
	def get_geometry(self,
									 state: jax.Array,
									 params,
									 key = None):
		"""
		(magnetizations, J_tilde_flat, geometry_stats) for the current state.

		`key` is required ONLY by backbone_source "mc_corr", which samples; the LBP and Hi
		paths are deterministic and ignore it. Optional rather than positional so the call
		sites that predate it keep working unchanged.

		DISPATCH IS ON backbone_source, NOT use_lbp. It used to be on use_lbp, which worked
		only because the two agreed - "mc_corr" is the first source that needs neither the
		factor graph nor the Hi heuristic, and under the old branch it would have fallen
		through to the LBP arm and SILENTLY RETURNED LBP's J_tilde under an MC label.
		"""
		if self.nmc_setup.backbone_source == "mc_corr":
			if key is None:
				raise ValueError(
					"get_geometry needs a PRNG key under backbone_source 'mc_corr': it "
					"estimates the correlations by sampling. Pass key = <jax PRNG key>."
				)

			cfg = self.nmc_setup.mc_cfg if self.nmc_setup.mc_cfg is not None else mc_corr.MCCfg()

			#lambdas[0] is only where the INCREASE phase starts looking; mc_corr derives the
			#real starting lambda from beta and epsilon, because a lambda calibrated for LBP
			#does not pin an MC chain (see mc_corr.MCCfg). The floor and the ratio DO carry
			#over, so the two arms descend on the same geometry.
			_ratio = (params.lbp_lambdas[2] if self.nmc_setup.mc_lambda_ratio is None
								else self.nmc_setup.mc_lambda_ratio)
			_lam_min = (params.lbp_lambdas[1] if self.nmc_setup.mc_lambda_min is None
									else self.nmc_setup.mc_lambda_min)
			_lambdas = jnp.array([params.lbp_lambdas[0], _lam_min, _ratio])

			#"rung" expresses J_tilde in the replica's own units; params.beta_start is that
			#rung under bench_apt.py and is a TRACED leaf, so each vmapped replica carries
			#its own. The sampling temperature is mc_beta either way.
			_beta_div = (None if self.nmc_setup.mc_beta_div == "sample"
									 else params.beta_start)

			corr, J_tilde_flat, mag_cat, mc_stats = mc_corr.surrogate_MC(
				self.Jh_cat,
				state,
				self.epsilon,
				self.colors_cat,
				self.J_flat_cat,
				self.corr_idx,
				self.nmc_setup.mc_beta,
				_lambdas,
				key,
				cfg,
				self.nmc_setup.mc_pq_sigma,
				self.nmc_setup.mc_split_ratio,
				_beta_div,
			)

			magnetizations = mag_cat.reshape(self.n_instances, self.N0)
			geometry_stats = mc_stats if self.nmc_setup.track_stats else {}

			return magnetizations, J_tilde_flat, geometry_stats

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

			#the Hi heuristic has no factor graph and no correlations
			J_tilde_flat = None

		else:
			#THE SURROGATE'S TEMPERATURE. params.beta_start is this replica's rung under
			#bench_apt.py (_rung_params sets beta_start = beta_f = the rung), and it is a
			#TRACED leaf, so each of the vmapped NMC replicas carries its own. The factor
			#graph was built at self.nmc_setup.lbp_beta, so the table is rescaled by the
			#ratio - 1.0 when the switch is off, leaving the old path bit-for-bit unchanged.
			if self.nmc_setup.lbp_beta_from_rung:
				_lbp_beta = params.beta_start
				_pot_scale = _lbp_beta/self.nmc_setup.lbp_beta
			else:
				_lbp_beta = params.lbp_beta
				_pot_scale = jnp.array(1.0, dtype = jnp.float32)

			def run_surrogate_LBP(s: jax.Array):
				beliefs, bp_arrays, lbp_stats = lbp.surrogate_LBP(self.BP,
																							self.FG.variable_groups[0],
																							s,
																							self.epsilon,
																							_lbp_beta,
																							_pot_scale,
																							params.lbp_num_iters,
																							self.nmc_setup.problem_mode,
																							params.lbp_tolerance_m*_lbp_beta,
																							params.lbp_tolerance_d,
																							params.lbp_lambdas)

				#magnetizations from the surrogate hamiltonian
				mag_raw = (
					(beliefs[self.FG.variable_groups[0]][:, 1] - beliefs[self.FG.variable_groups[0]][:, 0])
				)/_lbp_beta/2

				#the final surrogate part: a lambda*epsilon*s field pinning every spin to the
				#state the walker is currently in. It is an artifact of the convexification,
				#not part of the problem, so "depinned" removes it. Kept as a named quantity
				#so the correlations below can be put on the same footing (they read cavity
				#fields that carry the identical term).
				if self.nmc_setup.problem_mode in ["pubo", "pising"]:
					pin = lbp_stats['final_lambda']*(2*s - 1)*self.epsilon
				else:
					pin = lbp_stats['final_lambda']*(s - 0.5)*self.epsilon

				mag_depinned = jnp.abs(mag_raw - pin)
				mag_pinned   = jnp.abs(mag_raw)

				mag = mag_depinned if self.nmc_setup.lbp_pinning == "depinned" else mag_pinned

				#k-local correlations and effective couplings (NMC paper Eq. 13/C3 and 15).
				#Computed and reported only - backbone selection still thresholds `mag`, so
				#existing behaviour is unchanged. J_tilde is what a correlation-based cluster
				#growth would threshold.
				#Gated EXPLICITLY rather than left to dead-code elimination. With
				#backbone_source "lbp_corr" these feed cluster growth and must be computed
				#whatever track_stats says; otherwise lbp_stats is their only route to an
				#output, so jit would eliminate them anyway - better to state the condition
				#than to have it hold by accident, invisibly.
				if self.corr_rows is not None and (self.nmc_setup.track_stats or
																									self.nmc_setup.backbone_source == "lbp_corr"):
					corr_dep, Jt_dep, corr_pin, Jt_pin = lbp.factor_correlations(
						bp_arrays, self.corr_vsfe0, self.corr_rows, self.corr_J, self.corr_idx,
						_lbp_beta, s, self.epsilon, lbp_stats['final_lambda']
					)

					if self.nmc_setup.lbp_pinning == "depinned":
						corr, J_tilde = corr_dep, Jt_dep
					else:
						corr, J_tilde = corr_pin, Jt_pin

					#label by ARITY, not by list position - the groups are one per interaction
					#order, so an enumerate() index would make "corr_k0" mean k = 2, and would
					#mean a different arity on an instance carrying a different set of orders
					for idx_g, c_d, jt_d, c_p, jt_p in zip(self.corr_idx,
																								 corr_dep, Jt_dep, corr_pin, Jt_pin):
						k = idx_g.shape[1]

						for tag, c, jt in (("depinned", c_d, jt_d), ("pinned", c_p, jt_p)):
							lbp_stats[f"corr_k{k}_{tag}_absmean"]   = jnp.mean(jnp.abs(c))
							lbp_stats[f"corr_k{k}_{tag}_absmax"]    = jnp.max(jnp.abs(c))
							lbp_stats[f"Jtilde_k{k}_{tag}_absmean"] = jnp.mean(jnp.abs(jt))
							lbp_stats[f"Jtilde_k{k}_{tag}_absmax"]  = jnp.max(jnp.abs(jt))

					lbp_stats["mag_depinned_absmean"] = jnp.mean(mag_depinned)
					lbp_stats["mag_pinned_absmean"]   = jnp.mean(mag_pinned)
				else:
					corr, J_tilde = None, None

				return mag, corr, J_tilde, lbp_stats


			magnetizations, correlations, J_tilde, lbp_stats = run_surrogate_LBP(state)
			magnetizations = magnetizations.reshape(self.n_instances, self.N0)

			#J_tilde is one array per interaction order; flatten to the concatenate(Jt)
			#ordering that factor_clustering indexes with. Stays in CAT space, which is
			#where the factor graph and J_tilde both live - Jh_stack pads J.indices with 0,
			#so a padded row there is a phantom k-fold self-factor on spin 0.
			J_tilde_flat = jnp.concatenate(J_tilde) if J_tilde is not None else None

			if self.nmc_setup.track_stats:
				geometry_stats = lbp_stats
			else:
				geometry_stats = {}

		return magnetizations, J_tilde_flat, geometry_stats


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

			#Hot end of the backbone's within-step ramp. The shipped convention is the
			#ABSOLUTE params.nmc_neq_beta, so as the outer anneal cools the boost ratio
			#drifts (38x at env step 0 to 60x at step 49 on chimera128). backbone_temp_x
			#instead pins the hot end to beta/temp_x, holding the ratio fixed - the
			#Nonlocal-Monte-Carlo convention (cnmc.py:371, temp_x = 20 in every shipped
			#example); the NMC paper quotes 2-10x (Sec. VII).
			if self.nmc_setup.backbone_temp_x is None:
				neq_beta_hot = params.nmc_neq_beta
			else:
				neq_beta_hot = state.beta/self.nmc_setup.backbone_temp_x

			if self.nmc_setup.neq_beta_anneal:
				nmc_neq_beta = jnp.exp(
					jnp.log(neq_beta_hot) + 
					jnp.log((state.beta + beta_range_neq)/neq_beta_hot)*i_cycle/params.nmc_neq_cycles
				)
			else:
				#constant boost, every cycle - the reference's convention
				nmc_neq_beta = neq_beta_hot

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
			
			#--- UNLEARNING PHASE ------------------------------------------------------------
			#Homogeneous MCMC over the WHOLE system at this rung's beta, nothing frozen. Runs
			#per cycle rather than once per env step, which is the placement the reference uses
			#and the paper describes. The state flows through as the phase-2 ENDPOINT - no
			#argmin is taken here, because the intermediate phases run at a boosted temperature
			#and their running minimum is an artefact of that, not a sample at `beta`.
			if params.nmc_nsw_unlearn > 0:
				#RESET THE RUNNING MINIMUM before the phase starts. mcmc_sampling seeds its own
				#running minimum from the seminkey it is handed (nmc.py: min_s_old =
				#seminkey.min_s), and the seminkey arriving here carries min_s/min_e from the
				#INHOMOGENEOUS phases, which ran at beta/backbone_temp_x. Passed through, the
				#unlearning phase's "best seen" would be min(hot-phase best, its own best) and
				#the hot phase - having thrashed at a 10x temperature - usually wins. That is a
				#cherry-picked moment during the scrambling, not a settled state at `beta`, and
				#carrying it forward is exactly the ratchet that drops Gibbs->NMC swap
				#acceptance from 0.587 to 0.253 (measured, 3/3 instances).
				#
				#Re-seeding to (s, s, e, e) makes the minimum mean "best during unlearning",
				#which is the only phase here that runs at the rung's own temperature.
				def unlearn(Jh, sekey, colors):
					sekey = Seminkey(sekey.s, sekey.s, sekey.e, sekey.e, sekey.key)
					#SCALAR beta, not the [start, end] pair. With anneal = True mcmc_sampling
					#reads beta[0]/beta[1] and interpolates; with anneal = False it does
					#jnp.full((nsw,), beta), which needs a scalar and raises on a 2-vector.
					args = (Jh, sekey, beta)
					kw = dict(anneal = False, totrack = self.nmc_setup.track_stats)
					if not self.nmc_setup.coloring:
						return nmc.mcmc_sampling(*args, params.nmc_nsw_unlearn,
																		 self.nmc_setup.problem_mode, **kw)
					return nmc.mcmc_sampling_colored(*args, colors, params.nmc_nsw_unlearn,
																					 self.nmc_setup.problem_mode, **kw)

				seminkey, stats_unlearn = jax.vmap(unlearn)(
					self.Jh_stack, seminkey, self.colors_stack)

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

				#present only when the unlearning phase is enabled, so the stats tree stays a
				#faithful record of which phases actually ran
				if params.nmc_nsw_unlearn > 0:
					mc_stats["unlearn"] = stats_unlearn

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
			if params.use_nmc and self.nmc_setup.nmc_carry == "unlearn_argmin":
				#THE FINAL UNLEARNING PHASE MEASURES ITS OWN MINIMUM. Entering from the
				#ENDPOINT, with the running minimum re-seeded to it, so min_e coming out of
				#this phase is the best state seen AT `beta` and nowhere else. The other two
				#modes enter from seminkey.min_s, the argmin over the NMC cycles - but those
				#cycles ran at beta/backbone_temp_x, so their minimum is a cherry-picked moment
				#during the scrambling rather than a configuration this replica could hold at
				#its own temperature. Carrying that forward is the ratchet that drops
				#Gibbs->NMC swap acceptance from 0.587 to 0.253 (measured, 3/3 instances).
				mcmc_s, mcmc_min_s, mcmc_e, mcmc_min_e = (seminkey.s, seminkey.s, seminkey.e, seminkey.e)
			elif params.use_nmc:
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

			if params.use_nmc and self.nmc_setup.nmc_carry == "endpoint":
				#the equilibrium phase's endpoint, as npt.py:647 hands its swap. No accept/reject:
				#that phase is ordinary MCMC at the base beta, so its endpoint is already drawn from
				#the right distribution and gating it against the step's start would bias it.
				final_s, final_e = seminkey.s, e
				key = seminkey.key

				if self.nmc_setup.track_stats:
					#same keys, so a run stays comparable across the two conventions
					total_stats["metropolis_ar"] = jnp.array(1.0)
					total_stats["metropolis_jump"] = (old_s != final_s).sum()/old_s.size
					total_stats["metropolis_de"] = final_e - old_e

			elif params.use_nmc:
				#"argmin"          - min over the whole step, Metropolis'd against its start
				#"unlearn_argmin"  - same test, but min_s/min_e here are the FINAL UNLEARNING
				#                    PHASE's own minimum (re-seeded above), so the state being
				#                    offered was drawn at `beta`. Accept always when lower,
				#                    otherwise exp(-beta*dE).
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

		if self.nmc_setup.geometry_in_step:
			#fold_in, NOT another split of `key`: splitting would consume stream positions and
			#shift `instance_keys` above, changing every existing run's randomness. fold_in
			#derives an independent key from the same parent and leaves `instance_keys`
			#bit-for-bit as it was, so the Hi and LBP paths are unaffected by this argument
			#existing. Only "mc_corr" reads it.
			magnetizations, jt, geometry_stats = self.get_geometry(
				state.s, params, jax.random.fold_in(key, _GEOM_KEY_FOLD))
		else:
			#the caller computes it, after its own post-step bookkeeping. Zeros rather than a
			#missing field so the observation tree keeps its shape and two vmapped groups still
			#merge; nothing reads these (bench_apt.py overwrites mag and jt after the swap).
			magnetizations = jnp.zeros((self.n_instances, self.N0), dtype = jnp.float32)
			jt = None
			geometry_stats = {}

		return (
			self.get_obs(
				state, 
				magnetizations,
				params,
				jt
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

		#fold_in rather than a further split, for the reason given in step_env: the reset's
		#own instance_keys must keep their stream positions or every cached warm-up changes.
		magnetization, jt, geometry_stats = self.get_geometry(
			state.s, params, jax.random.fold_in(key, _GEOM_KEY_FOLD))

		return (
			self.get_obs(
				state, 
				magnetization,
				params,
				jt
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
							params: NMCParams,
							jt: Union[jax.Array, None] = None) -> EnvObs:
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
			norm_beta = state.beta/params.beta_f,

			jt = jt
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