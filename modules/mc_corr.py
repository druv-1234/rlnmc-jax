"""
Direct Monte Carlo estimation of the NMC surrogate correlations.

A second, independent source for the same quantity `modules/lbp.py` computes in closed
form: the k-local factor correlation of the NMC paper's Eq. (13)/(C3),

	<prod_{i in da} r_i>

one scalar per factor, in `concatenate(Jt)` order, CAT space - exactly what
`lbp.factor_correlations` returns as `corr_pin`. Nothing here computes `J_tilde`; the
arctanh of Eq. (14)/(15) is deliberately out of scope for this pass, so the two paths are
compared on the raw correlation where a sampling error bar means something.

WHY THIS IS CHEAP TO BUILD. The surrogate Hamiltonian is exactly a modified external
field. With the repo's sign convention (`get_energy` computes -(sum J prod s + h.s)),

	H~(r) = H(r) - lam sum_i eps_i s*_i r_i
	      = -( sum_a J_a prod_{i in da} r_i + sum_i (h_i + lam eps_i s*_i) r_i )

so `Jh._replace(h = h + lam*eps*(2s*-1))` turns every existing spin kernel into a sampler
of the surrogate. `modules/nmc.py` is not touched, and neither is anything else in the
repo - this module is standalone.

PINNED ONLY. `lbp_pinning: "depinned"` subtracts lam*eps*s from each of the k CAVITY
fields (`lbp.py:346`), a message-space operation with no MC analogue: MC observes
configurations, not cavity fields, and the only exact removal is an importance
reweighting whose variance is precisely what the pinning exists to avoid. Everything here
is the PINNED correlation, and must be compared against `corr_pin`.

BLOCKS, NOT RUNNING MEANS. Sweeps are accumulated in consecutive, disjoint blocks. A
running mean flattens like 1/t whether or not the chain has mixed, so it cannot diagnose
settling; block means stop drifting only when the chain does. The spread across blocks is
also a standard error with the autocorrelation already absorbed - no tau_int anywhere -
and the cross-block product of two blocks that share no samples is what makes an unbiased
self-overlap possible from a single chain (see `overlaps`).
"""

import sys
from functools import partial
from typing import List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax

sys.path.append('.')

from modules import nmc
from modules.nmc_types import Jhdata


def surrogate_Jh(Jh: Jhdata, s_star: jax.Array, epsilon: jax.Array, lam) -> Jhdata:
	"""
	The localised surrogate of Eq. (2), as a field shift on the ORIGINAL couplings.

	:param Jh: CAT-space Hamiltonian (`env.Jh_cat`).
	:param s_star: reference configuration, bool `(N,)`, True = +1.
	:param epsilon: eps_i = |h_i| + sum_a |J_a| over factors touching i, `(N,)`
		(`nmcgym.py:296-299`).
	:param lam: scalar locality parameter.
	:return: Jhdata with h -> h + lam*eps*(2 s* - 1); J and Jat untouched.
	"""
	pm1 = 2.0*s_star.astype(jnp.float32) - 1.0
	return Jh._replace(h = Jh.h + lam*epsilon*pm1)


def _sweep(Jh: Jhdata, s: jax.Array, colors: jax.Array, beta, key):
	"""
	One chromatic Gibbs sweep. Spins of a colour share no factor (DSATUR runs on the
	clique expansion), so a whole colour class resamples in parallel and the sweep is
	still a valid Gibbs scan.

	Padding convention is the repo's: colour entries equal to N are out of range, read
	with `mode='fill'` and scattered back with `mode='drop'`.
	"""
	N = s.size
	n_colors = colors.shape[0]

	key, k_perm, k_r = jax.random.split(key, 3)
	perm = jax.random.permutation(k_perm, n_colors)
	r = jax.random.uniform(k_r, (N,))

	def step(s_carry, color):
		r_c = r.at[color].get(mode = 'fill', fill_value = 0.0)
		s_i, _, _ = jax.vmap(nmc.propose,
												 in_axes = (None, None, 0, 0, None, None))(
			Jh, s_carry, color, r_c, beta, "pising")
		return s_carry.at[color].set(s_i, mode = 'drop'), None

	s, _ = lax.scan(step, s, colors[perm])
	return s, key


def _accumulate_block(Jh, s, colors, beta, key, idx_list, block_len: int,
											with_corr: bool = True):
	"""
	`block_len` sweeps, accumulating the per-spin magnetisation and - when
	`with_corr` - the per-factor product. Returns the BLOCK MEANS, not the running
	totals.

	The correlation accumulator is a plain multiply-accumulate over the members of every
	factor that exists - `idx_list` is `[j.indices for j in Jh_cat.J]`, one `(n_terms_g, k)`
	array per interaction order, which is the same array `lbp.factor_correlations` indexes
	with and the same one `clustering.factor_seed_mask` consumes.

	`with_corr = False` exists for the lambda ladder's MONITOR phase, which only ever
	reads magnetisations: `p`, `q1` and `q0` are all built from `<r_i>`, so accumulating
	a product over every factor at every rung would be paid for and then discarded. On
	chimera128 that is 352 factors x block_len sweeps per rung thrown away. The flag is
	static, so the multiply-accumulate is not emitted at all rather than computed and
	masked.
	"""
	def body(carry, _):
		s_c, key_c, msum, csums = carry
		s_c, key_c = _sweep(Jh, s_c, colors, beta, key_c)
		pm1 = 2.0*s_c.astype(jnp.float32) - 1.0
		msum = msum + pm1
		if with_corr:
			csums = [c + jnp.prod(pm1[idx], axis = 1) for c, idx in zip(csums, idx_list)]
		return (s_c, key_c, msum, csums), None

	msum0 = jnp.zeros(s.size, dtype = jnp.float32)
	csum0 = ([jnp.zeros(idx.shape[0], dtype = jnp.float32) for idx in idx_list]
					 if with_corr else [])

	(s, key, msum, csums), _ = lax.scan(body, (s, key, msum0, csum0), None,
																			length = block_len)

	inv = 1.0/block_len
	return s, key, msum*inv, [c*inv for c in csums]


@partial(jax.jit, static_argnames = ['block_len', 'n_blocks', 'with_corr'])
def mc_blocks(Jh_surr: Jhdata,
							s0: jax.Array,
							colors: jax.Array,
							beta,
							key,
							idx_list: List[jax.Array],
							block_len: int,
							n_blocks: int,
							with_corr: bool = True):
	"""
	Run `n_blocks` consecutive blocks of `block_len` sweeps from `s0`, returning the
	per-block means and the final configuration.

	No burn-in is discarded here on purpose. Settling is read off the block sequence
	afterwards, which is both a diagnostic and the thing that removes `settle_sweeps` as a
	number anyone has to guess.

	:return: (s_final, mag_blocks (n_blocks, N), corr_blocks list of (n_blocks, n_terms_g))
	"""
	def body(carry, _):
		s, key_c = carry
		s, key_c, mag, corr = _accumulate_block(Jh_surr, s, colors, beta, key_c,
																						idx_list, block_len, with_corr)
		return (s, key_c), (mag, corr)

	(s, _), (mag_blocks, corr_blocks) = lax.scan(body, (s0, key), None, length = n_blocks)
	return s, mag_blocks, corr_blocks


def mc_blocks_clones(Jh_surr, s0_clones, colors, beta, keys, idx_list,
										 block_len: int, n_blocks: int, with_corr: bool = True):
	"""
	`mc_blocks` over R independent clones - measure (B2) of App. B, no gamma coupling.
	`s0_clones` is `(R, N)`, `keys` is `(R,)` of PRNG keys.
	"""
	return jax.vmap(mc_blocks,
									in_axes = (None, 0, None, None, 0, None, None, None, None),
									out_axes = (0, 0, 0))(
		Jh_surr, s0_clones, colors, beta, keys, idx_list, block_len, n_blocks,
		with_corr)


# ---------------------------------------------------------------------------
# Post-processing. Host-side numpy on purpose: these read the whole block
# sequence, which is a diagnostic, not something the sampler needs.
# ---------------------------------------------------------------------------

def _pair_mean(x, axis):
	"""
	Mean over ORDERED pairs (a, b) with a != b, along `axis`.

	E[x_a x_b] with a != b is the identity behind Eq. (B4): two estimates that share no
	samples have no variance cross-term, so their product is unbiased for <r>^2, while
	squaring ONE estimate is biased upward by exactly its own variance.
	"""
	import numpy as np
	n = x.shape[axis]
	if n < 2:
		return None
	s = x.sum(axis = axis)
	sq = (x*x).sum(axis = axis)
	return (s*s - sq)/(n*(n - 1))


def overlaps(mag_blocks, s_star, b0: int = 0):
	"""
	p (Eq. 8), q1 (Eq. 9) and q0 (Eq. B4) from block means, discarding the first `b0`
	blocks.

	q0 is computed TWICE, from two different sources of independence:
	  - `q0_clones`: over clone pairs. Clones are separate chains, so exactly independent.
	    Needs R >= 2.
	  - `q0_blocks`: over block pairs within a chain. Blocks are disjoint but CONSECUTIVE,
	    so they are independent only up to a seam correlation of order tau_int/block_len.
	    Works at R = 1.
	`q0_clones - q0_blocks` is that seam bias, in the same units, from the same run.

	They are NOT interchangeable, and the difference is physical, not statistical. A single
	chain lives in ONE basin, so its own disjoint segments keep agreeing with each other
	even after the surrogate has fragmented into many basins - measured on chimera128,
	`q0_blocks` tracks `q1` to three decimals all the way down and detects nothing, while
	`q0_clones` falls to -0.06 against `q1 = 0.58` at the same lambda. That is why
	`clone_gap` refuses to run at R = 1 rather than silently substituting `q0_blocks`, and
	why the R = 1 stopping rule falls back to the directed p-vs-q1 test instead.

	`q1 - q0` is an unbiased estimate of the Monte Carlo noise floor (1/N) sum_i Var(rbar_i),
	which is the scale the p-vs-q gap has to be judged against.

	:param mag_blocks: `(R, n_blocks, N)` per-clone per-block mean magnetisation.
	:param s_star: bool `(N,)` reference configuration.
	:return: dict of scalars plus `p_per_clone` and `se_p`.
	"""
	import numpy as np

	m = np.asarray(mag_blocks)[:, b0:, :]                  #(R, B, N)
	R, B, N = m.shape
	ref = 2.0*np.asarray(s_star).astype(np.float64) - 1.0

	m_bar = m.mean(axis = 1)                               #(R, N)  per-clone estimate

	p_per_clone = (m_bar*ref).mean(axis = 1)               #(R,)
	p = float(p_per_clone.mean())

	q1 = float((m_bar*m_bar).mean(axis = 1).mean())        #biased up by Var(m_bar)

	q0_clones = _pair_mean(m_bar, axis = 0)
	q0_clones = None if q0_clones is None else float(q0_clones.mean())

	q0_blocks = _pair_mean(m, axis = 1)                    #(R, N)
	q0_blocks = None if q0_blocks is None else float(q0_blocks.mean())

	#SE of p from the block-to-block spread: tau_int is absorbed, never estimated.
	p_block = (m*ref[None, None, :]).mean(axis = 2)        #(R, B)
	se_p = float(p_block.std(axis = 1, ddof = 1).mean()/np.sqrt(B*R)) if B > 1 else float("nan")

	return {"p": p, "q1": q1, "q0_clones": q0_clones, "q0_blocks": q0_blocks,
					"se_p": se_p, "p_per_clone": p_per_clone.tolist(),
					"R": R, "n_blocks": B, "b0": b0}


def clone_gap(mag_blocks, b0: int = 0):
	"""
	The self-overlap minus the CLONE overlap, q1 - q0, with a jackknife-over-blocks error
	bar. Requires R >= 2 - and that requirement is structural, not statistical.

	q1 - q0 asks whether independently evolved copies of the surrogate end up describing the
	same state. While the surrogate is confined to one pure state they do, and the gap is
	just the sampling noise floor. When the pin gets too weak to hold them, the clones fall
	into different basins and the gap opens up - q0 can even go NEGATIVE, which is two
	clones sitting in anti-aligned states.

	This is why blocks cannot substitute for clones here. Blocks are segments of ONE chain,
	which is inside ONE basin by construction, so a block-based q0 tracks q1 all the way
	down and sees nothing. The bias argument for clones is real but secondary; this is the
	part R = 1 cannot do at all.

	Because the statistic is non-linear in the block means, the error bar is a delete-one
	jackknife over blocks rather than a plain block standard error.

	:return: (gap, se_gap), or (None, None) when R < 2.
	"""
	import numpy as np

	m = np.asarray(mag_blocks)[:, b0:, :]
	R, B, _ = m.shape
	if R < 2 or B < 2:
		return None, None

	def stat(mm):
		mb = mm.mean(axis = 1)
		q1 = float((mb*mb).mean())
		q0 = float(_pair_mean(mb, axis = 0).mean())
		return q1 - q0

	full = stat(m)
	ps = np.array([stat(np.delete(m, b, axis = 1)) for b in range(B)])
	se = float(np.sqrt((B - 1)/B*((ps - ps.mean())**2).sum()))
	return full, se


def clone_split(mag_blocks, b0: int = 0):
	"""
	Have the clones fallen into DIFFERENT states? Dimensionless, no tuned constant.

	`q1 - q0` is not meaningful on its own - it is strictly positive even when every clone
	sits in the same state, because it is exactly the Monte Carlo noise floor
	(1/N) sum_i Var(rbar_i). Testing whether it is "significantly nonzero" therefore fires
	everywhere given enough samples, which is the mistake this replaces.

	The fix is to compare it against that noise floor MEASURED INDEPENDENTLY, from the
	block-to-block spread of each clone's own mean:

	    expected  = (1/N) sum_i mean_a Var_blocks(rbar^a_i)      <- clones in one state
	    observed  = q1 - q0
	    ratio     = observed / expected

	Clones confined to a single pure state give ratio ~ 1 at any sweep count, autocorrelation
	included, because both sides absorb it identically. Clones that have scattered into
	different basins give a ratio that grows without bound. Requires R >= 2.

	:return: (observed, expected, ratio), or (None, None, None) when R < 2.
	"""
	import numpy as np

	m = np.asarray(mag_blocks)[:, b0:, :]
	R, B, _ = m.shape
	if R < 2 or B < 2:
		return None, None, None

	mbar = m.mean(axis = 1)
	q1 = float((mbar*mbar).mean())
	q0 = float(_pair_mean(mbar, axis = 0).mean())
	observed = q1 - q0

	#variance of each clone's OWN mean, from its blocks: sigma_b^2 / B
	expected = float((m.var(axis = 1, ddof = 1)/B).mean())

	return observed, expected, (observed/expected if expected > 0 else float("inf"))


def settle_block(mag_blocks, s_star, n_sigma: float = 2.0, min_tail: int = 4):
	"""
	First block index after which the block means of p no longer trend.

	Splits the remaining blocks in half and asks whether the two halves agree to within
	`n_sigma` standard errors. Unlike a running mean - which flattens like 1/t regardless
	of mixing - this can genuinely fail, which is the whole point of using it as a
	settling test rather than a plot.

	:return: (b0, ok). `ok` is False if no split passed, i.e. the chain never settled
		within the blocks that were run.
	"""
	import numpy as np

	m = np.asarray(mag_blocks)
	R, B, N = m.shape
	ref = 2.0*np.asarray(s_star).astype(np.float64) - 1.0
	p_block = (m*ref[None, None, :]).mean(axis = 2)        #(R, B)

	for b0 in range(0, B - min_tail + 1):
		tail = p_block[:, b0:]
		half = tail.shape[1]//2
		if half < 2:
			break
		a, b = tail[:, :half], tail[:, half:]
		diff = a.mean() - b.mean()
		se = np.sqrt(a.std(ddof = 1)**2/a.size + b.std(ddof = 1)**2/b.size)
		#a perfectly frozen chain has zero spread: that is settled, not un-settleable.
		#Without this branch the deeply pinned rungs report "never settled", which is the
		#opposite of the truth.
		if se == 0.0:
			if abs(diff) == 0.0:
				return b0, True
		elif abs(diff) <= n_sigma*se:
			return b0, True
	return B - min_tail, False


def corr_from_blocks(corr_blocks, b0: int = 0):
	"""
	Flatten the per-group per-block correlation means into one `(n_factors,)` vector in
	`concatenate(Jt)` order - the same ordering `nmcgym.py:418` produces and
	`clustering.py` consumes - plus its block-spread standard error.
	"""
	import numpy as np

	c = [np.asarray(x)[:, b0:, :] for x in corr_blocks]    #list of (R, B, n_terms_g)
	flat = np.concatenate([x.reshape(x.shape[0], x.shape[1], -1) for x in c], axis = 2)
	R, B, _ = flat.shape

	mean = flat.mean(axis = (0, 1))
	se = flat.reshape(R*B, -1).std(axis = 0, ddof = 1)/np.sqrt(R*B) if R*B > 1 else None
	return mean, se


# ---------------------------------------------------------------------------
# corr -> J_tilde. Device-side, jit-safe: this is the one piece of
# post-processing that has to run where `get_geometry` runs.
# ---------------------------------------------------------------------------

def _cavity_prod_from_corr(corr: jax.Array, tJ: jax.Array) -> jax.Array:
	"""
	Invert the tanh addition formula for the cavity-field product.

	`lbp.factor_correlations` builds the correlation as (lbp.py:372)

		corr = (tJ + prod)/(1 + tJ*prod),   tJ = tanh(beta*J_a)

	which is tanh(atanh(tJ) + atanh(prod)), so it inverts exactly:

		prod = (corr - tJ)/(1 - tJ*corr)

	NOT evaluated in that form. When `beta*J_a` is large both `tJ` and `corr`
	sit one ulp below 1, and both the numerator and the denominator are then
	differences of nearly-equal float32 numbers - the denominator in particular
	keeps about one significant bit. Writing `a = 1 - tJ`, `b = 1 - corr`,

		corr - tJ      =  a - b
		1 - tJ*corr    =  a + b - a*b        =  a + b*(1 - a)

	has no cancellation at all: the small quantities are the inputs, not the
	results. The mirror substitution `a = 1 + tJ`, `b = 1 + corr` does the same
	job at the -1 end, and `tJ`'s sign picks the branch because `corr` saturates
	towards whichever end `tJ` does.

	Both denominators are STRICTLY positive given `clip_off_unit` inputs: in the
	`+1` branch `a >= eps` and `b*(1 - a) >= 0`, so `a + b*(1 - a) >= eps`.
	"""
	from modules import lbp

	c = lbp.clip_off_unit(corr)
	t = lbp.clip_off_unit(tJ)

	#+1 end
	a_p, b_p = 1.0 - t, 1.0 - c
	num_p, den_p = a_p - b_p, a_p + b_p*(1.0 - a_p)

	#-1 end
	a_m, b_m = 1.0 + t, 1.0 + c
	num_m, den_m = b_m - a_m, a_m + b_m*(1.0 - a_m)

	num = jnp.where(t >= 0, num_p, num_m)
	den = jnp.where(t >= 0, den_p, den_m)

	return lbp.clip_off_unit(num/den)


def corr_to_Jtilde(corr: jax.Array, J_flat: jax.Array, beta,
									 beta_div = None) -> jax.Array:
	"""
	The effective coupling of Eq. (14)/(15), from an MC correlation estimate.

	Reproduces `lbp.py:395` - `J_tilde = J_a + arctanh(prod tanh(beta*h))/beta` -
	by recovering `prod` from `corr` first, rather than evaluating the literal
	`arctanh(corr)/beta`. The two are the same identity, and the difference is
	not cosmetic:

	  `arctanh(corr)/beta` is capped at `arctanh(1 - eps32)/beta = 8.3178/beta`.
	  At the coldest chimera128 NMC rung (beta = 48.49) that cap is 0.17, while
	  `|J_a|` alone reaches 1.0 under `normalize_weights` - so the literal form
	  reports 0.17 for a factor whose J_tilde is 1.006, a factor of six low, and
	  it does so WITHOUT saturating any intermediate. Measured, J_a = 1, beta =
	  48.49, prod = 0.3: true 1.00638, literal 0.17154, this function 1.00000.
	  CLAUDE.md records the same cap voiding an earlier Vizier study.

	What this form does when the correlation IS saturated is degrade to
	`J_tilde = J_a`, which is the honest answer: once `tanh(beta*J_a)` is 1 to
	float32, `corr` carries no information about the cavity product, and no
	rearrangement puts it back. The bare coupling is what remains known.

	SEPARATE, AND NOT A FLOAT32 PROBLEM: an MC `corr` is a mean of T products of
	+-1, so `|corr| <= 1 - 1/T` and the reachable `arctanh` is `~ln(2T)/2`
	regardless of precision. That statistical ceiling (3.5-4.7 at realistic T)
	binds long before the float32 one (8.3178), so the sweep budget - not the
	dtype - is what sets usable J_tilde range. `jtilde_ceiling` below states it.

	DECOUPLED BETAS (`beta_div`). Monte Carlo cannot read correlations at an APT
	rung's own beta: at beta ~ 48 a single spin flip costing dE ~ 1 carries weight
	e^-48 ~ 1e-21, the chain never moves, every sample is identical, and every
	factor reports |corr| = 1. Measured on chimera128: 97-100% of factors saturated
	at EVERY lambda on the ladder, with J_tilde pinned flat at |J_a|. LBP is immune
	because it never samples - it evaluates the cavity product in log space, where
	the differences between numbers near 1 survive, and
	`arctanh(prod tanh(beta*h))/beta` tends to a finite `min_i |h_i|` limit as beta
	grows. Sampling cannot reach that limit: it lives in the e^(-2 beta h) tail,
	which needs ~e^(2 beta h) samples to observe.

	So `beta` here is the temperature the chain actually ran at (order 1), and
	`beta_div` lets the surrogate term be expressed in a rung's units instead.
	Note what that does and does not buy: the contribution becomes
	`arctanh(prod_at_beta_sample)/beta_rung`, which is O(1/beta_rung) and therefore
	SHRINKS with the rung rather than converging to the cavity-field limit LBP
	reaches. It rescales; it does not recover the information.

	:param corr: `(..., n_factors)` k-local correlations, `concatenate(Jt)` order.
	:param J_flat: `(n_factors,)` bare couplings in the SAME order, i.e.
		`jnp.concatenate([j.data for j in Jh_cat.J])`.
	:param beta: scalar inverse temperature - the same one the sampler ran at.
	:return: `(..., n_factors)` J_tilde, float32, ready for `EnvObs.jt`.
	"""
	from modules import lbp

	#`beta` is the SAMPLING temperature and must be the one used to invert, because
	#it is the measure that produced `corr`: lbp.py:372 builds corr from
	#tanh(beta_sample*J_a) and the cavity product under that same measure.
	tJ   = jnp.tanh(beta*J_flat)
	prod = _cavity_prod_from_corr(corr, tJ)

	#`beta_div` rescales the surrogate contribution into another temperature's
	#units - see the note on DECOUPLED BETAS below. Defaults to beta_sample, which
	#is the paper's Eq. (15) exactly.
	bd = beta if beta_div is None else beta_div
	return J_flat + lbp.atanh_saturated(prod)/bd


def jtilde_ceiling(J_flat: jax.Array, beta, n_samples: int) -> jax.Array:
	"""
	The largest `|J_tilde|` an MC run of `n_samples` post-settle sweeps can
	report, per factor:

		|J_tilde| <= |J_a| + ln(2T)/(2 beta)

	because `|corr| <= 1 - 1/T` for a mean of T products of +-1, and
	`arctanh(1 - 1/T) = ln(2T)/2 + O(1/T)`.

	This is the quantity to compare a candidate `theta_seed` against BEFORE
	running a study: a threshold above this ceiling selects nothing INFORMATIVE,
	and `clustering.factor_clustering` reports that as an all-False backbone with
	no error. On chimera128 at beta = 48.49 with T = 1000 the surrogate's whole
	contribution is 0.078, so `theta` is then thresholding `|J_a|` and little else.

	ONE ESCAPE, and it is intended. A factor whose T samples ALL agreed returns
	`corr = +-1` exactly - not `1 - 1/T` - so `clip_off_unit` takes over and
	`J_tilde` goes to `|J_a| + 8.3178/beta`, above this bound. That is not a bug
	to clamp: such a factor is rigid beyond what the budget can measure, which is
	the strongest evidence of rigidity available, and backbone selection should
	and does take it. `surrogate_MC` reports `corr_saturated` so the fraction in
	that regime is visible rather than inferred.
	"""
	import numpy as _np
	return jnp.abs(J_flat) + float(_np.log(2.0*n_samples)/2.0)/beta


# ---------------------------------------------------------------------------
# Device-side overlap statistics.
#
# The numpy versions above read the whole block sequence and stay the reference
# implementation - `tests/test_mc_corr.py` checks these against them. These exist
# because the lambda ladder has to run inside `lax.while_loop`, inside the jitted
# `get_geometry`, inside `pmap`, where numpy cannot go.
# ---------------------------------------------------------------------------

class MCCfg(NamedTuple):
	"""
	Sweep budget and stop sensitivity. Every field is STATIC - they become
	`lax.scan` trip counts and a `lax.while_loop` iteration bound - so this is
	hashable and belongs in `nmc_setup`, not `NMCParams`.

	`monitor_*` and `corr_*` are deliberately separate budgets. The monitor phase
	only needs the precision to notice a genuine p/q/q0 departure; the correlation
	phase has to hit a target SEM on the correlations themselves and is what sets
	the J_tilde ceiling (see `jtilde_ceiling`). Sharing one budget between them
	ties two unrelated requirements together.

	`monitor_settle_blocks` is NOT optional padding, and it is the one knob here
	whose absence produces a WRONG ANSWER rather than a noisy one. Clones warm-start
	from the previous rung, so each arrives at a new lambda carrying its own
	relaxation history. Averaging over that transient makes the clones look like
	they disagree, which is precisely the signature `split_ratio` exists to detect -
	so the stop fires on a fixture that has nothing to detect. Measured on the tree
	negative control at 16x25 monitor blocks and no settle: q0 reached -0.0756 (two
	clones ANTI-aligned, on a graph with a single Gibbs state) and pq_t hit +11.1.
	With a settle phase the same fixture stays where it belongs. `mcmc_corr_promt.md`
	Sec. 4 specifies settle-then-monitor for this reason; dropping it was the bug.

	`monitor_block_len` matters MORE than the block count, and 50 is a floor rather
	than a preference. The block-to-block spread IS the error bar - tau_int is
	absorbed into it, never estimated - so blocks short enough to stay correlated
	with each other report an error bar smaller than the truth, and `pq_t` inflates
	by exactly that factor. Measured, warm-started, tree vs chimera128, max `pq_t`
	down the whole ladder:

	    settle  mon  block_len  sweeps/rung    tree     chimera128   separation
	       0     16      25         400       +11.12      +14.22        none
	       8     16      25         600       +17.00      +36.94        poor
	       8     16      50        1200        +1.94      +68.57         35x
	       8     24      50        1600        +1.19      +85.58         72x

	At block_len 25 the tree - which has ONE Gibbs state and therefore nothing to
	detect - is indistinguishable from chimera128, which has everything to detect.
	At 50 they are 35x apart and any threshold in between works. That is the whole
	reason the defaults below are what they are, and why `block_len` is calibrated
	rather than searched.

	`p_target`, `lambda_seed` and `max_increase_steps` drive the lambda-INCREASE
	phase, which is the one thing stopping the whole ladder being vacuous.

	LAMBDA IS THE CONTROL VARIABLE, AND BETA VERY NEARLY CANCELS. It is tempting to
	read the pin as `beta*lam*eps` and conclude a colder chain needs proportionally
	less lambda. It does not, because beta multiplies the couplings too:

	    exp(-beta*H~) = exp( beta*[ sum_a J_a prod r + sum_i (h_i + lam*eps_i*s*_i) r_i ] )

	so at each spin the contest is `lam*eps_i` against that spin's own coupling
	field - and `eps_i` IS `|h_i| + sum_a |J_a|`. The ratio is therefore `lam`
	alone; beta only sharpens the crossover. Measured, normalised chimera128, the
	lambda at which p first reaches 0.9:

	    beta       1.5    3.0    6.0   9.64   24.0   48.5
	    lam*       2.5    1.0    1.0    1.0    0.7    0.7
	    beta*lam*  3.75   3.0    6.0    9.6   16.8   33.9

	A 32x change in beta moves `lam*` by 3.6x, not 32x, and `beta*lam*` grows
	rather than staying fixed - so the `beta*lam*eps` reading is wrong. The weak
	residual trend is thermal smearing: a hotter chain needs extra lambda to beat
	entropy, a colder one sits at the energetic balance point.

	WHAT THIS MEANS FOR THE SHIPPED LAMBDA. `lbp_lambdas = [0.015, 0.001, 0.5]`
	does not pin at ANY temperature - at lam ~ 0.01-0.03 the measured p runs 0.016
	to 0.346 across every beta from 1.5 to 48.5. It is not a temperature mismatch
	to be compensated; lambda is simply two orders of magnitude too small. With it,
	p = q1 = q0 = 0.003 and the paper's criterion is satisfied VACUOUSLY - 0 = 0 = 0
	- so `stop_fired` never fires, the ladder runs to its floor, and the
	correlations come off a surrogate that was never pinned. A silent false
	negative, not a tolerance to widen.

	LBP survives the identical lambda because its stopping rule is message
	convergence, not overlap (and chimera's `lbp_tolerance_d = 1.0` disables its
	distance check outright), so the LBP ladder's lambda CANNOT be reused here.

	That does NOT mean the LBP arm runs unpinned. At the rung temperatures its
	configs actually use (`lbp_beta_from_rung: true`, beta 9.6-48.5) its anneal stops
	early at lam_op = 0.96, inside the pinned band - BP stops converging once the
	surrogate leaves a single pure state, so message convergence is a physically
	related proxy for the overlap test rather than an unrelated one. Only at
	`lbp_beta = 1.0` does it reach the floor, and there its J_tilde becomes
	state-INDEPENDENT (corr 1.0000 between two references 63 of 128 spins apart),
	i.e. a fixed property of the instance recomputed every step for nothing.

	`surrogate_MC` raises lambda until `p >= p_target` before descending, the direct
	analogue of `lbp.surrogate_LBP`'s `run_bp_lmbd_increase` (lbp.py:466) raising
	lambda until the MAP state is within `tolerance_d` of s*. `lambda_seed` is where
	that climb starts, and 1.0 is the measured `lam*` over most of the usable beta
	range rather than a guess. Seeding HIGH is the safe direction: the operating
	lambda is set by where the stop fires, not by where the descent began -
	measured at beta = 9.64, lam_start in {1, 4, 16} all back off to lam_op = 1.0
	with p = q1 = q0 ~ 0.98 - so an overshoot costs a few descent rungs while an
	undershoot costs increase steps AND risks `max_increase_steps` running out.
	"""
	replicas: int = 2
	monitor_settle_blocks: int = 8
	monitor_blocks: int = 16
	monitor_block_len: int = 50
	corr_blocks: int = 32
	corr_block_len: int = 50
	corr_settle_blocks: int = 8
	persist: int = 2
	p_target: float = 0.9
	lambda_seed: float = 1.0
	max_increase_steps: int = 12


class MCStats(NamedTuple):
	"""Overlap statistics at one lambda. All scalars, all float32."""
	p: jax.Array
	q1: jax.Array
	q0: jax.Array
	se_p: jax.Array
	noise_floor: jax.Array        #q1 - q0, the observed floor
	noise_expected: jax.Array     #the independently measured floor
	split_ratio: jax.Array        #observed/expected - scale free
	pq_t: jax.Array               #(q1 - p)/se_p - DIRECTED, see stop_fired


#Float32 resolution on quantities of order 1. p, q1 and q0 are all means of
#values in [-1, 1], so a difference between them below this is rounding, not
#signal. It is the floor on every DENOMINATOR below, and it is what keeps a
#perfectly frozen chain - se_p and the noise floor both exactly 0 - from
#reporting an infinite test statistic. `settle_block` makes the same allowance
#for the same reason: zero spread means settled, not un-settleable.
_F32_RES = 1e-6


def _pair_mean_dev(x, axis: int):
	"""
	`_pair_mean` on device. Mean over ORDERED pairs (a, b), a != b.

	Same identity - (S^2 - sum x^2)/(n(n-1)) - for the same reason: two estimates
	that share no samples have no variance cross-term, so their product is
	unbiased for <r>^2 where squaring ONE estimate is biased up by its variance.
	Requires n >= 2, which `surrogate_MC` enforces statically via `MCCfg.replicas`.
	"""
	n = x.shape[axis]
	s = x.sum(axis = axis)
	sq = (x*x).sum(axis = axis)
	return (s*s - sq)/(n*(n - 1))


def overlap_stats(mag_blocks: jax.Array, s_star: jax.Array) -> MCStats:
	"""
	p (Eq. 8), q1 (Eq. 9), q0 (Eq. B4) and the two error scales, from per-clone
	per-block mean magnetisations `(R, B, N)`.

	`q0` is the CLONE overlap only. There is no `q0_blocks` fallback here on
	purpose: a single chain sits in one basin, so its own disjoint segments keep
	agreeing after the surrogate has fragmented, and a block-based q0 therefore
	detects nothing (measured on chimera128 - see `overlaps`). R >= 2 is
	structural for this statistic, not a precision preference.

	`noise_expected` is the floor measured independently of q0: each clone's own
	block-to-block variance divided by the block count, which is the variance of
	that clone's mean. `split_ratio = (q1 - q0)/noise_expected` is then ~1 while
	the clones share one pure state at ANY sweep count, because autocorrelation
	enters both sides identically, and grows without bound once they scatter.
	"""
	R, B, _ = mag_blocks.shape
	ref = (2.0*s_star.astype(jnp.float32) - 1.0)[None, None, :]

	m_bar = mag_blocks.mean(axis = 1)                       #(R, N) per-clone estimate

	p  = (m_bar*ref[0]).mean()
	q1 = (m_bar*m_bar).mean()                               #biased up by Var(m_bar)
	q0 = _pair_mean_dev(m_bar, 0).mean()                    #unbiased

	#SE of p from the block-to-block spread: tau_int absorbed, never estimated
	p_block = (mag_blocks*ref).mean(axis = 2)               #(R, B)
	se_p = (jnp.std(p_block, axis = 1, ddof = 1).mean()/jnp.sqrt(float(B*R))
					if B > 1 else jnp.array(jnp.nan, dtype = jnp.float32))

	expected = (jnp.var(mag_blocks, axis = 1, ddof = 1)/B).mean() if B > 1 else \
						 jnp.array(jnp.nan, dtype = jnp.float32)
	observed = q1 - q0

	return MCStats(p = p, q1 = q1, q0 = q0, se_p = se_p,
								 noise_floor = observed, noise_expected = expected,
								 split_ratio = observed/jnp.maximum(expected, _F32_RES),
								 pq_t = (q1 - p)/jnp.maximum(se_p, _F32_RES))


def stop_fired(st: MCStats, pq_sigma, split_ratio) -> jax.Array:
	"""
	Has the surrogate demonstrably left the single pure state at this lambda?

	The paper's condition is p = q1 = q0. Two independent ways for it to break,
	OR'd, because they catch different failure modes:

	  (q1 - p) > pq_sigma*se_p        the DIRECTED departure
	  split_ratio > split_ratio_thr   the clones have scattered into basins

	DIRECTED, not |p - q1|. `mcmc_corr_promt.md` Sec. 4 writes the stop as
	symmetric, and that form is unusable: while the surrogate is perfectly pinned
	p EXCEEDS q1 by tens of sigma - measured on chimera128, (q1-p)/se_p runs
	-44.8, -42.0, -21.9 down the ladder and only flips to +31.4 at the actual
	breakdown. A symmetric test fires at the TOP of the ladder, where nothing is
	wrong. Only the sign change carries the signal.

	The directed test is also the better separated of the two: on the tree
	negative control it stays <= 1.8 where chimera reaches +31, whereas the
	q1 - q0 gap test only separates 3.02 (tree, must not fire) from 3.90
	(chimera, must fire).
	"""
	#Written as products, not ratios. A perfectly pinned chain has se_p and
	#noise_expected both exactly zero, and `q1 - p` is then a float32 residue of
	#order 1e-12; any ratio form turns that into a huge statistic and fires at the
	#TOP of the ladder, where the surrogate is as pure as it will ever be. Flooring
	#each denominator at float32 resolution and multiplying instead makes the
	#degenerate case fire nothing, which is the correct reading of it.
	se_floor  = jnp.maximum(st.se_p, _F32_RES)
	exp_floor = jnp.maximum(st.noise_expected, _F32_RES)

	pq_hit    = (st.q1 - st.p) > pq_sigma*se_floor
	split_hit = st.noise_floor > split_ratio*exp_floor

	return jnp.logical_or(pq_hit, split_hit)


def corr_mean_sem(corr_blocks: List[jax.Array]):
	"""
	`corr_from_blocks` on device: per-group per-block means -> one
	`(n_factors,)` vector in `concatenate(Jt)` order, plus its block-spread SEM.

	The SEM pools R*B blocks as if independent. Consecutive blocks within a chain
	are not - they carry a seam correlation of order tau_int/block_len - so this
	is an optimistic error bar, and `block_len` is the knob that makes it honest.
	"""
	means, sems = [], []
	for x in corr_blocks:                                   #(R, B, n_terms_g)
		R, B, n = x.shape
		flat = x.reshape(R*B, n)
		means.append(flat.mean(axis = 0))
		sems.append(jnp.std(flat, axis = 0, ddof = 1)/jnp.sqrt(float(R*B))
								if R*B > 1 else jnp.zeros((n,), dtype = jnp.float32))
	return jnp.concatenate(means), jnp.concatenate(sems)


# ---------------------------------------------------------------------------
# The lambda ladder. Structural twin of `lbp.surrogate_LBP` (lbp.py:399-545):
# same `lambdas = [start, min, ratio]` contract, same decreasing while_loop,
# same rollback to the last rung that passed.
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames = ['cfg'])
def surrogate_MC(Jh: Jhdata,
								 s: jax.Array,
								 epsilon: jax.Array,
								 colors: jax.Array,
								 J_flat: jax.Array,
								 idx_list: List[jax.Array],
								 beta,
								 lambdas: jax.Array,
								 key,
								 cfg: MCCfg,
								 pq_sigma,
								 split_ratio,
								 beta_div = None):
	"""
	Anneal lambda downward, stop when the surrogate leaves the single pure state,
	back off one rung, and read the correlations there.

	The LBP path does exactly this with message convergence and MAP distance as
	its stopping pair (`lbp.py:498-526`); this does it with the overlaps p, q1, q0.
	`lambdas` supplies the same `[start, min, ratio]` triple the LBP arm reads, but
	only `min` and `ratio` carry over unchanged - see INCREASE below for why `start`
	cannot.

	Structure, and why each piece is where it is:

	- INCREASE, before anything descends. The two arms' stopping rules are not
	  interchangeable and neither are their lambdas. LBP stops on message
	  convergence, which says nothing about whether the surrogate is localised;
	  this stops on p = q1 = q0, which is satisfied TRIVIALLY by an unpinned chain
	  because all three are then 0. Reusing the LBP arm's `lambdas[0]` therefore
	  does not merely start in the wrong place - it makes the whole ladder vacuous.
	  So lambda is first RAISED until `p >= cfg.p_target`, seeded analytically at
	  `atanh(p_target)/(beta*mean(eps))` so the loop corrects rather than searches.
	  `lbp.py:466`'s `run_bp_lmbd_increase` is the same idea against a different
	  criterion. `MCCfg` carries the measurement.

	- WARM START. Clones carry from rung to rung, cold-starting at `r = s*` only
	  on the first. This mirrors LBP reusing the previous rung's messages
	  (`lbp.py:508`), and it is the difference between re-equilibrating once and
	  re-equilibrating at every rung. `mc_blocks` already returned `s_final`;
	  nothing used it before.

	- PERSIST. The stop must hold for `cfg.persist` CONSECUTIVE rungs. Not
	  cosmetic: in the saved chimera128/tree data this is the only thing keeping
	  the tree negative control from firing on a single noisy rung.

	- ROLLBACK via `lam_safe`, which tracks the last rung that did NOT fire. With
	  `persist = 2` that is the rung before the first of the two fires, so the
	  operating lambda is one rung more conservative than the first warning -
	  the same direction `lbp.py:524` backs off in.

	- MONITOR vs CORRELATION phases, with separate budgets and separate
	  accumulators. The monitor phase runs `with_corr = False`, so the per-factor
	  product is never even emitted for the rungs that get discarded.

	:param Jh: CAT-space Hamiltonian, `env.Jh_cat`.
	:param s: reference configuration, bool `(N,)`.
	:param epsilon: `(N,)`, `env.epsilon`.
	:param colors: one colour-class stack, `env.colors_stack[0]`.
	:param J_flat: `(n_factors,)` bare couplings in `concatenate(Jt)` order.
	:param idx_list: `env.corr_idx`, `[j.indices for j in Jh_cat.J]`.
	:param beta: the SAMPLING temperature. Must be where the distribution still has
		spread - order 1 - not an APT rung's beta; see `corr_to_Jtilde`.
	:param beta_div: optional divisor for the surrogate term, to express J_tilde in
		a rung's units. Defaults to `beta`.
	:param lambdas: `[lambda_start, lambda_min, ratio]`, ratio < 1. `lambda_start` is
		only where the INCREASE phase begins; the descent starts from whatever lambda
		that phase had to reach to pin the chains (see `MCCfg.p_target`).
	:param cfg: static `MCCfg`.
	:return: `(corr_flat, Jt_flat, mag_flat, mc_stats)` - `corr_flat` and `Jt_flat`
		are `(n_factors,)`, `mag_flat` is `(N,)`, `mc_stats` a dict of scalars.
	"""
	assert cfg.replicas >= 2, (
		f"surrogate_MC needs at least 2 clones, got {cfg.replicas}: q0 is the "
		f"pairwise CLONE overlap (Eq. B4) and does not exist at R = 1. A "
		f"block-based substitute tracks q1 all the way down and detects nothing.")

	R = cfg.replicas
	N = epsilon.size

	n_mon_total = cfg.monitor_settle_blocks + cfg.monitor_blocks

	def _monitor(s_cl, key_in, lam):
		"""
		One rung: settle, then monitor, magnetisations only, then the overlaps.

		Both phases are one `mc_blocks_clones` call and the settle blocks are
		dropped by a static slice, rather than a second call - the sampler state
		has to flow through anyway, and a slice costs nothing while a second
		launch does.
		"""
		Jh_s = surrogate_Jh(Jh, s, epsilon, lam)
		ks = jax.random.split(key_in, R + 1)
		s_end, mag_b, _ = mc_blocks_clones(Jh_s, s_cl, colors, beta, ks[1:], idx_list,
																			 cfg.monitor_block_len, n_mon_total,
																			 False)
		mon = lax.slice_in_dim(mag_b, cfg.monitor_settle_blocks, n_mon_total, axis = 1)
		return s_end, ks[0], overlap_stats(mon, s)

	#--- INCREASE phase: find a lambda that actually pins ----------------------
	#The analogue of `lbp.surrogate_LBP`'s `run_bp_lmbd_increase` (lbp.py:466-480),
	#which raises lambda until the MAP state is within `tolerance_d` of s*. Here the
	#test is `p >= p_target` directly, since p IS the overlap with s* (Eq. 8) and is
	#already computed by every monitor phase.
	#
	#Without this the ladder is vacuous whenever `lambdas[0]` is calibrated for LBP:
	#the chains never localise, p = q1 = q0 = 0 satisfies the pure-state criterion
	#trivially, nothing fires, and the correlations come from an unpinned surrogate.
	#See `MCCfg`'s docstring for the chimera128 measurement.
	#
	#SEED, so the loop below is a correction and not a search. `cfg.lambda_seed` is a
	#CONSTANT, not a function of beta, because lambda is the control variable and beta
	#very nearly cancels - see `MCCfg` for the measurement. An earlier version here used
	#the mean-field inversion `atanh(p_target)/(beta*mean(eps))`, which assumes the spin
	#feels only the pin; that undershoots by 2-6x precisely because it ignores the
	#couplings, and it undershoots WORST at cold beta, where it is most expensive to
	#correct (measured 4 increase steps at beta = 9.64 against 0 here).
	#
	#Floored at `lambdas[0]` so a config that deliberately starts high keeps its value;
	#the seed can only ever raise the starting point, never lower it.
	lam_seed = jnp.maximum(lambdas[0], cfg.lambda_seed)

	#Cold start at r = s*, matching LBP's message init and Sec. 4 of the spec.
	s0 = jnp.tile(s[None, :], (R, 1))
	s_cl, key, st0 = _monitor(s0, key, lam_seed)

	def inc_cond(c):
		_, _, _, st, n_up = c
		return jnp.logical_and(st.p < cfg.p_target, n_up < cfg.max_increase_steps)

	def inc_body(c):
		s_c, k_c, lam, _, n_up = c
		#divide by the ratio: the increase retraces the descent's own geometry, so a
		#finer `ratio` refines both phases together rather than only one of them.
		lam_up = lam/lambdas[2]
		s_new, k_new, st_new = _monitor(s_c, k_c, lam_up)
		return (s_new, k_new, lam_up, st_new, n_up + 1)

	s_cl, key, lam_start, st0, inc_steps = lax.while_loop(
		inc_cond, inc_body, (s_cl, key, lam_seed, st0, jnp.array(0, dtype = jnp.int32)))

	#`pinned_ok` records whether the increase phase actually reached p_target rather
	#than running out of steps. A False here means every statistic below describes an
	#unpinned surrogate and the whole call should be discarded, not merely distrusted.
	pinned_ok = st0.p >= cfg.p_target

	#--- DESCENT, from the lambda the increase phase settled on ----------------
	fired0 = stop_fired(st0, pq_sigma, split_ratio)

	init = (s_cl, key, lam_start,
					jnp.where(fired0, 1, 0),                 #consecutive fires
					lam_start,                               #lam_safe
					s_cl,                                    #state at lam_safe
					st0, st0,                                #stats now, stats at lam_safe
					jnp.logical_not(fired0))                 #has ANY rung passed?

	def cond(c):
		_, _, lam, n_fired, _, _, _, _, _ = c
		return jnp.logical_and(n_fired < cfg.persist, lam > lambdas[1])

	def body(c):
		s_c, k_c, lam, n_fired, lam_safe, s_safe, _, st_safe, any_pass = c
		lam_new = lam*lambdas[2]
		s_new, k_new, st = _monitor(s_c, k_c, lam_new)
		fired = stop_fired(st, pq_sigma, split_ratio)

		n_new = jnp.where(fired, n_fired + 1, 0)
		#lam_safe follows the last rung that PASSED; on a fire it is left alone
		lam_safe_new = jnp.where(fired, lam_safe, lam_new)
		s_safe_new = jnp.where(fired, s_safe, s_new)
		st_safe_new = jax.tree.map(lambda a, b: jnp.where(fired, a, b), st_safe, st)

		return (s_new, k_new, lam_new, n_new, lam_safe_new, s_safe_new, st, st_safe_new,
						jnp.logical_or(any_pass, jnp.logical_not(fired)))

	(s_cl, key, lam_last, n_fired, lam_op, s_op, st_last, st_op, any_pass) = \
		lax.while_loop(cond, body, init)

	#--- correlations at the operating lambda --------------------------------
	Jh_op = surrogate_Jh(Jh, s, epsilon, lam_op)

	ks = jax.random.split(key, R + 1)
	#settle: discard, no accumulation of anything
	s_set, _, _ = mc_blocks_clones(Jh_op, s_op, colors, beta, ks[1:], idx_list,
																 cfg.corr_block_len, cfg.corr_settle_blocks, False)
	ks2 = jax.random.split(ks[0], R + 1)
	_, mag_c, corr_b = mc_blocks_clones(Jh_op, s_set, colors, beta, ks2[1:], idx_list,
																			cfg.corr_block_len, cfg.corr_blocks, True)

	corr, corr_sem = corr_mean_sem(corr_b)
	Jt = corr_to_Jtilde(corr, J_flat, beta, beta_div)
	st_corr = overlap_stats(mag_c, s)

	n_corr_samples = cfg.corr_blocks*cfg.corr_block_len

	stats = {
		#the operating point, and whether we got there by backing off or by
		#running out of ladder - these are different outcomes and the ledger
		#should not conflate them
		"final_lambda":    lam_op,
		"last_lambda":     lam_last,
		"backed_off":      (n_fired >= cfg.persist).astype(jnp.float32),
		"floor_reached":   (lam_last <= lambdas[1]).astype(jnp.float32),

		#the INCREASE phase: where the descent actually started, and whether it got
		#there. `pinned_ok` False means the chains never reached p_target, so every
		#statistic below describes an UNPINNED surrogate - the run is void, not noisy.
		"init_lambda":     lam_start,
		"init_steps":      inc_steps.astype(jnp.float32),
		"init_p":          st0.p,
		"pinned_ok":       pinned_ok.astype(jnp.float32),

		#NO SAFE RUNG. `lam_safe` is seeded with the descent's own first lambda, so
		#when that rung ALREADY fires the rollback has nothing to roll back to and
		#returns a lambda that itself failed the test. Reported rather than silently
		#returned: it is the signature of a ladder whose start was still too weak, and
		#it is exactly what the shipped `lbp_lambdas` produced before the increase
		#phase existed.
		"no_safe_lambda":  jnp.logical_not(any_pass).astype(jnp.float32),

		#the stop statistics AT the operating lambda
		"p":               st_op.p,
		"q1":              st_op.q1,
		"q0":              st_op.q0,
		"se_p":            st_op.se_p,
		"pq_t":            st_op.pq_t,
		"split_ratio":     st_op.split_ratio,
		"noise_floor":     st_op.noise_floor,

		#and at the rung that tripped it, so a stop can be audited
		"pq_t_last":       st_last.pq_t,
		"split_ratio_last": st_last.split_ratio,

		#correlation-phase quality
		"corr_sem_mean":   corr_sem.mean(),
		"corr_sem_max":    corr_sem.max(),
		"corr_absmean":    jnp.abs(corr).mean(),
		"corr_saturated":  (jnp.abs(corr) >= 1.0 - 1e-6).mean(),
		"Jt_absmean":      jnp.abs(Jt).mean(),
		"Jt_absmax":       jnp.abs(Jt).max(),

		#the F3 ceiling, reported next to what was achieved so an empty backbone
		#is self-diagnosing rather than a mystery
		"Jt_ceiling_mean": jtilde_ceiling(
			J_flat, beta if beta_div is None else beta_div, n_corr_samples).mean(),
		#how much of J_tilde is the surrogate at all, vs the bare coupling. If this
		#is ~0 the backbone is thresholding |J_a| and the anneal is decoration.
		"Jt_surrogate_absmean": jnp.abs(Jt - J_flat).mean(),
		"Jt_surrogate_frac":    (jnp.abs(Jt - J_flat).mean()
														 /jnp.maximum(jnp.abs(Jt).mean(), _F32_RES)),
		"mc_sweeps":       jnp.array(float(n_corr_samples), dtype = jnp.float32),
		"corr_phase_p":    st_corr.p,
	}

	#Per-spin magnetisation from the CORRELATION phase - the same samples `corr` came
	#from, so the two describe one measure rather than two. `get_geometry` has to
	#return this (and `bench_apt.py:1088` reads its dtype), and it is free here: the
	#block means are already materialised.
	#
	#NOT on `lbp_mag`'s scale. That is |mag_raw - pin|, a FIELD magnitude in energy
	#units; this is |<r_i>| in [0, 1]. Nothing thresholds it under `backbone_source`
	#"mc_corr" - growth is gated on J_tilde per factor - but `rlnmc` feeds `mag` to
	#its policy, so the two must not be compared or have one threshold applied to both.
	mag = jnp.abs(mag_c.mean(axis = (0, 1)))

	return corr, Jt, mag, stats
