"""
DIMACS CNF -> native high-order Ising, in the sparse text format readers.read_ising_txt reads.

WHY THIS EXISTS. backbone_source "lbp_corr" is pising-only (nmcgym.py raises on anything
else): Eq. (C3) presumes an Ising k-body factor J*prod(s), and a wcnf clause indicator is a
different object, so the effective couplings would be meaningless rather than merely
inaccurate. The uf4sat instances are wcnf, so the correlation-based backbone cannot see them
at all. Converting the CNF to its native high-order spin polynomial fixes that.

NOT a Rosenberg/QUBO reduction. The sampler handles arbitrary order directly, so a k-SAT
clause becomes one degree-1..k monomial group rather than a quadratic gadget with ancillas.

THE MATH.

  U_j = 1 iff clause j is unsatisfied, and H = sum_j w_j*U_j is the count we minimise.
  Each literal contributes its own "is false" indicator, so U_j is their product:

      g(x_i) = (1 - x_i)  for a positive literal x_i
      g(x_i) = x_i        for a negated literal ~x_i
      U_j    = prod_{i in clause j} g(x_i)

  Substituting x_i = (1 + s_i)/2 with sigma_i = +1 for a positive literal and -1 for a
  negated one turns each indicator into (1 - sigma_i*s_i)/2, so

      U_j = (1/2^K_j) * sum_{S subset of clause j} (-1)^|S| * (prod_{i in S} sigma_i) * prod_{i in S} s_i

  and therefore

      H = sum_S c_S * prod_{i in S} s_i  +  c_empty,
      c_{j,S} = w_j * (-1)^|S| * (prod_{i in S} sigma_{j,i}) / 2^K_j

  Monomials sharing a support S are summed across clauses. c_empty = sum_j w_j/2^K_j is a
  constant offset and is dropped - it does not move the argmin, but it is NOT zero and the
  header records it, because the run's `ground_states` must carry -c_empty rather than 0.

OUTPUT FORMAT (readers.read_ising_txt, selected by bench_config.instance_format "ising_txt"):

  - one term per line, `i j ... v`: k indices then the coefficient, arity from token count
  - a bias is written in the legacy diagonal form `i i v`, matching chimera and ising_mixed
  - indices are 1-BASED. Set `one_based: true` in the bench config rather than relying on
    read_ising_txt's auto-detection: it decides from the top index against N, and a CNF
    whose highest-numbered variable happens to be unused would make that ambiguous and raise.
  - a support is written ONCE. The format's rule is that a repeated term restates the same
    value and is never summed, so the merge has to happen here.

SIGN. These files state E = +(sum J*prod(s) + sum h*s) while get_energy computes -(...);
ising_txt_for_jax(negate = True), the default, bridges the two. So coefficients are written
as plain c_S and the composition reproduces H exactly. Do NOT set negate_couplings false.

PRECISION. Every c_S is a multiple of 1/2^k and the totals stay far below 2^24/2^k, so every
partial sum is exactly representable in float32 - the accumulation is exact, and rescaling by
a power of two would buy nothing. write_ising_txt asserts each coefficient survives the trip
through float and through its own text unchanged.
"""

import os
import sys
import argparse

from collections import defaultdict
from fractions import Fraction
from itertools import combinations
from typing import Any, Dict, List, Tuple

sys.path.append('.')


def read_dimacs(instance_path: str,
								weighted: bool = False) -> Tuple[int, int, List[List[int]], List[Fraction]]:
	"""
	Parse a DIMACS CNF into (N, M, clauses, weights).

	N comes from the `p cnf N M` header, never from the largest literal seen: a variable
	that appears in no clause is still a spin, and inferring N would silently drop it.

	Tolerated and skipped: `c` comments, blank lines, a stray `%` (whole line or trailing),
	and the lone `0` terminator SATLIB writes after it. Literals are read as a token stream
	terminated by `0`, so a clause split across lines parses the same as one on a single
	line. `weighted` reads one leading weight per clause, the convention readers.get_wcnf
	calls `no_weights = False`.

	Clause-level normalisation, both of which read_ising_txt would otherwise reject or
	silently mis-order downstream:
	  - a repeated literal is collapsed (x OR x == x)
	  - a clause containing both x and ~x is a tautology, U_j == 0, and is dropped
	"""
	n_vars = None
	n_clauses = None

	tokens: List[str] = []

	with open(instance_path, 'r') as f:
		for lineno, line in enumerate(f, 1):
			line = line.split('%')[0].strip()

			if not line or line.startswith('c'):
				continue

			if line.startswith('p'):
				tok = line.split()
				if len(tok) < 4 or tok[1] != "cnf":
					raise ValueError(f"{instance_path}:{lineno}: expected `p cnf N M`, got {line!r}")
				n_vars, n_clauses = int(tok[2]), int(tok[3])
				continue

			tokens.extend(line.split())

	if n_vars is None:
		raise ValueError(f"{instance_path}: no `p cnf N M` header")

	clauses: List[List[int]] = []
	weights: List[Fraction] = []

	n_empty = 0
	n_tauto = 0

	i = 0
	while i < len(tokens):
		if weighted:
			try:
				w = Fraction(tokens[i])
			except ValueError:
				raise ValueError(f"{instance_path}: could not parse clause weight {tokens[i]!r}")
			i += 1
		else:
			w = Fraction(1)

		lits: List[int] = []
		while i < len(tokens):
			try:
				l = int(tokens[i])
			except ValueError:
				raise ValueError(f"{instance_path}: could not parse literal {tokens[i]!r}")
			i += 1
			if l == 0:
				break
			lits.append(l)
		else:
			if lits:
				raise ValueError(f"{instance_path}: clause {lits} is not terminated by 0")

		#the SATLIB terminator: a `%` line followed by a lone `0`. Nothing to convert.
		if not lits:
			n_empty += 1
			continue

		#x OR x == x, and a support must not repeat an index: s_i*s_i = 1 would silently
		#lower the order, which read_ising_txt rejects outright
		seen: Dict[int, int] = {}
		tauto = False
		uniq: List[int] = []
		for l in lits:
			v = abs(l)
			if v in seen:
				if seen[v] != l:
					tauto = True
					break
				continue
			seen[v] = l
			uniq.append(l)

		if tauto:
			n_tauto += 1
			continue

		for l in uniq:
			if abs(l) > n_vars:
				raise ValueError(
					f"{instance_path}: literal {l} exceeds the header's N = {n_vars}"
				)

		clauses.append(uniq)
		weights.append(w)

	if n_clauses is not None and len(clauses) + n_tauto != n_clauses and n_empty <= 1:
		print(f"NOTE: {os.path.basename(instance_path)}: header declares {n_clauses} clauses, "
					f"read {len(clauses)} usable ({n_tauto} tautologies dropped)")

	return n_vars, len(clauses), clauses, weights


def cnf_to_monomials(clauses: List[List[int]],
										 weights: List[Fraction]) -> Tuple[Dict[Tuple[int, ...], Fraction], Fraction]:
	"""
	Expand every clause over its 2^K subsets and merge by support.

	Returns (monomials, constant): monomials maps a 1-based sorted index tuple to its
	coefficient, and `constant` is the dropped degree-0 term sum_j w_j/2^K_j.

	Exact arithmetic throughout (Fraction). Every coefficient is w/2^K up to sign, so the
	denominators are powers of two and nothing here rounds; a support whose contributions
	cancel exactly is removed rather than written as a zero-weight term.
	"""
	mono: Dict[Tuple[int, ...], Fraction] = defaultdict(Fraction)
	constant = Fraction(0)

	for lits, w in zip(clauses, weights):
		K = len(lits)

		base = Fraction(w, 2**K)

		var = [abs(l) for l in lits]
		sig = [1 if l > 0 else -1 for l in lits]

		#S = empty set: the constant. Split out so the inner loop never branches on it.
		constant += base

		for m in range(1, K + 1):
			sign_m = -1 if (m % 2) else 1
			for S in combinations(range(K), m):
				sgn = sign_m
				for i in S:
					sgn *= sig[i]

				mono[tuple(sorted(var[i] for i in S))] += base*sgn

	return {k: v for k, v in mono.items() if v != 0}, constant


def _fmt(v: Fraction) -> str:
	"""
	Coefficient as text, asserted to round-trip exactly.

	Every c_S is a dyadic rational well inside float's exact range, so float(v) is exact
	and repr() is the shortest string that reads back as the same float. Both halves are
	checked rather than assumed - this is the one place the exact arithmetic above could
	silently become inexact.
	"""
	x = float(v)

	if Fraction(x) != v:
		raise ValueError(f"coefficient {v} is not exactly representable as a float")

	s = repr(x)

	if float(s) != x:
		raise ValueError(f"coefficient {v} does not round-trip through {s!r}")

	return s


def write_ising_txt(out_path: str,
										monomials: Dict[Tuple[int, ...], Fraction],
										constant: Fraction,
										N: int,
										source: str,
										n_clauses: int) -> Dict[int, int]:
	"""
	Write the monomials in read_ising_txt's format. Returns {order: count}.

	Biases use the legacy `i i v` diagonal form, which is what chimera and ising_mixed
	ship; degree-2+ terms follow, sorted by (order, indices) so two conversions of the same
	instance are byte-identical and a diff between instances is meaningful.
	"""
	by_order: Dict[int, List[Tuple[Tuple[int, ...], Fraction]]] = defaultdict(list)
	for k, v in monomials.items():
		by_order[len(k)].append((k, v))

	counts = {k: len(v) for k, v in by_order.items()}

	lines = [
		f"# {os.path.basename(source)} converted to native high-order Ising by problems/cnf_to_ising.py",
		f"#",
		f"# E(s) = + sum J*prod(s) + sum h*s   (the file convention; ising_txt_for_jax negates)",
		f"#",
		f"# N               : {N}",
		f"# clauses         : {n_clauses}",
		f"# orders present  : {sorted(counts)}",
		f"# terms per order : " + ", ".join(f"k={k}: {counts[k]}" for k in sorted(counts)),
		f"# index base      : 1-based  (set one_based: true in the bench config)",
		f"#",
		f"# dropped constant: {_fmt(constant)}   (= sum_j w_j/2^K_j)",
		f"# E(s) = (unsatisfied clause weight) - {_fmt(constant)}, so a satisfying assignment",
		f"# has E = {_fmt(-constant)} - that, not 0, is this instance's `ground_states` entry.",
	]

	for k in sorted(by_order):
		for idx, v in sorted(by_order[k]):
			if k == 1:
				lines.append(f"{idx[0]} {idx[0]} {_fmt(v)}")
			else:
				lines.append(" ".join(str(i) for i in idx) + " " + _fmt(v))

	with open(out_path, 'w') as f:
		f.write("\n".join(lines) + "\n")

	return counts


def convert(cnf_path: str,
						out_path: str,
						weighted: bool = False,
						N: Any = None) -> Dict[str, Any]:
	"""Convert one .cnf to one ising .txt. Returns a summary dict."""
	n_vars, n_clauses, clauses, weights = read_dimacs(cnf_path, weighted = weighted)

	if N is not None and N != n_vars:
		raise ValueError(f"{cnf_path}: header declares N = {n_vars}, expected {N}")

	monomials, constant = cnf_to_monomials(clauses, weights)

	counts = write_ising_txt(out_path, monomials, constant, n_vars, cnf_path, n_clauses)

	return {
		"src": cnf_path,
		"dst": out_path,
		"N": n_vars,
		"clauses": n_clauses,
		"constant": constant,
		"ground_state": -constant,
		"counts": counts,
		"orders": tuple(sorted(k for k in counts if k >= 2)),
	}


def _parse_ids(spec: str) -> List[int]:
	ids: List[int] = []
	for part in spec.split(','):
		part = part.strip()
		if not part:
			continue
		if '-' in part:
			a, b = part.split('-', 1)
			ids.extend(range(int(a), int(b) + 1))
		else:
			ids.append(int(part))
	return ids


def main(argv = None):
	p = argparse.ArgumentParser(description = __doc__.split("\n")[1],
															formatter_class = argparse.RawDescriptionHelpFormatter)

	p.add_argument("src", nargs = "?", help = "single .cnf to convert")
	p.add_argument("dst", nargs = "?", help = "output .txt for the single-file form")

	p.add_argument("--src-dir",  help = "directory of .cnf instances (batch form)")
	p.add_argument("--dst-dir",  help = "output directory (batch form); created if absent")
	p.add_argument("--in-prefix",  default = "uf4sat",
								 help = "input stem is <in-prefix>_n<N>_m<M>_<id>.cnf")
	p.add_argument("--out-prefix", default = "uf4sat_ising",
								 help = "output stem is <out-prefix>_n<N>_<id>.txt, which is the name "
												"bench.py builds for instance_format ising_txt")
	p.add_argument("-N", "--N", type = int, help = "expected variable count (checked against the header)")
	p.add_argument("-M", "--M", type = int, help = "clause count, used only to build the input filename")
	p.add_argument("--ids", default = "1-64", help = "instance ids, e.g. 1-64 or 1,5,9")

	p.add_argument("--weighted", action = "store_true",
								 help = "each clause line begins with a weight (readers.get_wcnf's no_weights = False)")

	args = p.parse_args(argv)

	if args.src:
		if not args.dst:
			p.error("the single-file form needs both src and dst")
		info = convert(args.src, args.dst, weighted = args.weighted, N = args.N)
		print(f"{os.path.basename(info['src'])} -> {os.path.basename(info['dst'])}: "
					f"N={info['N']}, {info['clauses']} clauses, "
					f"{sum(info['counts'].values())} terms {info['counts']}, "
					f"ground state {float(info['ground_state'])}")
		return 0

	if not (args.src_dir and args.dst_dir and args.N and args.M):
		p.error("the batch form needs --src-dir, --dst-dir, --N and --M")

	os.makedirs(args.dst_dir, exist_ok = True)

	infos = []
	for i in _parse_ids(args.ids):
		cnf = os.path.join(args.src_dir, f"{args.in_prefix}_n{args.N}_m{args.M}_{i}.cnf")
		txt = os.path.join(args.dst_dir, f"{args.out_prefix}_n{args.N}_{i}.txt")

		if not os.path.isfile(cnf):
			raise FileNotFoundError(cnf)

		infos.append(convert(cnf, txt, weighted = args.weighted, N = args.N))
		print(f"  {os.path.basename(cnf)} -> {os.path.basename(txt)}  "
					f"{sum(infos[-1]['counts'].values())} terms")

	#read_utils._assert_matching_arities takes the order count from instance 0 and loops
	#range() of it, so an instance carrying a DIFFERENT set of orders either vanishes from
	#the batch silently or fails deep inside jnp.stack. Catch it here, where the message can
	#name the file.
	orders = {info["orders"] for info in infos}
	if len(orders) != 1:
		offenders = {info["dst"]: info["orders"] for info in infos}
		raise ValueError(
			"the converted instances do not all carry the same SET of interaction orders, so "
			f"they cannot be batched: {sorted(orders)}.\n{offenders}"
		)

	gs = {info["ground_state"] for info in infos}

	print()
	print(f"converted {len(infos)} instances, orders {sorted(orders)[0]} + biases")
	if len(gs) == 1:
		g = float(next(iter(gs)))
		print(f"ground_states entry for the bench config: {g}   (= -sum_j w_j/2^K_j)")
	else:
		print(f"ground states differ across instances: {sorted(float(x) for x in gs)}")
	print("bench config keys: problem_class = <the --dst-dir's parent name>, "
				"instance_format = \"ising_txt\", one_based = true, problem_mode = \"pising\"")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
