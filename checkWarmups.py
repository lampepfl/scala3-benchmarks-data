#!/usr/bin/env python3
"""Check whether benchmark warmups are sufficient, from the raw per-iteration data.

Scans raw/<machine>/<jvm>/<patchVersion>/<version>/*.csv and classifies each
(machine, jvm, suite, benchmark, warmup_iterations) into:

  WARMUP_TOO_SHORT  many forks still get faster during the measurement window,
                    at a consistent position: a JIT compilation/deoptimization
                    that lands at a roughly fixed invocation count, shortly
                    after warmup ended. Fixable by increasing @Warmup
                    iterations past the reported settle position.
  MULTI_STATE       fork steady-state (tail) means split into distinct groups
                    that alternate over time: each JVM launch settles
                    permanently into one of several performance states (JIT
                    inlining lottery). More warmup does NOT fix this; the
                    benchmark's hot code must be made JIT-robust (see the
                    kmeans while-loop rewrite in the main repo).
  LEVEL_SHIFT       tail means split into groups but chronologically
                    contiguous: a real performance change over time (compiler
                    regression/improvement or config change), not instability.
  NOISY             spikes only in few forks or at scattered positions:
                    GC/JIT/OS noise, not fixable by warmup.
  OK                no fork deviates more than SLOW_FACTOR from its steady tail.

Usage: python3 checkWarmups.py [-v] [filter]
  -v      also list OK benchmarks
  filter  substring of "machine/jvm/suite.benchmark" to restrict the check

Exit code is 1 if any benchmark is WARMUP_TOO_SHORT or MULTI_STATE.

Entirely written by Claude Fable 5.
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

SLOW_FACTOR = 1.15    # iteration is "slow" if > SLOW_FACTOR * fork tail median
STATE_RATIO = 1.25    # separation of tail-mean clusters that indicates states
MIN_AFFECTED = 0.10   # fork fraction below which slowness counts as noise
MIN_CLUSTER = 0.08    # smallest fork fraction that counts as a "state"
MIN_ALTERNATIONS = 5  # fast/slow alternations over time to call it a lottery
MAX_POS_IQR = 3       # settle positions concentrated => warmup-fixable
MIN_FORKS = 10        # below this, report "insufficient data"


def tail_median(times):
    return median(times[-max(3, len(times) // 3):])


def settle_position(times, ref):
    """1-based iteration after which the fork stays fast, if the slow
    iterations form one contiguous run that ends before the window does."""
    slow = [i for i, t in enumerate(times) if t > SLOW_FACTOR * ref]
    if not slow or slow[-1] == len(times) - 1:
        return None
    if slow[-1] - slow[0] + 1 != len(slow):
        return None
    return slow[-1] + 1


def two_means(values):
    """1-D 2-means: split minimizing within-cluster variance.
    Returns (low_median, high_median, n_high) or None if not separated."""
    ms = sorted(values)
    n = len(ms)
    best, best_score = None, None
    prefix = [0.0]
    prefix_sq = [0.0]
    for v in ms:
        prefix.append(prefix[-1] + v)
        prefix_sq.append(prefix_sq[-1] + v * v)
    for i in range(1, n):
        var_lo = prefix_sq[i] - prefix[i] ** 2 / i
        var_hi = (prefix_sq[n] - prefix_sq[i]) - (prefix[n] - prefix[i]) ** 2 / (n - i)
        score = var_lo + var_hi
        if best_score is None or score < best_score:
            best_score, best = score, i
    i = best
    if min(i, n - i) < max(2, MIN_CLUSTER * n):
        return None
    lo, hi = median(ms[:i]), median(ms[i:])
    if lo <= 0 or hi / lo < STATE_RATIO:
        return None
    return lo, hi, n - i


def quantile(sorted_vals, q):
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def main():
    verbose = "-v" in sys.argv[1:]
    args = [a for a in sys.argv[1:] if a != "-v"]
    pattern = args[0] if args else ""
    root = Path(__file__).resolve().parent / "raw"
    if not root.is_dir():
        sys.exit(f"raw/ not found next to this script ({root})")

    # key -> list of (chronological stamp, iteration times)
    forks = defaultdict(list)
    for path in sorted(root.rglob("*.csv")):
        machine, jvm = path.relative_to(root).parts[0:2]
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                times = [float(x) for x in row["times"].split()]
                if len(times) >= 4:
                    key = (machine, jvm, row["suite"], row["benchmark"],
                           row["warmup_iterations"])
                    forks[key].append((path.stem, times))

    problems, ok_names = 0, []
    for (machine, jvm, suite, bench, warmup), runs in sorted(forks.items()):
        name = f"{machine}/{jvm}/{suite}.{bench}"
        if pattern and pattern not in name:
            continue
        runs.sort(key=lambda r: r[0])
        n = len(runs)
        tails = [tail_median(t) for _, t in runs]
        iters = max(len(t) for _, t in runs)

        verdict, detail = "OK", ""
        if n < MIN_FORKS:
            verdict, detail = "SKIPPED", f"insufficient data ({n} forks)"
        elif (states := two_means(tails)) is not None:
            lo, hi, n_hi = states
            threshold = (lo + hi) / 2
            labels = [t > threshold for t in tails]
            alternations = sum(a != b for a, b in zip(labels, labels[1:]))
            if alternations >= MIN_ALTERNATIONS:
                verdict = "MULTI_STATE"
                detail = (f"{n_hi}/{n} forks settle {hi / lo:.2f}x slower "
                          f"(~{lo:.3g} vs ~{hi:.3g}, {alternations} fast/slow "
                          f"alternations over time); warmup cannot fix this")
            else:
                verdict = "LEVEL_SHIFT"
                detail = (f"tail means ~{lo:.3g} -> ~{hi:.3g} "
                          f"({hi / lo:.2f}x) in {alternations + 1} contiguous "
                          f"eras: likely a real change over time, not warmup")

        if verdict == "OK":
            settles = []
            spiky = 0
            for _, t in runs:
                pos = settle_position(t, tail_median(t))
                if pos is not None:
                    settles.append(pos)
                elif any(x > SLOW_FACTOR * tail_median(t) for x in t):
                    spiky += 1
            if settles:
                settles.sort()
                iqr = quantile(settles, 0.75) - quantile(settles, 0.25)
                concentrated = iqr <= MAX_POS_IQR
                if len(settles) / n >= MIN_AFFECTED and concentrated:
                    suggest = quantile(settles, 0.90)
                    top = Counter(settles).most_common(3)
                    verdict = "WARMUP_TOO_SHORT"
                    detail = (f"{len(settles)}/{n} forks settle during "
                              f"measurement, at consistent positions "
                              f"{[f'iter {p}: {c} forks' for p, c in top]} "
                              f"(of {iters} iters); warmup_iterations={warmup},"
                              f" suggest at least +{suggest}")
                else:
                    verdict = "NOISY"
                    where = ("scattered positions" if not concentrated
                             else "consistent position but rare")
                    detail = (f"{len(settles) + spiky}/{n} forks with spikes, "
                              f"{where}; not fixable by warmup")
            elif spiky:
                verdict = "NOISY"
                detail = f"{spiky}/{n} forks with trailing/erratic spikes"

        if verdict == "OK":
            ok_names.append(name)
            if verbose:
                print(f"{'OK':17s} {name}")
        else:
            if verdict in ("WARMUP_TOO_SHORT", "MULTI_STATE"):
                problems += 1
            print(f"{verdict:17s} {name}: {detail}")

    print(f"\n{len(ok_names)} benchmarks OK, {problems} with warmup/state problems.")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
