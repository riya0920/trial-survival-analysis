"""Does Efron actually beat Breslow, and by how much?

The old docstring said Breslow was fine because "the difference is negligible
unless ties are heavy". That is the received wisdom, it is probably true, and
it had never been checked in this project. This checks it.

THE DESIGN
----------
Simulate from a KNOWN hazard ratio, then round the event times to a coarser and
coarser grid. Rounding is what creates ties, and it is not an artificial
manipulation: trial data arrives on a day grid, and registry and claims data
routinely arrive on a month grid. Each grid gives a different tie burden on the
SAME underlying data, so the two estimators can be compared at several tie
densities without changing anything else.

Then measure the thing that matters, which is not "do the numbers differ" but:

    WHICH ESTIMATOR IS CLOSER TO THE PLANTED TRUTH, and in which DIRECTION is
    the other one wrong?

Breslow is expected to attenuate -- bias toward HR = 1 -- because it uses the
full risk set for every death in a tied set, double-counting subjects who have
already failed and inflating the denominator. Attenuation matters
asymmetrically in a trial: it understates a treatment benefit (conservative)
and understates a harm signal (not conservative at all).

Averaged over replicates, because a single simulated trial cannot distinguish
an estimator's bias from its sampling noise.

Run:  python run_ties.py --reps 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import simulate as SIM
from survival import cox_ph

# read from the generator rather than restated, so the two cannot drift
TRUE_HR = SIM.TRUE_HR_TREATMENT
GRIDS = [
    ("continuous", None),
    ("daily", 1 / 30.4375),      # months per day
    ("weekly", 7 / 30.4375),
    ("monthly", 1.0),
    ("quarterly", 3.0),
]


def _round_to(t, grid):
    if grid is None:
        return t
    return np.maximum(grid, np.round(t / grid) * grid)


def tie_burden(time, event):
    """Fraction of events that share their time with another event."""
    ev = np.asarray(time)[np.asarray(event) == 1]
    if len(ev) == 0:
        return 0.0, 0
    _u, counts = np.unique(ev, return_counts=True)
    tied = counts[counts > 1].sum()
    return float(tied / len(ev)), int(counts.max())


def one_rep(seed, n=400):
    d = SIM.simulate_trial(n=n, seed=seed)
    t = np.asarray(d["time"], dtype=float)
    e = np.asarray(d["event"], dtype=int)
    x = np.asarray(d["arm"], dtype=float).reshape(-1, 1)

    out = {}
    for label, grid in GRIDS:
        tt = _round_to(t, grid)
        frac, biggest = tie_burden(tt, e)
        row = {"tied_fraction": frac, "largest_tie": biggest}
        for method in ("breslow", "efron"):
            fit = cox_ph(tt, e, x, ties=method)
            row[method] = float(fit["hr"][0])
        out[label] = row
    return out


def main(reps=200, n=400):
    per_grid = {label: {"breslow": [], "efron": [], "tied": [], "biggest": []}
                for label, _g in GRIDS}
    for r in range(reps):
        rep = one_rep(1000 + r, n)
        for label, row in rep.items():
            per_grid[label]["breslow"].append(row["breslow"])
            per_grid[label]["efron"].append(row["efron"])
            per_grid[label]["tied"].append(row["tied_fraction"])
            per_grid[label]["biggest"].append(row["largest_tie"])

    print("=" * 78)
    print(f"TIE HANDLING -- {reps} replicates, n={n}, planted HR = {TRUE_HR}")
    print("=" * 78)
    print(f"  {'grid':<12}{'% events tied':>14}{'max tie':>9}"
          f"{'Breslow HR':>12}{'Efron HR':>11}{'winner':>10}")

    rows = []
    for label, _g in GRIDS:
        d = per_grid[label]
        b, e = float(np.mean(d["breslow"])), float(np.mean(d["efron"]))
        tied = float(np.mean(d["tied"]))
        biggest = int(np.mean(d["biggest"]))
        eb, ee = abs(b - TRUE_HR), abs(e - TRUE_HR)
        winner = ("Efron" if ee < eb - 1e-6 else
                  "Breslow" if eb < ee - 1e-6 else "tie")
        print(f"  {label:<12}{tied:>13.1%}{biggest:>9}"
              f"{b:>12.4f}{e:>11.4f}{winner:>10}")
        rows.append({"grid": label, "tied_fraction": tied,
                     "largest_tie": biggest, "breslow_hr": b, "efron_hr": e,
                     "breslow_error": eb, "efron_error": ee, "winner": winner})

    print("\n" + "-" * 78)
    print("WHAT THIS SHOWS")
    print("-" * 78)
    cont = rows[0]
    print(f"  With no ties the two estimators agree to "
          f"{abs(cont['breslow_hr'] - cont['efron_hr']):.2e} -- they are the")
    print("  same expression when every event time is unique, which is the")
    print("  first thing to confirm before believing any difference elsewhere.")

    # MEASURED AGAINST THE CONTINUOUS ESTIMATE, NOT AGAINST THE PLANTED HR.
    # Both estimators sit near 0.735 against a planted 0.70 even with NO ties,
    # which is finite-sample bias at n=400 and has nothing to do with tie
    # handling. Differencing against the tie-free estimate on the SAME data
    # isolates the tie effect from that baseline. The first version of this
    # report quoted distance-from-truth, and so silently attributed the
    # finite-sample bias to Breslow.
    ref = rows[0]["efron_hr"]
    worst = max(rows, key=lambda r: r["tied_fraction"])
    print("")
    print("  Attenuation attributable to TIE HANDLING, measured against the")
    print(f"  tie-free estimate on the same data ({ref:.4f}):")
    print(f"    {'grid':<12}{'Breslow drift':>15}{'Efron drift':>14}")
    for r in rows[1:]:
        print(f"    {r['grid']:<12}{r['breslow_hr'] - ref:>+15.4f}"
              f"{r['efron_hr'] - ref:>+14.4f}")
    print("")
    print(f"  At the {worst['grid']} grid ({worst['tied_fraction']:.0%} of "
          f"events tied, largest tie {worst['largest_tie']})")
    print(f"  Breslow drifts {worst['breslow_hr'] - ref:+.4f} toward the null; "
          f"Efron drifts {worst['efron_hr'] - ref:+.4f}.")
    print("  Efron is very nearly invariant to the grid, which is the property")
    print("  worth having: an estimate should depend on the data, not on how")
    print("  coarsely somebody recorded the dates.")

    n_efron = sum(1 for r in rows[1:] if r["winner"] == "Efron")
    print(f"\n  Efron is closer to the planted HR on {n_efron} of "
          f"{len(rows) - 1} tied grids.")
    print("  Breslow attenuates toward the null and Efron essentially does")
    print("  not -- the drift table above, not a general claim about the two")
    print("  estimators. The DIRECTION is what matters: attenuation understates")
    print("  a treatment benefit, which is conservative, and understates a HARM")
    print("  signal by exactly as much, which is not conservative at all.")
    print("\n  The old default was Breslow, justified by 'the difference is")
    print("  negligible unless ties are heavy'. The claim is defensible and")
    print("  was never checked. Monthly-rounded follow-up is not an exotic")
    print("  case; it is what registry and claims data look like.")

    os.makedirs("out", exist_ok=True)
    with open("out/ties.json", "w") as fh:
        json.dump({"true_hr": TRUE_HR, "reps": reps, "n": n, "grids": rows},
                  fh, indent=2)
    print("\nwrote out/ties.json")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    main(a.reps, a.n)
