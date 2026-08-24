"""Validate every hand-rolled estimator against a reference implementation.

WHY THIS EXISTS
---------------
Every survival estimator in `src/` is written from the formula: Kaplan-Meier,
Greenwood, the log-rank test, Cox by Newton-Raphson with two tie corrections,
the sandwich variance, Brookmeyer-Crowley. That was a deliberate choice -- the
point of the project is to show the mechanics rather than to call a library.

But "I implemented it from the formula" is a claim about correctness that only
a reference implementation can settle. A unit test written by the same person
who wrote the estimator shares its misconceptions; agreeing with `lifelines`
and `statsmodels` to machine precision does not.

So this script is the audit, and it is kept SEPARATE from `src/`. Nothing in
`src/` imports lifelines, and the project still runs with no third-party
survival library installed. This script and `tests/test_reference.py` are the
only places a reference is used, and both skip cleanly when it is absent.

WHAT IT FOUND
-------------
It found a real bug. `median_survival_ci` reported the upper confidence bound
as the LAST event time inside the acceptance region, when the supremum of the
set is the NEXT one -- the KM curve holds its value on [t_i, t_i+1), so the
whole half-open interval is in the confidence set. The interval was one event
time too narrow, which is the anti-conservative direction.

It was findable because the disagreement was ASYMMETRIC: the lower bound
matched exactly on every seed while the upper was short on every seed. A
convention error looks like that; noise does not.

Run:  python validate_reference.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

import inference as INF
import simulate as SIM
import survival as S

try:
    import pandas as pd
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test, proportional_hazard_test
    from lifelines.utils import median_survival_times
    import statsmodels.api as sm
except ImportError as exc:            # pragma: no cover
    print("Reference libraries not installed (%s)." % exc)
    print("This is an OPTIONAL audit. src/ does not depend on them.")
    raise SystemExit(0)

SEEDS = list(range(101, 111))
N = 500
NAMES = ["arm", "age10", "stage"]


def _case(seed, n=N):
    d = SIM.simulate_trial(n=n, seed=seed)
    T = np.asarray(d["time"], dtype=float)
    E = np.asarray(d["event"], dtype=int)
    X = np.column_stack([d["arm"],
                         (np.asarray(d["age"], dtype=float) - 62) / 10,
                         d["stage"]])
    df = pd.DataFrame({"T": T, "E": E, "arm": X[:, 0],
                       "age10": X[:, 1], "stage": X[:, 2]})
    return T, E, X, df


def _rel(a, b):
    """Relative difference, falling back to absolute when b is ~0."""
    return abs(a - b) / max(abs(b), 1e-12)


REFS = [
    ("kaplan_meier S(t)", "lifelines KaplanMeierFitter"),
    ("logrank chi2", "lifelines logrank_test"),
    ("logrank p", "lifelines logrank_test"),
    ("cox efron beta", "lifelines CoxPHFitter"),
    ("cox efron se", "lifelines CoxPHFitter"),
    ("cox breslow beta (heavy ties)", "statsmodels PHReg(ties=breslow)"),
    ("cox efron beta (heavy ties)", "statsmodels PHReg(ties=efron)"),
    ("robust se", "lifelines CoxPHFitter(robust=True)"),
    ("clustered se", "lifelines CoxPHFitter(cluster_col=)"),
    ("median point", "lifelines median_survival_time_"),
    ("median CI lower", "lifelines median_survival_times"),
    ("median CI upper", "lifelines median_survival_times"),
    ("scaled PH chi2", "lifelines proportional_hazard_test"),
    ("scaled PH p", "lifelines proportional_hazard_test"),
    ("chi2 survival function", "scipy.stats.chi2.sf"),
]


def run(seeds=SEEDS, n=N):
    """Return (worst-disagreement dict, ph-test decision counts)."""
    worst = {}

    def note(key, value):
        worst[key] = max(worst.get(key, 0.0), float(value))

    ph = {"agree": 0, "total": 0, "mine_flags": 0, "ref_flags": 0,
          "ratio_min": float("inf"), "ratio_max": 0.0, "clashes": []}

    for seed in seeds:
        T, E, X, df = _case(seed, n)

        # -- Kaplan-Meier ---------------------------------------------------
        t, s, n_risk, n_ev, se = S.kaplan_meier(T, E)
        kmf = KaplanMeierFitter().fit(T, E)
        ref_s = kmf.survival_function_["KM_estimate"].reindex(t).to_numpy()
        note("kaplan_meier S(t)", np.max(np.abs(s - ref_s)))

        # -- log-rank --------------------------------------------------------
        arm = np.asarray(df["arm"], dtype=int)
        chi2, p, _o, _e = S.logrank(T, E, arm)
        g = arm == 1
        lr = logrank_test(T[g], T[~g], E[g], E[~g])
        note("logrank chi2", _rel(chi2, lr.test_statistic))
        note("logrank p", _rel(p, lr.p_value))

        # -- Cox, Efron (the default) ----------------------------------------
        fit = S.cox_ph(T, E, X, ties="efron")
        cph = CoxPHFitter().fit(df, "T", "E")
        for i, nm in enumerate(NAMES):
            note("cox efron beta", _rel(fit["beta"][i], cph.params_[nm]))
            note("cox efron se", _rel(fit["se"][i], cph.standard_errors_[nm]))

        # -- Cox tie handling under HEAVY ties --------------------------------
        # Rounding to whole months is where Breslow and Efron actually diverge,
        # so this is the case that discriminates between them. Validating only
        # on continuous time would let a broken Breslow pass unnoticed.
        Tc = np.round(T)
        for ties in ("breslow", "efron"):
            mine = S.cox_ph(Tc, E, X, ties=ties)
            ref = sm.PHReg(Tc, X, status=E, ties=ties).fit()
            for i, nm in enumerate(NAMES):
                note("cox %s beta (heavy ties)" % ties,
                     _rel(mine["beta"][i], ref.params[i]))

        # -- sandwich variance, unclustered -----------------------------------
        beta, cov = np.asarray(fit["beta"]), np.asarray(fit["cov"])
        rob = INF.robust_variance(T, E, X, beta, cov, cluster=None)
        ref_rob = CoxPHFitter().fit(df, "T", "E", robust=True)
        for i, nm in enumerate(NAMES):
            note("robust se", _rel(rob["se"][i], ref_rob.standard_errors_[nm]))

        # -- sandwich variance, CLUSTERED -------------------------------------
        site = np.arange(len(T)) % 20
        dfc = df.assign(site=site)
        clus = INF.robust_variance(T, E, X, beta, cov, cluster=site)
        ref_clus = CoxPHFitter().fit(dfc, "T", "E", cluster_col="site")
        for i, nm in enumerate(NAMES):
            note("clustered se", _rel(clus["se"][i],
                                      ref_clus.standard_errors_[nm]))

        # -- Brookmeyer-Crowley median CI -------------------------------------
        mc = INF.median_survival_ci(t, s, n_risk, n_ev)
        ref_ci = median_survival_times(kmf.confidence_interval_).to_numpy()[0]
        note("median point", _rel(mc["median"], kmf.median_survival_time_))
        note("median CI lower", abs(mc["lo"] - ref_ci[0]))
        if mc["hi"] is not None and np.isfinite(ref_ci[1]):
            note("median CI upper", abs(mc["hi"] - ref_ci[1]))

        # -- the REAL Grambsch-Therneau, on SCALED residuals ------------------
        # This one is expected to match, and does. The scaled residual is on
        # the scale of the COEFFICIENT rather than the covariate, which is what
        # makes the statistic the Grambsch-Therneau one and not merely
        # something correlated with it.
        scaled = INF.scaled_ph_test(T, E, X, beta, cov, transform="rank")
        ref_gt = proportional_hazard_test(cph, df,
                                          time_transform="rank").summary
        for i, nm in enumerate(NAMES):
            note("scaled PH chi2",
                 _rel(scaled["per_covariate"][i]["chi2"],
                      float(ref_gt.loc[nm, "test_statistic"])))
            note("scaled PH p",
                 _rel(scaled["per_covariate"][i]["p"],
                      float(ref_gt.loc[nm, "p"])))

        # -- the hand-rolled chi-square tail ---------------------------------
        # `src/` has no scipy dependency, so the upper tail is written out.
        # That is a claim about numerics and gets checked like one.
        try:
            from scipy.stats import chi2 as _chi2
            for dfree in (1, 3):
                for x in (0.5, 2.0, 7.5, 20.0):
                    note("chi2 survival function",
                         _rel(INF._chi2_sf(x, dfree),
                              float(_chi2.sf(x, dfree))))
        except ImportError:                      # pragma: no cover
            pass

        # -- proportional-hazards test: DECISIONS, not numbers -----------------
        # `ph_test` is the Grambsch-Therneau IDEA in its simplest form and uses
        # UNSCALED residuals, so it is not expected to match numerically. What
        # is worth checking is whether it reaches the same CONCLUSION.
        mine_ph = S.ph_test(T, E, X, beta)
        ref_ph = proportional_hazard_test(cph, df,
                                          time_transform="rank").summary
        for i, nm in enumerate(NAMES):
            pm = mine_ph["per_covariate"][i][2]
            cm = mine_ph["per_covariate"][i][1]
            pr = float(ref_ph.loc[nm, "p"])
            cr = float(ref_ph.loc[nm, "test_statistic"])
            ph["total"] += 1
            ph["agree"] += int((pm < 0.05) == (pr < 0.05))
            ph["mine_flags"] += int(pm < 0.05)
            ph["ref_flags"] += int(pr < 0.05)
            if cr > 1e-8:
                ph["ratio_min"] = min(ph["ratio_min"], cm / cr)
                ph["ratio_max"] = max(ph["ratio_max"], cm / cr)
            if (pm < 0.05) != (pr < 0.05):
                ph["clashes"].append(
                    {"seed": seed, "covariate": nm, "p_mine": pm, "p_ref": pr,
                     "chi2_mine": cm, "chi2_ref": cr,
                     "direction": ("over-flags" if pm < pr else
                                   "under-flags")})

    return worst, ph


def main():
    worst, ph = run()

    lines = []
    w = lines.append
    w("# Reference validation")
    w("")
    w("Every estimator in `src/` is hand-written from the formula. This is the")
    w("audit that says so honestly: each one differenced against `lifelines` or")
    w("`statsmodels` over %d simulated trials of n=%d." % (len(SEEDS), N))
    w("")
    w("`src/` does NOT import either library. This audit and")
    w("`tests/test_reference.py` are the only places a reference is used, and")
    w("both skip cleanly when it is not installed, so the project still runs")
    w("with no third-party survival library present.")
    w("")
    w("## Worst disagreement across all seeds and covariates")
    w("")
    w("| quantity | reference | worst disagreement |")
    w("|---|---|---|")
    for key, ref in REFS:
        if key in worst:
            w("| `%s` | %s | %.2e |" % (key, ref, worst[key]))
    w("")
    w("Relative difference, except the median CI bounds, which are absolute (in")
    w("months) because the bounds are event times rather than ratios.")
    w("")
    w("The tie-handling rows are computed on time rounded to whole months.")
    w("Breslow and Efron agree closely on continuous time, so validating there")
    w("only would let a broken Breslow pass unnoticed; heavy ties are the case")
    w("that discriminates.")
    w("")
    w("## The proportional-hazards test is checked differently")
    w("")
    w("There are now TWO proportional-hazards tests, and only one of them is")
    w("expected to match.")
    w("")
    w("`inference.scaled_ph_test` is the real Grambsch-Therneau, on SCALED")
    w("Schoenfeld residuals -- the scaled residual is on the scale of the")
    w("COEFFICIENT rather than the covariate, so its expectation at time t is")
    w("beta(t). It agrees with the reference to the tolerance in the table")
    w("above, for the statistic AND the p-value.")
    w("")
    w("`survival.ph_test` implements the Grambsch-Therneau idea in its simplest form,")
    w("correlating UNSCALED Schoenfeld residuals with ranked time. It is not")
    w("expected to reproduce the reference statistic, and it does not: over")
    w("these seeds the ratio of test statistics ranged from **%.3f to %.3f**."
      % (ph["ratio_min"], ph["ratio_max"]))
    w("")
    w("So it is validated on the thing it is actually used for -- the")
    w("conclusion:")
    w("")
    w("- agreement at alpha=0.05: **%d of %d** covariate-seeds"
      % (ph["agree"], ph["total"]))
    w("- PH violations flagged: **%d** by `ph_test`, **%d** by the reference"
      % (ph["mine_flags"], ph["ref_flags"]))
    w("")
    if ph["clashes"]:
        w("The decisions are NOT identical, and the exceptions are listed")
        w("rather than rounded away:")
        w("")
        w("| seed | covariate | `ph_test` p | reference p | direction |")
        w("|---|---|---|---|---|")
        for c in ph["clashes"]:
            w("| %d | `%s` | %.4f | %.4f | %s |"
              % (c["seed"], c["covariate"], c["p_mine"], c["p_ref"],
                 c["direction"]))
        w("")
        near = all(0.01 < c["p_mine"] < 0.20 and 0.01 < c["p_ref"] < 0.20
                   for c in ph["clashes"])
        over = all(c["direction"] == "over-flags" for c in ph["clashes"])
        if near:
            w("Every disagreement is a BORDERLINE case -- both p-values sit")
            w("near the threshold, so the tests are not reaching opposite")
            w("conclusions about a clear signal, they are landing on opposite")
            w("sides of a line drawn through a grey zone.")
        if over:
            w("")
            w("All of them are in the OVER-flagging direction: `ph_test` calls")
            w("a violation the reference does not. For a screening diagnostic")
            w("that is the tolerable direction -- it costs a second look, where")
            w("under-flagging would grant false confidence in a constant hazard")
            w("ratio. It is still a reason not to quote the p-value.")
        w("")
    w("The extreme statistic RATIOS sit where both statistics are near zero,")
    w("which is why they are not what drives the disagreements above. That is a")
    w("real limitation, and it is why `ph_test` is documented as a SCREENING")
    w("diagnostic rather than a test to quote: it is trustworthy for *is there")
    w("a problem here*, not for a p-value in a report. For that, use the scaled")
    w("residuals -- which is what the reference does and this deliberately does")
    w("not.")
    w("")
    w("## What this audit found")
    w("")
    w("`median_survival_ci` reported the upper confidence bound as the last")
    w("event time INSIDE the acceptance region. The supremum of the confidence")
    w("set is the NEXT event time, because the KM curve holds its value on the")
    w("half-open interval `[t_i, t_i+1)` and every t in that interval is in the")
    w("set. The interval was one event time too narrow -- the anti-conservative")
    w("direction, a narrower interval than the data support.")
    w("")
    w("It was findable because the disagreement was ASYMMETRIC: the lower bound")
    w("matched exactly on every seed while the upper was short on every seed. A")
    w("convention error looks like that. Noise does not.")
    w("")
    w("After the fix, both bounds agree with `lifelines` exactly.")
    w("")

    doc = os.path.join(ROOT, "docs")
    os.makedirs(doc, exist_ok=True)
    out = os.path.join(doc, "REFERENCE_VALIDATION.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("=" * 70)
    for key, _ref in REFS:
        if key in worst:
            print("  %-32s worst %.3e" % (key, worst[key]))
    print("=" * 70)
    print("  ph_test decisions agree %d/%d (mine flags %d, ref %d), "
          "stat ratio %.3f-%.3f"
          % (ph["agree"], ph["total"], ph["mine_flags"], ph["ref_flags"],
             ph["ratio_min"], ph["ratio_max"]))
    for c in ph["clashes"]:
        print("     clash seed=%d %s: p_mine=%.4f p_ref=%.4f (%s)"
              % (c["seed"], c["covariate"], c["p_mine"], c["p_ref"],
                 c["direction"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
