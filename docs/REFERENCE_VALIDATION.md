# Reference validation

Every estimator in `src/` is hand-written from the formula. This is the
audit that says so honestly: each one differenced against `lifelines` or
`statsmodels` over 10 simulated trials of n=500.

`src/` does NOT import either library. This audit and
`tests/test_reference.py` are the only places a reference is used, and
both skip cleanly when it is not installed, so the project still runs
with no third-party survival library present.

## Worst disagreement across all seeds and covariates

| quantity | reference | worst disagreement |
|---|---|---|
| `kaplan_meier S(t)` | lifelines KaplanMeierFitter | 3.83e-15 |
| `logrank chi2` | lifelines logrank_test | 9.22e-15 |
| `logrank p` | lifelines logrank_test | 9.12e-14 |
| `cox efron beta` | lifelines CoxPHFitter | 8.24e-06 |
| `cox efron se` | lifelines CoxPHFitter | 5.48e-07 |
| `cox breslow beta (heavy ties)` | statsmodels PHReg(ties=breslow) | 3.65e-15 |
| `cox efron beta (heavy ties)` | statsmodels PHReg(ties=efron) | 3.68e-15 |
| `robust se` | lifelines CoxPHFitter(robust=True) | 1.78e-06 |
| `clustered se` | lifelines CoxPHFitter(cluster_col=) | 1.25e-06 |
| `median point` | lifelines median_survival_time_ | 0.00e+00 |
| `median CI lower` | lifelines median_survival_times | 0.00e+00 |
| `median CI upper` | lifelines median_survival_times | 0.00e+00 |
| `scaled PH chi2` | lifelines proportional_hazard_test | 5.09e-06 |
| `scaled PH p` | lifelines proportional_hazard_test | 5.12e-06 |
| `chi2 survival function` | scipy.stats.chi2.sf | 2.72e-14 |

Relative difference, except the median CI bounds, which are absolute (in
months) because the bounds are event times rather than ratios.

The tie-handling rows are computed on time rounded to whole months.
Breslow and Efron agree closely on continuous time, so validating there
only would let a broken Breslow pass unnoticed; heavy ties are the case
that discriminates.

## The proportional-hazards test is checked differently

There are now TWO proportional-hazards tests, and only one of them is
expected to match.

`inference.scaled_ph_test` is the real Grambsch-Therneau, on SCALED
Schoenfeld residuals -- the scaled residual is on the scale of the
COEFFICIENT rather than the covariate, so its expectation at time t is
beta(t). It agrees with the reference to the tolerance in the table
above, for the statistic AND the p-value.

`survival.ph_test` implements the Grambsch-Therneau idea in its simplest form,
correlating UNSCALED Schoenfeld residuals with ranked time. It is not
expected to reproduce the reference statistic, and it does not: over
these seeds the ratio of test statistics ranged from **0.070 to 2.654**.

So it is validated on the thing it is actually used for -- the
conclusion:

- agreement at alpha=0.05: **29 of 30** covariate-seeds
- PH violations flagged: **3** by `ph_test`, **2** by the reference

The decisions are NOT identical, and the exceptions are listed
rather than rounded away:

| seed | covariate | `ph_test` p | reference p | direction |
|---|---|---|---|---|
| 109 | `age10` | 0.0374 | 0.0564 | over-flags |

Every disagreement is a BORDERLINE case -- both p-values sit
near the threshold, so the tests are not reaching opposite
conclusions about a clear signal, they are landing on opposite
sides of a line drawn through a grey zone.

All of them are in the OVER-flagging direction: `ph_test` calls
a violation the reference does not. For a screening diagnostic
that is the tolerable direction -- it costs a second look, where
under-flagging would grant false confidence in a constant hazard
ratio. It is still a reason not to quote the p-value.

The extreme statistic RATIOS sit where both statistics are near zero,
which is why they are not what drives the disagreements above. That is a
real limitation, and it is why `ph_test` is documented as a SCREENING
diagnostic rather than a test to quote: it is trustworthy for *is there
a problem here*, not for a p-value in a report. For that, use the scaled
residuals -- which is what the reference does and this deliberately does
not.

## What this audit found

`median_survival_ci` reported the upper confidence bound as the last
event time INSIDE the acceptance region. The supremum of the confidence
set is the NEXT event time, because the KM curve holds its value on the
half-open interval `[t_i, t_i+1)` and every t in that interval is in the
set. The interval was one event time too narrow -- the anti-conservative
direction, a narrower interval than the data support.

It was findable because the disagreement was ASYMMETRIC: the lower bound
matched exactly on every seed while the upper was short on every seed. A
convention error looks like that. Noise does not.

After the fix, both bounds agree with `lifelines` exactly.

