# DATA-3 — Trial analysis with survival methods — working system, 8 known gaps

Kaplan-Meier, log-rank, Cox proportional hazards, and Schoenfeld-residual PH
testing **implemented rather than imported** — by choice, to show the
mechanics — verified against a simulation with a known true hazard ratio and
then **differenced against `lifelines` and `statsmodels`** to prove the
arithmetic is right rather than merely self-consistent.

The centrepiece is the immortal-time-bias demonstration: the same cohort
analysed wrongly and then correctly, where treatment provably does nothing.

```bash
python run_analysis.py     # Table 1, KM, Cox, PH, immortal time -> out/*.png
python run_ties.py --reps 60   # Efron vs Breslow against a planted HR
python write_report.py         # -> docs/ANALYSIS_REPORT.md
python -m pytest tests -q      # 86 tests
python validate_reference.py   # audit vs lifelines/statsmodels -> docs/
```

---

## The six things worth reading

### 1. Immortal time bias, wrong then right

A cohort of 1,200 where treatment is initiated at a **random time after
baseline** and has **no effect whatsoever** — `simulate_immortal_time()` does
not put a treatment term in the hazard at all. True HR = 1.00 by construction.

| analysis | HR | 95% CI | p |
|---|---|---|---|
| **[WRONG]** group by "ever treated", follow from baseline | **0.366** | 0.32–0.43 | 4×10⁻³⁹ |
| **[RIGHT]** time-varying exposure on (start, stop] intervals | **1.069** | 0.91–1.26 | 0.42 |
| truth | 1.00 | | |

The naive analysis reads as a **63% reduction in the hazard of death**, with a
p-value of 4×10⁻³⁹. It is entirely an artefact.

**Where it comes from:** 3,722 person-months elapsed between baseline and
treatment initiation across the treated patients. Every one of those months is
guaranteed event-free — a patient who died in month 2 could never appear in the
"treated at month 6" group. The naive grouping credits all of that immortal
time to the treatment. The counting-process split assigns it to the *untreated*
state, where it belongs, and the effect vanishes.

*Explaining it to a PM in three sentences:* To get the treatment, you first had
to survive long enough to receive it. Grouping people by whether they were ever
treated therefore compares "people who lived at least a while" against
everybody, so the treated group looks healthier before the drug does anything.
The fix is to count each patient as untreated until the day they are treated,
and treated afterwards.

### 2. The estimators are verified against known truth

This is the only reason a hand-rolled Cox model should be believed.

| covariate | estimated HR | 95% CI | true HR | covered |
|---|---|---|---|---|
| treatment | 0.743 | 0.64–0.86 | 0.70 | ✓ |
| age (per decade) | 1.232 | 1.15–1.32 | 1.25 | ✓ |
| advanced stage | 2.069 | 1.77–2.41 | 1.80 | ✓ |

`test_cox_coverage_across_many_replications` goes further: a single interval
containing the truth could be luck, so it checks that ~95% of intervals cover
it across 30 replications.

**Language is kept precise throughout.** HR 0.74 means that at any instant,
among patients still alive and in follow-up, the treatment arm is experiencing
deaths at roughly 74% of the rate of the control arm, averaged over follow-up
under proportional hazards. It is *not* a risk ratio and *not* "26% less likely
to die" — how many people are alive at four years depends on the baseline
hazard too, and the HR alone does not tell you.

### 3. The details a clinical reviewer checks first

- **Number-at-risk table** under every KM curve (`out/km_curve.png`).
  Non-negotiable: without it a reader cannot tell whether the flat tail at 20%
  survival rests on 200 patients or 3, and those imply completely different
  conclusions from an identical-looking picture.
- **Greenwood confidence bands**, so the tail is visibly less certain than the
  start.
- **Median follow-up by reverse Kaplan-Meier: 39.3 months** — against a median
  *observed* time of 10.0 months, which is the number people quote and is
  biased downward, because patients who had the event stop contributing
  follow-up. The gap here is nearly fourfold.
- **"Median not reached"** is reported as such and never extrapolated. Both
  medians were reached in this run, and the output says what it would report
  instead: a landmark estimate plus the follow-up duration.
- **Censoring mechanism described**, with the direction of the bias named:
  administrative censoring is benign by construction; if *sicker* patients drop
  out, censoring is informative and KM is biased **optimistic**, because the
  people removed from the risk set were the ones about to have events.

### 4. The PH assumption is tested, and the options priced

Schoenfeld residuals correlated against ranked time:

| covariate | rho | chi² | p |
|---|---|---|---|
| treatment | 0.001 | 0.00 | 0.985 |
| age | 0.046 | 1.43 | 0.231 |
| stage | −0.007 | 0.04 | 0.846 |

No violation detected — and the output says plainly that this **is not proof of
proportionality**, only a failure to detect a departure with 691 events, by a
test with limited power.

The interesting content is what happens when it *does* fail, and the code
handles that branch. Each remedy changes what can be claimed:

1. **Stratify** — each stratum gets its own baseline hazard. Cost: the
   stratifying variable no longer has a coefficient, so you have bought
   correctness for the others by giving up any statement about that one.
2. **Time-varying coefficient** — cost: there is no single HR any more. The
   answer becomes "HR was X early and Y late", which is more honest and much
   harder to put in an abstract.
3. **Restricted mean survival time** — needs no PH assumption at all, and is in
   months, which clinicians read more easily than a ratio.

`test_ph_test_flags_a_deliberately_time_varying_effect` constructs an effect
that reverses over time and asserts the diagnostic catches it — otherwise the
test above is decoration.

### 5. Competing risks: where 1 − KM goes wrong, and by how much

The spec offers competing risks *or* immortal time as the methods showcase. The
first build chose immortal time; this adds the other.

Treating a competing event as censored and reading 1 − KM as cumulative
incidence is not a subtle error. Censoring assumes the removed patient could
still have the event later — but a patient who died of something else cannot
later die of this. The risk gets redistributed to the survivors, so the estimate
is **always biased upward, and the bias grows with follow-up**:

| horizon | Aalen-Johansen CIF | 1 − KM (wrong) | absolute | relative |
|---|---|---|---|---|
| 12 mo | 0.212 | 0.254 | +0.042 | +19.7% |
| 24 mo | 0.330 | 0.454 | +0.124 | +37.4% |
| 36 mo | 0.389 | 0.590 | +0.201 | +51.7% |
| **48 mo** | **0.412** | **0.662** | **+0.250** | **+60.8%** |

At 48 months the naive estimator claims 66.2% of patients have had the event
when the truth is 41.2%. See `out/competing_risks.png` — the shaded gap *is* the
error.

The competing hazard here is deliberately **larger** than the event of interest,
which is the realistic case in an older population and precisely when this goes
most wrong.

**Two regression models, two different questions** — people argue about these as
though one is right:

| model | HR (treatment) | 95% CI |
|---|---|---|
| cause-specific (aetiological) | 0.675 | 0.58–0.79 |
| Fine-Gray (subdistribution) | 0.711 | 0.61–0.83 |
| *planted cause-specific truth* | *0.600* | *covered ✓* |

The Fine-Gray HR is **attenuated toward 1**, and that is not an error in either
model. Cause-specific asks *"among patients still at risk, does treatment change
the rate?"*. Fine-Gray asks *"does treatment change the probability of ever
having had this event by time t?"*. The treatment lowers the rate — but by
keeping patients alive it leaves them at risk longer, which partly offsets the
effect on cumulative incidence. `test_fine_gray_is_attenuated_toward_one` pins
the relationship.

Which to report follows from the question: **mechanism → cause-specific;
absolute risk, resource allocation, or what to tell a patient → Fine-Gray.**
Reporting one and interpreting it as the other is how a treatment that changes
nobody's actual chance of the event gets described as reducing it by a third.

### 6. RMST — no PH assumption, and in units a patient understands

The first build recommended restricted mean survival time as the remedy when PH
fails and did not implement it.

```
tau = 36 months (PRE-SPECIFIED)
  treatment RMST   16.93 months
  control RMST     14.24 months
  difference       +2.68 months (+1.11, +4.22)
```

*"Over the first three years, a patient on treatment lives on average 2.7 months
longer."* That sentence needs no proportional-hazards assumption and no
explanation of what a hazard is.

τ must be **pre-specified** — different horizons give different answers, so
choosing it after seeing the curves lets the analyst pick the flattering one.

### Plus: pre-specified vs exploratory

The writeup separates them, and the separation is *conditional on what actually
happened in the run*: the stratified model appears under "exploratory" only if
the PH test failed and it was therefore fitted in response to the data. A model
chosen after seeing the data is exploratory however well motivated the choice
was, and a p-value computed afterwards does not mean what a p-value means.

---

## A bug this test suite caught

`test_km_censoring_removes_from_risk_set_without_dropping_survival` failed on
first run — and **the test was wrong, not the estimator**. With times 1 (event),
2 (censored), 3 (event), I had expected survival to end at 1/3, having kept the
censored subject in the risk set at t=3. It correctly ends at 0: only one
subject is still at risk and they die. The test now carries the arithmetic in a
comment, and the point it illustrates — that a KM value of 0% resting on one
patient is not the same statement as 0% resting on two hundred — is the reason
the at-risk table is mandatory.

## Efron tie handling, and the claim it replaced

The gap list said: *"Efron tie handling is not implemented. Breslow only. With
heavy ties Breslow biases coefficients toward zero."* The docstring justified
Breslow on the grounds that "the difference is negligible unless ties are
heavy". That is the received wisdom, it is probably true, **and it had never
been checked.**

Efron is now implemented and is the default. `run_ties.py` checks the claim by
simulating from a planted hazard ratio and then rounding the *same* event times
onto coarser grids — which is not an artificial manipulation, since trial data
arrives on a day grid and registry and claims data arrive on a month grid.

| grid | % events tied | largest tie | Breslow HR | Efron HR |
|---|---|---|---|---|
| continuous | 0.0% | 1 | 0.7346 | 0.7346 |
| daily | 29.7% | 3 | 0.7348 | 0.7346 |
| weekly | 83.5% | 9 | 0.7357 | 0.7344 |
| monthly | 98.2% | 30 | 0.7405 | 0.7347 |
| quarterly | 99.7% | 83 | **0.7524** | **0.7347** |

With no ties the two agree to `0.00e+00` — they are the same expression when
every event time is unique, which is the first thing to confirm before
believing any difference elsewhere.

**The attenuation is measured against the tie-free estimate on the same data,
not against the planted HR.** Both estimators sit near 0.735 against a planted
0.70 even with no ties; that is finite-sample bias at n=400 and has nothing to
do with tie handling. The first version of this report quoted
distance-from-truth and so silently attributed that bias to Breslow. Isolated
properly:

| grid | Breslow drift | Efron drift |
|---|---|---|
| monthly | +0.0059 | +0.0001 |
| quarterly | **+0.0179** | **+0.0001** |

**Efron is very nearly invariant to the grid**, which is the property worth
having: an estimate should depend on the data, not on how coarsely somebody
recorded the dates.

The direction is what matters. Breslow uses the full risk set for every death
in a tied set, double-counting subjects who have already failed, inflating the
denominator and biasing toward the null. In a trial, attenuation understates a
treatment benefit — conservative — and understates a **harm** signal by exactly
as much, which is not conservative at all.

## CONSORT flow, and why a simulated one is almost a lie

`src/consort.py`. The CONSORT diagram exists to make participant loss visible,
and its value is entirely in the numbers it is uncomfortable to publish. **None
of that is observed here** — every attrition figure is one I chose, so a
generated diagram cannot demonstrate that a trial retained its participants.

What does generalise is the invariant: *every randomised participant appears in
exactly one terminal state, and the arithmetic closes at every stage.*
`validate()` refuses to emit a diagram that fails it, and each check
corresponds to a way real trial reports break rather than a way this generator
might:

- **allocation not summing to randomised** — someone randomised and then absent
  from the diagram entirely, the most serious failure because nothing
  downstream shows it;
- **ITT larger than allocated** — an analysis population that grew after
  randomisation, usually from a post-unblinding reclassification;
- **per-protocol larger than ITT** — impossible by construction, and a sign two
  denominators have been mixed;
- **negative completion** — a participant counted in two discontinuation
  reasons.

The ITT/per-protocol distinction is stated where it is most often misstated:
per-protocol is **not** "the cleaner analysis", it is the biased one. Adherence
is a post-randomisation outcome, so conditioning on it reintroduces exactly the
confounding randomisation was there to remove.

## A generated analysis report

`write_report.py` → [`docs/ANALYSIS_REPORT.md`](docs/ANALYSIS_REPORT.md). Every
number is computed at build time; none is typed. A report with hand-entered
numbers disagrees with the code after the first re-run, and in a trial context
a stale number is not a typo — it is the thing regulators and journals read.

Structured as a statistical analysis plan, because the ordering carries
meaning: analysis population before result, assumption check before the
estimate that depends on it, limitations attached to the estimate rather than
collected in a footnote. It also **reports its own gap** where one exists — the
median survival has no confidence interval, and the report says so and explains
why (a Brookmeyer–Crowley inversion of the Greenwood variance, not implemented)
rather than quietly presenting a point estimate.

## Median intervals, robust variance, and the exact tie likelihood

`src/inference.py` closes four named gaps.

### A median needs a different interval from everything else

```
median survival (months) | 13.8 (12.0-15.1) | 10.2 (8.9-11.8)
```

A Kaplan-Meier median is a quantile of a **step function**. Its sampling
distribution is not symmetric, so `+/- 1.96 SE` is the wrong *shape*, not merely
imprecise. **Brookmeyer-Crowley** inverts the test instead: the confidence set
is every time at which S(t)=0.5 would not be rejected.

Built on **log(-log S)** rather than S directly, because a plain-scale interval
can extend past 0 or 1 - and an interval containing "survival probability 1.04"
tells the reader the method was wrong rather than the data.

An upper bound of `None` means the set is **open above**, which is the honest
answer when the curve never falls far enough to reject later. Reporting the
largest observed time instead would present a made-up number as an estimate.

### Clustering only helps when the covariate is cluster-level

The sandwich estimator is implemented, and getting the *test fixture* right
took three attempts - which turned out to be the finding:

| design | clustered / plain SE |
|---|---|
| covariate shared in cluster, t ~ x | 1.12 |
| covariate **individual** + strong omitted frailty | 0.98 |
| covariate **cluster-level** + omitted frailty | **2.66** |

The covariate has to be **cluster-level** - a site effect, a centre-level
exposure - which is exactly the multi-centre trial case the correction exists
for. With an individual-level covariate the within-cluster score residuals do
not line up and clustering buys nothing; across five seeds the ratio ran 0.70 to
1.28, which is noise rather than a correction. Both claims are tested across
seeds, because a single draw would have asserted a stable relationship that does
not exist.

Under positive within-cluster correlation the model-based SE is **always too
small**, and that error is anti-conservative: narrower intervals and more
significance than the data support, which is the direction that gets a finding
published and then fails to replicate.

### The exact tie likelihood, so "Efron approximates it" is checkable

`exact_tie_loglik()` enumerates every ordering of a tied set - the quantity
Efron approximates and Breslow ignores. On a 3-tie:

```
exact    -1.752      efron    -3.571      breslow  -4.470
                     error 1.818          error 2.718
```

Efron is closer, and Breslow understates the contribution - the direction that
attenuates coefficients toward the null. It is **factorial in the tie size** and
refuses above 8 tied deaths, which is the entire reason Efron exists.

## Validated against a reference implementation

Every estimator here is written from the formula. That is the point of the
project, and it is also a correctness claim that only a reference can settle:
**a unit test written by the person who wrote the estimator shares its
misconceptions.** The existing tests plant a hazard ratio and check it is
recovered, which catches a wrong answer but not a subtly wrong *convention*.

`validate_reference.py` differences all of it against `lifelines` and
`statsmodels` over 10 simulated trials:

| quantity | reference | worst disagreement |
|---|---|---|
| `kaplan_meier` S(t) | lifelines `KaplanMeierFitter` | 3.8e-15 |
| log-rank chi2 / p | lifelines `logrank_test` | 9.2e-15 / 9.1e-14 |
| Cox Efron beta / se | lifelines `CoxPHFitter` | 8.2e-06 / 5.5e-07 |
| Cox Breslow beta, heavy ties | statsmodels `PHReg(ties=breslow)` | 3.7e-15 |
| Cox Efron beta, heavy ties | statsmodels `PHReg(ties=efron)` | 3.7e-15 |
| robust sandwich se | lifelines `CoxPHFitter(robust=True)` | 1.8e-06 |
| **clustered** sandwich se | lifelines `CoxPHFitter(cluster_col=)` | 1.2e-06 |
| median + Brookmeyer-Crowley bounds | lifelines `median_survival_times` | **0** |

The tie rows are computed on time **rounded to whole months**. Breslow and
Efron agree closely on continuous time, so validating only there would let a
broken Breslow pass unnoticed — heavy ties are the case that discriminates,
and are also the realistic one, since trials record time in days.

### It found a bug, in the anti-conservative direction

`median_survival_ci` reported the upper confidence bound as the **last event
time inside** the acceptance region. The supremum of the confidence set is the
**next** one: the KM curve holds its value on `[t_i, t_i+1)`, so every t in
that half-open interval is in the set. The interval was one event time too
narrow — narrower than the data support, which is the direction that
manufactures confidence.

It was findable because the disagreement was **asymmetric**: the lower bound
matched exactly on every seed while the upper was short on every seed. A
convention error looks like that. Noise does not. Both bounds now agree
exactly, and `tests/test_reference.py` pins the convention in words as well as
in numbers, so it cannot be "fixed" back by someone reading only the number.

### The PH test is audited on its decisions, not its arithmetic

`ph_test` is deliberately the Grambsch-Therneau idea in its *simplest* form —
unscaled Schoenfeld residuals against ranked time — so it is not expected to
reproduce the reference statistic, and does not: the ratio of test statistics
ranged **0.070 to 2.654**. What matters is whether it reaches the same
conclusion, and it agreed on **29 of 30** covariate-seeds.

The one exception is reported rather than rounded away: seed 109, `age10`,
p=0.0374 here against p=0.0564 in the reference. Both straddle 0.05, so the
tests are not disagreeing about a clear signal — they are landing on opposite
sides of a line drawn through a grey zone. It is in the **over-flagging**
direction, which for a screening diagnostic is the tolerable one: it costs a
second look, where under-flagging would grant false confidence in a constant
hazard ratio. It remains a reason not to quote the p-value.

**`src/` does not import either library.** The audit and its tests are the only
places a reference appears, and both skip cleanly when it is absent, so the
project still runs with no third-party survival library installed.

## The proportional-hazards test, done properly

There are now **two** PH tests here, and only one of them is meant to match a
reference.

`survival.ph_test` correlates **unscaled** Schoenfeld residuals with ranked
time. It gets the direction right, and the reference audit measured exactly how
far that falls short: it agreed with `lifelines` on 29 of 30 decisions while its
statistic ranged **0.07 to 2.65 times** the reference's. A screen, not a
quotable test.

`inference.scaled_ph_test` is the real Grambsch-Therneau:

| quantity | worst disagreement vs `lifelines` |
|---|---|
| scaled PH chi² | **5.1e-06** |
| scaled PH p-value | **5.1e-06** |

### Why scaling is not cosmetic

A raw Schoenfeld residual is on the scale of the **covariate** and centres on
zero — it carries no information about the size of the coefficient. The scaled
residual

```
s* = β + d · V · s
```

is on the scale of the **coefficient**, so its expectation at time *t* is
literally β(*t*). That is what makes the plot readable and the slope
interpretable in log-hazard units. Two tests pin the contrast: the scaled
residuals average to β, the unscaled ones average to zero.

### Two things it adds beyond matching

**A global test.** Testing three covariates separately at 0.05 and reporting
the smallest p is a multiple-comparison problem, and it is the usual way a PH
violation gets "found". The global test is a quadratic form on *p* degrees of
freedom.

**The time transform is a parameter, and it changes the answer.** A test
against raw time is dominated by the longest follow-up, where the risk set is
smallest and the residuals noisiest. Quoting a PH p-value without naming the
transform is not quoting anything, so `rank`, `identity` and `log` are all
available and an unknown one is refused rather than defaulted.

### The chi-square tail is hand-rolled too

`src/` has no scipy dependency — the reference libraries **audit** this code,
they never provide it. So the upper tail of the chi-square is written out as a
series and continued fraction, which is a claim about numerics and is checked
like one: **2.7e-14** against `scipy.stats.chi2.sf`.

A test also asserts that `ph_test` **still does not match**. If it ever starts
agreeing exactly, somebody has quietly replaced it with the scaled version and
the documented distinction between a screen and a test has stopped being true.

## What is still missing, and why it cannot be closed here

- **No real dataset.** Everything is simulated, which is the strongest option
  for *verifying methods* and worthless for showing they survive real data -
  no non-exponential baseline hazard, no crossover, no missing covariates, no
  protocol deviations. Closing it needs a dataset that cannot be shipped in a
  repository.
- **The CONSORT attrition is chosen, not observed.** No screening log, no
  eligibility criteria, no site structure, no randomisation schedule or
  allocation concealment, no blinding, no protocol document, no adverse-event
  accounting. The *invariant* is enforced and is the part that generalises.
- **No site structure to cluster on.** The clustered variance is implemented
  and tested, and the primary analysis does not use it because this trial has
  one site. Applying it anyway would report a correction for a correlation
  that does not exist.
- **The exact tie likelihood is not used in fitting**, only for comparison. It
  is factorial in the tie size, which makes it impractical on any real event
  time distribution - Efron exists for this reason and is the default.
- **Fine-Gray uses a simplified IPCW scheme** - right-censoring-complete
  weights, no left truncation, no time-varying weights. R's `cmprsk` handles
  those and is not available offline.
- **The scaled PH test covers one transform family and no strata.**
  `scaled_ph_test` is the real Grambsch-Therneau and matches the reference (see
  above), but it offers `rank`/`identity`/`log` and not the Kaplan-Meier
  transform, and it has no stratified variant — a violation confined to one
  stratum can hide in a pooled test.
- **Stratified Cox pools per-stratum estimates by inverse variance** rather
  than maximising a single stratified partial likelihood. Close for balanced
  strata, not identical.
- **No multiplicity handling, no interim analysis, no alpha spending.** The
  p-values are nominal, and this trial has no interim looks to spend alpha on.

## Files

| path | what |
|---|---|
| `src/survival.py` | KM, Greenwood, at-risk, log-rank, Cox (Breslow), Schoenfeld, PH test, stratified Cox |
| `src/simulate.py` | trial simulation with known HR; immortal-time cohort; counting-process split |
| `run_analysis.py` | Table 1, censoring, KM, Cox, PH, immortal time, pre-specified vs exploratory |
| `out/km_curve.png` | KM with number-at-risk table and Greenwood bands |
| `src/competing_risks.py` | Aalen-Johansen CIF, cause-specific and Fine-Gray Cox, RMST |
| `out/immortal_time.png` | the wrong-then-right figure pair |
| `out/competing_risks.png` | the shaded gap between CIF and 1 − KM |
| `src/consort.py` | participant flow, and the accounting invariant it enforces |
| `run_ties.py` | Efron vs Breslow across event-time grids |
| `write_report.py` | the generated statistical report |
| `src/inference.py` | Brookmeyer-Crowley medians, sandwich variance, exact ties |
| `validate_reference.py` | audit vs lifelines/statsmodels; found the median-CI bug |
| `tests/test_scaled_ph.py` | 13 tests: scaling, global test, chi2 tail |
| `tests/test_reference.py` | 10 tests: skip cleanly when no reference is installed |
| `tests/test_inference.py` | 18 tests: open bounds, when clustering helps, exact vs Efron |
| `tests/test_ties_consort.py` | 16 tests: tie recovery, and six ways a CONSORT flow breaks |
| `tests/test_survival.py` | 29 tests |
