# DATA-3 — Trial analysis with survival methods (first 20%)

Kaplan-Meier, log-rank, Cox proportional hazards, and Schoenfeld-residual PH
testing **implemented rather than imported** (lifelines is not installed), and
verified against a simulation with a known true hazard ratio.

The centrepiece is the immortal-time-bias demonstration: the same cohort
analysed wrongly and then correctly, where treatment provably does nothing.

```bash
python run_analysis.py     # Table 1, KM, Cox, PH, immortal time -> out/*.png
python -m pytest tests -q  # 19 tests
```

---

## The four things worth reading

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

## What is missing (the other 80%)

- **No real dataset.** Everything is simulated. That is the strongest option
  for *verifying methods*, and it is worthless for demonstrating that the
  methods survive real data — no non-exponential baseline hazard, no
  competing risks, no crossover, no missing covariates, no protocol deviations.
- **Efron tie handling is not implemented.** Breslow only. With heavy ties
  Breslow biases coefficients toward zero; Efron is the better default and its
  absence is stated in the docstring rather than hidden.
- **No competing-risks analysis.** The spec offers immortal time *or* competing
  risks as the methods showcase and this build chose immortal time. Fine-Gray
  and cause-specific hazards — and the fact that they answer different
  questions — are absent.
- **No RMST**, despite the code recommending it as a PH remedy.
- **No formal CONSORT diagram**, no participant flow, no protocol document.
- **No multiplicity handling**, no interim analysis, no alpha spending.
- **The PH test is a simplified Grambsch-Therneau** — correlation of
  unscaled Schoenfeld residuals with ranked time, not the scaled residuals with
  the full variance-covariance treatment.
- **Stratified Cox pools per-stratum estimates by inverse variance** rather than
  maximising a single stratified partial likelihood. Close for well-balanced
  strata, not identical.
- **No bootstrap or robust standard errors**, and no clustering.

## Files

| path | what |
|---|---|
| `src/survival.py` | KM, Greenwood, at-risk, log-rank, Cox (Breslow), Schoenfeld, PH test, stratified Cox |
| `src/simulate.py` | trial simulation with known HR; immortal-time cohort; counting-process split |
| `run_analysis.py` | Table 1, censoring, KM, Cox, PH, immortal time, pre-specified vs exploratory |
| `out/km_curve.png` | KM with number-at-risk table and Greenwood bands |
| `out/immortal_time.png` | the wrong-then-right figure pair |
| `tests/test_survival.py` | 19 tests |
