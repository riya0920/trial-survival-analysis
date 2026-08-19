"""Tests for the hand-rolled estimators.

A hand-rolled Cox model should not be believed on the strength of it running.
These tests check it against closed-form answers where one exists, against
known simulation truth where one does not, and against the specific behaviours
that are easy to get wrong: tie handling, the risk set, and the direction of
the immortal-time bias.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import simulate
import survival as S


# ---------------------------------------------------------------------------
# Kaplan-Meier against hand-computable cases
# ---------------------------------------------------------------------------
def test_km_with_no_censoring_is_the_empirical_survival():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 1, 1])
    t, s, ar, d, _se = S.kaplan_meier(time, event)
    assert list(t) == [1.0, 2.0, 3.0, 4.0]
    assert s == pytest.approx([0.75, 0.5, 0.25, 0.0])
    assert list(ar) == [4, 3, 2, 1]
    assert list(d) == [1, 1, 1, 1]


def test_km_censoring_removes_from_risk_set_without_dropping_survival():
    """A censored observation should reduce the number at risk and NOT cause a
    step down. Getting this wrong makes censoring look like events."""
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 0, 1])
    t, s, ar, _d, _se = S.kaplan_meier(time, event)
    # t=1: 3 at risk, 1 event      -> S = 1 - 1/3 = 2/3
    # t=2: censored, no step, but that subject leaves the risk set
    # t=3: only 1 subject still at risk and they have the event -> S = 0
    #
    # The estimator drops to 0 here and that is correct: the last person at
    # risk died. It is also why a KM tail must never be read without the
    # number-at-risk table -- "survival 0%" resting on a single patient is
    # not the same statement as "survival 0%" resting on two hundred.
    assert list(t) == [1.0, 3.0]
    assert s == pytest.approx([2 / 3, 0.0])
    assert list(ar) == [3, 1]


def test_km_handles_ties():
    time = np.array([2.0, 2.0, 5.0])
    event = np.array([1, 1, 1])
    t, s, ar, d, _se = S.kaplan_meier(time, event)
    assert list(t) == [2.0, 5.0]
    assert list(d) == [2, 1]
    assert s == pytest.approx([1 / 3, 0.0])


def test_median_survival_returns_none_when_not_reached():
    """'Not reached' must be reported as such. Returning the last observed
    time instead is how a survival analysis starts lying."""
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 0, 0, 0])
    t, s, _a, _d, _se = S.kaplan_meier(time, event)
    assert S.median_survival(t, s) is None


def test_at_risk_counts_are_correct():
    time = np.array([1.0, 5.0, 9.0, 12.0])
    assert S.at_risk_table(time, np.ones(4), [0, 6, 10]) == [4, 2, 1]


# ---------------------------------------------------------------------------
# Reverse KM
# ---------------------------------------------------------------------------
def test_reverse_km_follow_up_exceeds_median_observed_time():
    """The whole point of reverse KM. Median observed time is biased downward
    because patients who had the event stop contributing follow-up."""
    d = simulate.simulate_trial(n=800, seed=3)
    mfu = S.median_followup_reverse_km(d["time"], d["event"])
    assert mfu > np.median(d["time"])


# ---------------------------------------------------------------------------
# Log-rank
# ---------------------------------------------------------------------------
def test_logrank_finds_no_difference_between_identical_groups():
    rng = np.random.default_rng(1)
    time = rng.exponential(10, 600)
    event = np.ones(600, dtype=int)
    group = np.concatenate([np.zeros(300, int), np.ones(300, int)])
    chi2, p, _o, _e = S.logrank(time, event, group)
    assert p > 0.05


def test_logrank_detects_a_large_difference():
    rng = np.random.default_rng(2)
    time = np.concatenate([rng.exponential(5, 300), rng.exponential(20, 300)])
    event = np.ones(600, dtype=int)
    group = np.concatenate([np.zeros(300, int), np.ones(300, int)])
    chi2, p, _o, _e = S.logrank(time, event, group)
    assert p < 1e-6


# ---------------------------------------------------------------------------
# Cox
# ---------------------------------------------------------------------------
def test_cox_recovers_the_known_hazard_ratio():
    d = simulate.simulate_trial(n=2500, seed=11)
    X = np.column_stack([d["arm"], (d["age"] - 62) / 10, d["stage"]])
    fit = S.cox_ph(d["time"], d["event"], X)
    truth = [d["truth"]["hr_treatment"], d["truth"]["hr_age_per_decade"],
             d["truth"]["hr_stage"]]
    for i, tr in enumerate(truth):
        assert fit["ci_low"][i] <= tr <= fit["ci_high"][i], (
            f"covariate {i}: true HR {tr} outside "
            f"({fit['ci_low'][i]:.3f}, {fit['ci_high'][i]:.3f})")


def test_cox_coverage_across_many_replications():
    """A single interval containing the truth could be luck. Nominal 95%
    coverage across replications is the property that matters."""
    covered = 0
    reps = 30
    for seed in range(reps):
        d = simulate.simulate_trial(n=700, seed=100 + seed)
        X = np.column_stack([d["arm"], (d["age"] - 62) / 10, d["stage"]])
        fit = S.cox_ph(d["time"], d["event"], X)
        if fit["ci_low"][0] <= d["truth"]["hr_treatment"] <= fit["ci_high"][0]:
            covered += 1
    assert covered >= reps - 6, f"only {covered}/{reps} intervals covered truth"


def test_cox_with_no_effect_gives_a_hazard_ratio_near_one():
    rng = np.random.default_rng(7)
    n = 2000
    x = rng.integers(0, 2, n)
    time = rng.exponential(10, n)          # x does not enter
    event = np.ones(n, dtype=int)
    fit = S.cox_ph(time, event, x.reshape(-1, 1))
    assert fit["ci_low"][0] <= 1.0 <= fit["ci_high"][0]


def test_cox_direction_matches_the_log_rank():
    d = simulate.simulate_trial(n=1200, seed=13)
    fit = S.cox_ph(d["time"], d["event"], d["arm"].reshape(-1, 1))
    _chi2, p, o1, e1 = S.logrank(d["time"], d["event"], d["arm"])
    assert (fit["hr"][0] < 1) == (o1 < e1), (
        "Cox and log-rank disagree on which arm did better")


# ---------------------------------------------------------------------------
# PH diagnostics
# ---------------------------------------------------------------------------
def test_ph_test_does_not_flag_proportional_data():
    d = simulate.simulate_trial(n=1500, seed=17)
    X = d["arm"].reshape(-1, 1)
    fit = S.cox_ph(d["time"], d["event"], X)
    ph = S.ph_test(d["time"], d["event"], X, fit["beta"])
    assert ph["p"] > 0.01


def test_ph_test_flags_a_deliberately_time_varying_effect():
    """Construct an effect that reverses over time. If the diagnostic cannot
    detect that, it is decoration."""
    rng = np.random.default_rng(23)
    n = 3000
    arm = rng.integers(0, 2, n)
    # treated: strongly protected early, harmed late
    base = rng.exponential(12, n)
    time = np.where(arm == 1, np.where(base < 12, base * 3.2, base * 0.30), base)
    event = np.ones(n, dtype=int)
    X = arm.reshape(-1, 1)
    fit = S.cox_ph(time, event, X)
    ph = S.ph_test(time, event, X, fit["beta"])
    assert ph["p"] < 0.01, f"PH test failed to detect a reversing effect (p={ph['p']})"


# ---------------------------------------------------------------------------
# Immortal time bias -- the centrepiece
# ---------------------------------------------------------------------------
def test_naive_ever_treated_analysis_manufactures_a_large_benefit():
    z = simulate.simulate_immortal_time(n=1500, seed=41)
    X = np.column_stack([z["ever_treated"], (z["age"] - 64) / 10, z["stage"]])
    naive = S.cox_ph(z["time"], z["event"], X)
    assert naive["hr"][0] < 0.6, (
        "the naive analysis should show a large spurious benefit; if it does "
        "not, the immortal-time demonstration has stopped demonstrating")
    assert naive["ci_high"][0] < 1.0, "and it should look convincingly significant"


def test_time_varying_analysis_recovers_the_null_truth():
    z = simulate.simulate_immortal_time(n=1500, seed=41)
    cp = simulate.to_counting_process(z)
    tv = simulate.cox_counting_process(cp)
    assert tv["ci_low"][0] <= z["truth"]["hr_treatment"] <= tv["ci_high"][0], (
        f"true HR 1.00 outside ({tv['ci_low'][0]:.3f}, {tv['ci_high'][0]:.3f})")


def test_the_correction_moves_the_estimate_toward_the_truth():
    z = simulate.simulate_immortal_time(n=1500, seed=41)
    X = np.column_stack([z["ever_treated"], (z["age"] - 64) / 10, z["stage"]])
    naive = S.cox_ph(z["time"], z["event"], X)
    tv = simulate.cox_counting_process(simulate.to_counting_process(z))
    truth = z["truth"]["hr_treatment"]
    assert abs(tv["hr"][0] - truth) < abs(naive["hr"][0] - truth)


def test_counting_process_split_conserves_person_time():
    """Splitting must not create or destroy follow-up. If it does, every
    downstream rate is wrong."""
    z = simulate.simulate_immortal_time(n=400, seed=43)
    cp = simulate.to_counting_process(z)
    assert (cp["stop"] - cp["start"]).sum() == pytest.approx(z["time"].sum())
    assert cp["event"].sum() == z["event"].sum()


def test_nobody_contributes_treated_time_before_being_treated():
    """The exact property the naive analysis violates."""
    z = simulate.simulate_immortal_time(n=400, seed=43)
    cp = simulate.to_counting_process(z)
    # every treated interval must begin at or after that subject's start time
    assert (cp["start"][cp["treated"] == 1] > 0).all(), (
        "a treated interval starting at time 0 means someone was counted as "
        "treated before treatment began")
