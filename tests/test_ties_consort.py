"""Tests for Efron tie handling and CONSORT participant accounting.

The Efron tests are recovery tests: plant a hazard ratio, create ties by
rounding, and check the estimator that is supposed to be less biased is less
biased. Checking only that the two methods differ would pass for an
implementation that differs in the wrong direction.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import consort as CS
import simulate as SIM
from survival import cox_ph


def _trial(seed=7, n=400):
    d = SIM.simulate_trial(n=n, seed=seed)
    return (np.asarray(d["time"], float), np.asarray(d["event"], int),
            np.asarray(d["arm"], float).reshape(-1, 1))


# --------------------------------------------------------------------------
# Efron
# --------------------------------------------------------------------------

def test_the_two_methods_are_identical_when_there_are_no_ties():
    """The first thing to confirm. With unique event times the two expressions
    reduce to the same partial likelihood, so any difference here would be an
    implementation bug rather than a property of the estimators."""
    t, e, x = _trial()
    assert len(np.unique(t[e == 1])) == int(e.sum())      # genuinely no ties
    a = cox_ph(t, e, x, ties="efron")
    b = cox_ph(t, e, x, ties="breslow")
    assert a["hr"][0] == pytest.approx(b["hr"][0], abs=1e-9)
    assert a["loglik"] == pytest.approx(b["loglik"], abs=1e-9)


def test_a_single_death_at_a_time_uses_the_full_risk_set():
    """Efron's correction is l/d for the l-th of d deaths; with d=1 the
    correction is zero, so the two must agree even in a tied dataset as long as
    no tie contains two EVENTS."""
    # Covariate deliberately NOT aligned with the event indicator: the first
    # version put every event in one arm, which is perfect separation, sends
    # beta to -infinity and overflows in exp(). Both methods "agreed" because
    # both diverged, which tests nothing.
    t = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
    e = np.array([1, 0, 1, 0, 1, 0, 1, 0])                # one event per time
    x = np.array([[1.0], [1.0], [0.0], [0.0], [1.0], [0.0], [0.0], [1.0]])
    assert cox_ph(t, e, x, ties="efron")["hr"][0] == pytest.approx(
        cox_ph(t, e, x, ties="breslow")["hr"][0], abs=1e-9)


def test_breslow_is_more_attenuated_than_efron_under_heavy_ties():
    """The claim the default rests on, checked rather than assumed. Averaged
    over replicates: one trial cannot separate an estimator's bias from its
    sampling noise."""
    ref, bres, efr = [], [], []
    for seed in range(12):
        t, e, x = _trial(seed=100 + seed)
        ref.append(cox_ph(t, e, x, ties="efron")["hr"][0])
        tq = np.maximum(3.0, np.round(t / 3.0) * 3.0)     # quarterly grid
        bres.append(cox_ph(tq, e, x, ties="breslow")["hr"][0])
        efr.append(cox_ph(tq, e, x, ties="efron")["hr"][0])
    ref_hr = float(np.mean(ref))
    # measured against the TIE-FREE estimate on the same data, which isolates
    # the tie effect from finite-sample bias
    drift_b = abs(float(np.mean(bres)) - ref_hr)
    drift_e = abs(float(np.mean(efr)) - ref_hr)
    assert drift_e < drift_b
    assert float(np.mean(bres)) > ref_hr        # attenuated toward the null


def test_efron_is_nearly_invariant_to_the_time_grid():
    """The property worth having: an estimate should depend on the data, not on
    how coarsely somebody recorded the dates."""
    t, e, x = _trial(seed=3)
    base = cox_ph(t, e, x, ties="efron")["hr"][0]
    for grid in (1.0, 3.0):
        tg = np.maximum(grid, np.round(t / grid) * grid)
        assert cox_ph(tg, e, x, ties="efron")["hr"][0] == pytest.approx(
            base, abs=0.01)


def test_the_method_used_is_reported_on_the_fit():
    """A fit that does not say how it handled ties cannot be reproduced."""
    t, e, x = _trial()
    assert cox_ph(t, e, x)["ties"] == "efron"          # the default changed
    assert cox_ph(t, e, x, ties="breslow")["ties"] == "breslow"


def test_an_unknown_tie_method_is_refused():
    t, e, x = _trial()
    with pytest.raises(ValueError) as err:
        cox_ph(t, e, x, ties="exact")
    assert "efron" in str(err.value)


def test_efron_still_recovers_the_planted_hazard_ratio():
    t, e, x = _trial(n=900, seed=31)
    fit = cox_ph(t, e, x, ties="efron")
    assert fit["ci_low"][0] < SIM.TRUE_HR_TREATMENT < fit["ci_high"][0]


# --------------------------------------------------------------------------
# CONSORT
# --------------------------------------------------------------------------

def _flow(**over):
    arms = {
        "Treatment": {"allocated": 100, "not_treated": 3,
                      "discontinued": {"withdrew consent": 5,
                                       "lost to follow-up": 2},
                      "analysed_itt": 100, "analysed_pp": 92},
        "Control": {"allocated": 100, "not_treated": 2,
                    "discontinued": {"withdrew consent": 4,
                                     "lost to follow-up": 3},
                    "analysed_itt": 100, "analysed_pp": 94},
    }
    for k, v in over.items():
        arm, field = k.split("__", 1)
        arms[arm][field] = v
    return CS.build_flow(260, {"ineligible": 40, "declined": 20}, arms)


def test_a_well_formed_flow_closes():
    assert CS.validate(_flow()) == []


def test_completion_is_derived_not_asserted():
    f = _flow()
    a = f["arms"]["Treatment"]
    assert a["completed"] == a["allocated"] - a["not_treated"] - 7


def test_allocation_not_summing_to_randomised_is_caught():
    """The most serious CONSORT failure, because it is invisible in the
    published numbers: someone randomised and then absent from the diagram."""
    f = _flow(Treatment__allocated=90)
    problems = CS.validate(f)
    assert problems and "allocation does not close" in problems[0]


def test_an_analysis_population_larger_than_allocation_is_caught():
    """An analysis population cannot grow after randomisation."""
    f = _flow(Treatment__analysed_itt=105)
    assert any("exceeds allocation" in p for p in CS.validate(f))


def test_per_protocol_larger_than_itt_is_caught():
    """Impossible by construction, and a sign two denominators were mixed."""
    f = _flow(Treatment__analysed_pp=101)
    assert any("exceeds ITT" in p for p in CS.validate(f))


def test_double_counted_discontinuation_is_caught():
    """A participant counted in two terminal states drives completion
    negative, which is the only visible symptom."""
    f = _flow(Treatment__discontinued={"withdrew consent": 60,
                                       "lost to follow-up": 60})
    assert any("counted in two terminal states" in p for p in CS.validate(f))


def test_screening_arithmetic_is_checked():
    f = _flow()
    f["assessed"] = 999
    assert any("screening does not close" in p for p in CS.validate(f))


def test_the_rendered_diagram_shows_every_terminal_state():
    text = CS.render_text(_flow())
    for label in ("Assessed for eligibility", "Randomised",
                  "Did not receive intervention", "withdrew consent",
                  "lost to follow-up", "Completed follow-up",
                  "Analysed (ITT)", "Analysed (per protocol)"):
        assert label in text


def test_exclusion_reasons_are_itemised_not_just_totalled():
    """A total with no reasons is the version that hides the selection."""
    text = CS.render_text(_flow())
    assert "ineligible" in text and "declined" in text
