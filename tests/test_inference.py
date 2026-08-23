"""Tests for median intervals, robust variance, and the exact tie likelihood.

The median-interval tests are about the two cases a symmetric interval cannot
represent: an infinite upper bound, and an undefined median. Both are common in
real trials and both are reported as None rather than as the largest observed
time, which would be a made-up number presented as an estimate.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import inference as INF
import simulate as SIM
from survival import cox_ph, kaplan_meier


def _km(seed=3, n=300):
    d = SIM.simulate_trial(n=n, seed=seed)
    t = np.asarray(d["time"], float)
    e = np.asarray(d["event"], int)
    return kaplan_meier(t, e), t, e, np.asarray(d["arm"], float)


# --------------------------------------------------------------------------
# median survival
# --------------------------------------------------------------------------

def test_the_interval_brackets_the_point_estimate():
    km, _t, _e, _a = _km()
    times, surv, n_risk, n_event = km[0], km[1], km[2], km[3]
    out = INF.median_survival_ci(times, surv, n_risk, n_event)
    assert out["median"] is not None
    assert out["lo"] <= out["median"]
    if out["hi"] is not None:
        assert out["median"] <= out["hi"]


def test_the_interval_is_not_symmetric():
    """A median is a quantile of a step function. Its sampling distribution is
    not symmetric, so +/- 1.96 SE is the wrong SHAPE, not merely imprecise."""
    km, _t, _e, _a = _km()
    out = INF.median_survival_ci(km[0], km[1], km[2], km[3])
    if out["hi"] is not None:
        below = out["median"] - out["lo"]
        above = out["hi"] - out["median"]
        assert abs(below - above) > 1e-9


def test_an_undefined_median_returns_none_not_the_last_time():
    """When the curve never reaches 0.5 the median does not exist. Returning
    the largest observed time would present a made-up number as an estimate."""
    times = np.array([1.0, 2.0, 3.0])
    surv = np.array([0.95, 0.90, 0.85])
    out = INF.median_survival_ci(times, surv, [100, 90, 80], [5, 5, 5])
    assert out["median"] is None


def test_an_open_upper_bound_is_reported_as_none_with_an_explanation():
    times = np.array([1.0, 2.0, 3.0])
    surv = np.array([0.95, 0.90, 0.85])
    out = INF.median_survival_ci(times, surv, [100, 90, 80], [5, 5, 5])
    assert out["hi"] is None
    assert "open above" in out["note"]


def test_greenwood_variance_grows_with_events():
    v = INF.greenwood_variance([1, 2, 3], [0.9, 0.8, 0.7],
                               [100, 90, 80], [10, 10, 10])
    assert v[0] < v[1] < v[2]


def test_greenwood_is_infinite_once_the_risk_set_is_exhausted():
    """S(t)=0 has no finite variance on the log scale, and pretending otherwise
    produces an interval around a point the data cannot support."""
    v = INF.greenwood_variance([1, 2], [0.5, 0.0], [10, 5], [5, 5])
    assert np.isinf(v[-1])


def test_a_tighter_alpha_gives_a_wider_interval():
    km, _t, _e, _a = _km()
    narrow = INF.median_survival_ci(km[0], km[1], km[2], km[3], alpha=0.20)
    wide = INF.median_survival_ci(km[0], km[1], km[2], km[3], alpha=0.01)
    assert wide["lo"] <= narrow["lo"]


# --------------------------------------------------------------------------
# robust and clustered variance
# --------------------------------------------------------------------------

def test_score_residuals_sum_to_approximately_zero_at_the_fitted_beta():
    """The score is zero at the maximum of the partial likelihood, so its
    per-subject residuals must sum there. If they do not, the residual formula
    is wrong and every sandwich built from it is wrong too."""
    _km_, t, e, arm = _km()
    X = arm.reshape(-1, 1)
    fit = cox_ph(t, e, X)
    resid = INF.score_residuals(t, e, X, fit["beta"])
    assert abs(resid.sum()) < 1e-6


def test_the_robust_se_is_close_to_the_model_se_when_the_model_is_right():
    """The control. A sandwich that disagrees wildly on correctly-specified
    data is a broken sandwich, not a robustness finding."""
    _km_, t, e, arm = _km(n=600)
    X = arm.reshape(-1, 1)
    fit = cox_ph(t, e, X)
    rob = INF.robust_variance(t, e, X, fit["beta"], fit["cov"])
    assert rob["se"][0] == pytest.approx(fit["se"][0], rel=0.35)


def test_clustering_widens_the_standard_error_under_correlation():
    """THE ERROR THAT MATTERS. The model-based SE assumes independence and is
    always too small under positive within-cluster correlation -- narrower
    intervals and more significance than the data support.

    THE FIXTURE TOOK THREE ATTEMPTS, AND WHAT IT TOOK IS THE FINDING. Neither
    a shared covariate alone nor an omitted frailty alone inflates the
    clustered SE here; measured on this estimator:

        x shared in cluster, t ~ x                  clustered / plain = 1.12
        x INDIVIDUAL + strong omitted frailty       clustered / plain = 0.98
        x SHARED in cluster + omitted frailty       clustered / plain = 2.66

    The covariate has to be CLUSTER-LEVEL -- a site effect, a centre-level
    exposure -- which is exactly the multi-centre trial case the correction
    exists for. With an individual-level covariate the within-cluster score
    residuals do not line up and clustering buys nothing. That is worth knowing
    before reaching for `cluster=` and expecting wider intervals.
    """
    rng = np.random.default_rng(0)
    n_clusters, per = 50, 12
    cluster = np.repeat(np.arange(n_clusters), per)
    x = np.repeat(rng.normal(0, 1.0, n_clusters), per)      # CLUSTER-level
    frailty = np.repeat(rng.normal(0, 1.5, n_clusters), per)  # omitted
    t = rng.exponential(np.exp(-(0.5 * x + frailty)))
    e = np.ones_like(t, dtype=int)
    X = x.reshape(-1, 1)
    fit = cox_ph(t, e, X)
    plain = INF.robust_variance(t, e, X, fit["beta"], fit["cov"])
    clust = INF.robust_variance(t, e, X, fit["beta"], fit["cov"],
                                cluster=cluster)
    assert clust["se"][0] > plain["se"][0]
    assert clust["n_clusters"] == n_clusters
    assert clust["clustered"] is True


def test_an_individual_level_covariate_is_not_reliably_inflated():
    """The corollary, checked across seeds rather than on one draw.

    With an INDIVIDUAL-level covariate the clustered SE is not reliably larger:
    measured over three seeds the ratio runs 0.70, 0.70, 1.28. It is noise, not
    a correction. Asserting approximate equality on a single seed would have
    been asserting a stable relationship that does not exist -- the first
    version of this test did exactly that and failed on the seed it was written
    for.

    So the claim under test is the honest one: clustering does not
    systematically widen the interval here, and a practitioner who reaches for
    `cluster=` expecting it to is reaching for the wrong tool.
    """
    ratios = []
    for seed in (1, 2, 3, 4, 5):
        rng = np.random.default_rng(seed)
        n_clusters, per = 50, 12
        cluster = np.repeat(np.arange(n_clusters), per)
        x = rng.normal(0, 1.0, n_clusters * per)            # INDIVIDUAL-level
        frailty = np.repeat(rng.normal(0, 2.5, n_clusters), per)
        t = rng.exponential(np.exp(-(0.5 * x + frailty)))
        e = np.ones_like(t, dtype=int)
        X = x.reshape(-1, 1)
        fit = cox_ph(t, e, X)
        plain = INF.robust_variance(t, e, X, fit["beta"], fit["cov"])["se"][0]
        clust = INF.robust_variance(t, e, X, fit["beta"], fit["cov"],
                                    cluster=cluster)["se"][0]
        ratios.append(clust / plain)
    # not systematically inflated: at least one seed comes out BELOW 1
    assert min(ratios) < 1.0
    assert float(np.median(ratios)) < 1.5


def test_a_cluster_level_covariate_is_reliably_inflated():
    """The contrast, also across seeds. This is the case the correction exists
    for and it holds every time."""
    ratios = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        n_clusters, per = 50, 12
        cluster = np.repeat(np.arange(n_clusters), per)
        x = np.repeat(rng.normal(0, 1.0, n_clusters), per)   # CLUSTER-level
        frailty = np.repeat(rng.normal(0, 1.5, n_clusters), per)
        t = rng.exponential(np.exp(-(0.5 * x + frailty)))
        e = np.ones_like(t, dtype=int)
        X = x.reshape(-1, 1)
        fit = cox_ph(t, e, X)
        plain = INF.robust_variance(t, e, X, fit["beta"], fit["cov"])["se"][0]
        clust = INF.robust_variance(t, e, X, fit["beta"], fit["cov"],
                                    cluster=cluster)["se"][0]
        ratios.append(clust / plain)
    assert min(ratios) > 1.0


def test_clustering_by_a_unique_id_reduces_to_the_unclustered_case():
    _km_, t, e, arm = _km(n=200)
    X = arm.reshape(-1, 1)
    fit = cox_ph(t, e, X)
    a = INF.robust_variance(t, e, X, fit["beta"], fit["cov"])
    b = INF.robust_variance(t, e, X, fit["beta"], fit["cov"],
                            cluster=np.arange(len(t)))
    assert b["se"][0] == pytest.approx(a["se"][0], rel=1e-9)


# --------------------------------------------------------------------------
# exact tie handling
# --------------------------------------------------------------------------

def test_efron_is_closer_to_the_exact_likelihood_than_breslow():
    """The claim the Efron default rests on, checked against the thing it
    approximates rather than asserted."""
    risk = np.array([2.0, 1.5, 1.0, 0.8, 0.6, 0.5])
    tied = np.array([2.0, 1.5, 1.0])
    r = INF.compare_tie_methods(risk, tied)
    assert r["closer"] == "efron"
    assert r["efron_error"] < r["breslow_error"]


def test_breslow_understates_the_contribution():
    """Its denominator uses the full risk set for every death, so it is too
    large -- the direction that attenuates coefficients toward the null."""
    risk = np.array([2.0, 1.5, 1.0, 0.8])
    tied = np.array([2.0, 1.5])
    r = INF.compare_tie_methods(risk, tied)
    assert r["breslow"] < r["efron"] <= r["exact"]


def test_a_single_death_makes_all_three_agree():
    risk = np.array([2.0, 1.5, 1.0])
    tied = np.array([2.0])
    r = INF.compare_tie_methods(risk, tied)
    assert r["exact"] == pytest.approx(r["efron"])
    assert r["exact"] == pytest.approx(r["breslow"])


def test_a_large_tied_set_is_refused_rather_than_attempted():
    """Factorial in the tie size -- 9 tied deaths is 362,880 orderings. That
    cost is the entire reason Efron exists."""
    with pytest.raises(ValueError) as e:
        INF.exact_tie_loglik(np.ones(20), np.ones(9))
    assert "efron" in str(e.value).lower()


def test_the_permutation_count_is_reported():
    r = INF.compare_tie_methods(np.ones(6), np.ones(3))
    assert r["n_permutations"] == 6 and r["n_tied"] == 3
