"""The real Grambsch-Therneau test, on scaled Schoenfeld residuals.

`survival.ph_test` correlates UNSCALED residuals with ranked time. It gets the
direction right and is documented as a screen, but its statistic is not the
Grambsch-Therneau one -- the reference audit measured the ratio between them
ranging over more than an order of magnitude.

`inference.scaled_ph_test` is the quotable version. Most of these tests need no
reference library; the agreement tests skip without lifelines.
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
import survival as S

NAMES = ["arm", "age10", "stage"]


@pytest.fixture(scope="module")
def fitted():
    d = SIM.simulate_trial(n=500, seed=101)
    T = np.asarray(d["time"], dtype=float)
    E = np.asarray(d["event"], dtype=int)
    X = np.column_stack([d["arm"],
                         (np.asarray(d["age"], dtype=float) - 62) / 10,
                         d["stage"]])
    fit = S.cox_ph(T, E, X, ties="efron")
    return T, E, X, np.asarray(fit["beta"]), np.asarray(fit["cov"])


# ------------------------------------------------------------- the scaling
def test_scaled_residuals_are_centred_on_beta(fitted):
    """The whole point of scaling: s* is on the scale of the COEFFICIENT, so
    its expectation at time t is beta(t). Under proportional hazards, beta(t)
    is constant, so the residuals average to beta."""
    T, E, X, beta, cov = fitted
    _t, res = S.schoenfeld_residuals(T, E, X, beta)
    sstar = INF.scaled_schoenfeld(res, cov, beta)
    assert sstar.shape == res.shape
    assert np.allclose(sstar.mean(axis=0), beta, atol=0.05)


def test_unscaled_residuals_are_centred_on_zero_not_beta(fitted):
    """The contrast that makes the previous test meaningful. A raw Schoenfeld
    residual is observed-minus-expected covariate, so it centres on zero and
    carries no information about the size of the coefficient."""
    T, E, X, beta, _cov = fitted
    _t, res = S.schoenfeld_residuals(T, E, X, beta)
    assert np.allclose(res.mean(axis=0), 0.0, atol=0.05)


# --------------------------------------------------------------- the test
def test_per_covariate_and_global_are_reported(fitted):
    T, E, X, beta, cov = fitted
    out = INF.scaled_ph_test(T, E, X, beta, cov)
    assert len(out["per_covariate"]) == 3
    assert out["global"]["df"] == 3
    for row in out["per_covariate"]:
        assert row["df"] == 1
        assert 0.0 <= row["p"] <= 1.0
    assert 0.0 <= out["global"]["p"] <= 1.0


def test_the_global_test_exists_because_three_tests_at_005_is_not_one(fitted):
    """Testing three covariates separately and reporting the smallest p is a
    multiple-comparison problem, and the usual way a PH violation gets
    'found'. The global test is on p degrees of freedom for that reason."""
    T, E, X, beta, cov = fitted
    out = INF.scaled_ph_test(T, E, X, beta, cov)
    smallest = min(r["p"] for r in out["per_covariate"])
    assert out["global"]["p"] >= smallest


def test_the_time_transform_changes_the_answer(fitted):
    """Quoting a PH p-value without naming the transform is not quoting
    anything. A test against raw time is dominated by the longest follow-up,
    where the risk set is smallest and the residuals noisiest."""
    T, E, X, beta, cov = fitted
    rank = INF.scaled_ph_test(T, E, X, beta, cov, transform="rank")
    ident = INF.scaled_ph_test(T, E, X, beta, cov, transform="identity")
    a = [r["chi2"] for r in rank["per_covariate"]]
    b = [r["chi2"] for r in ident["per_covariate"]]
    assert a != b


def test_an_unknown_transform_is_refused(fitted):
    T, E, X, beta, cov = fitted
    with pytest.raises(ValueError):
        INF.scaled_ph_test(T, E, X, beta, cov, transform="whatever")


# ------------------------------------------------------- the chi-square tail
@pytest.mark.parametrize("df", [1, 2, 3, 5])
def test_chi2_survival_function_matches_scipy(df):
    """`src/` has no scipy dependency, so the upper tail is written out. That
    is a claim about numerics and is checked like one."""
    scipy_stats = pytest.importorskip("scipy.stats")
    for x in (0.001, 0.5, 2.0, 7.5, 20.0, 40.0):
        assert INF._chi2_sf(x, df) == pytest.approx(
            float(scipy_stats.chi2.sf(x, df)), rel=1e-9)


def test_chi2_tail_is_monotone_and_bounded():
    prev = 1.0
    for x in (0.0, 0.5, 1.0, 5.0, 10.0, 50.0):
        v = INF._chi2_sf(x, 3)
        assert 0.0 <= v <= 1.0
        assert v <= prev + 1e-12
        prev = v


# ------------------------------------------------------------- vs reference
def test_scaled_test_matches_lifelines(fitted):
    """THE POINT. The simplified screen does not match and is not supposed to;
    this one does."""
    pd = pytest.importorskip("pandas", reason="reference audit only")
    pytest.importorskip("lifelines", reason="reference audit only")
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    T, E, X, beta, cov = fitted
    df = pd.DataFrame({"T": T, "E": E, "arm": X[:, 0],
                       "age10": X[:, 1], "stage": X[:, 2]})
    mine = INF.scaled_ph_test(T, E, X, beta, cov, transform="rank")
    ref = proportional_hazard_test(CoxPHFitter().fit(df, "T", "E"), df,
                                   time_transform="rank").summary
    for i, nm in enumerate(NAMES):
        assert mine["per_covariate"][i]["chi2"] == pytest.approx(
            float(ref.loc[nm, "test_statistic"]), rel=1e-4)
        assert mine["per_covariate"][i]["p"] == pytest.approx(
            float(ref.loc[nm, "p"]), rel=1e-4)


def test_the_simplified_screen_still_does_not_match(fitted):
    """Kept deliberately. If `ph_test` ever starts agreeing exactly, somebody
    has quietly replaced it with the scaled version and the documented
    distinction between screen and test has stopped being true."""
    pd = pytest.importorskip("pandas", reason="reference audit only")
    pytest.importorskip("lifelines", reason="reference audit only")
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    T, E, X, beta, cov = fitted
    df = pd.DataFrame({"T": T, "E": E, "arm": X[:, 0],
                       "age10": X[:, 1], "stage": X[:, 2]})
    simple = S.ph_test(T, E, X, beta)
    ref = proportional_hazard_test(CoxPHFitter().fit(df, "T", "E"), df,
                                   time_transform="rank").summary
    ratios = [simple["per_covariate"][i][1] / float(ref.loc[nm,
                                                            "test_statistic"])
              for i, nm in enumerate(NAMES)]
    assert not all(abs(r - 1.0) < 1e-3 for r in ratios)
