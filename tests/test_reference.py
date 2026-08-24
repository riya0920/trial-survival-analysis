"""Differences the hand-rolled estimators against lifelines and statsmodels.

THESE TESTS SKIP when the reference libraries are absent, and that is the
point: `src/` has no third-party survival dependency and the project runs
without one. The reference is used to AUDIT the implementation, never to
provide it.

Why bother, given the estimators already have unit tests? Because a unit test
written by the person who wrote the estimator shares its misconceptions. The
existing tests plant a hazard ratio and check it is recovered, which catches a
wrong answer but not a subtly wrong convention -- and a subtly wrong convention
is exactly what these found in `median_survival_ci`.

The tolerances are tight on purpose. These are the same formulas evaluated on
the same data, so anything worse than convergence-level agreement is a real
difference in the arithmetic and should fail rather than be absorbed.
"""

import os
import sys
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import inference as INF
import simulate as SIM
import survival as S

pd = pytest.importorskip("pandas", reason="reference audit only")
lifelines = pytest.importorskip("lifelines", reason="reference audit only")
sm = pytest.importorskip("statsmodels.api", reason="reference audit only")

from lifelines import CoxPHFitter, KaplanMeierFitter          # noqa: E402
from lifelines.statistics import logrank_test                 # noqa: E402
from lifelines.utils import median_survival_times             # noqa: E402

NAMES = ["arm", "age10", "stage"]


@pytest.fixture(scope="module")
def case():
    d = SIM.simulate_trial(n=500, seed=101)
    T = np.asarray(d["time"], dtype=float)
    E = np.asarray(d["event"], dtype=int)
    X = np.column_stack([d["arm"],
                         (np.asarray(d["age"], dtype=float) - 62) / 10,
                         d["stage"]])
    df = pd.DataFrame({"T": T, "E": E, "arm": X[:, 0],
                       "age10": X[:, 1], "stage": X[:, 2]})
    return T, E, X, df


def test_kaplan_meier_matches_to_machine_precision(case):
    """No iteration is involved, so anything above 1e-12 is a formula error."""
    T, E, _X, _df = case
    t, s, _nr, _ne, _se = S.kaplan_meier(T, E)
    ref = KaplanMeierFitter().fit(T, E)
    got = ref.survival_function_["KM_estimate"].reindex(t).to_numpy()
    assert np.max(np.abs(s - got)) < 1e-12


def test_logrank_matches(case):
    T, E, _X, df = case
    arm = np.asarray(df["arm"], dtype=int)
    chi2, p, _o, _e = S.logrank(T, E, arm)
    g = arm == 1
    ref = logrank_test(T[g], T[~g], E[g], E[~g])
    assert chi2 == pytest.approx(ref.test_statistic, rel=1e-10)
    assert p == pytest.approx(ref.p_value, rel=1e-8)


def test_cox_efron_matches_lifelines(case):
    """lifelines uses Efron by default, which is why Efron is the default here."""
    T, E, X, df = case
    mine = S.cox_ph(T, E, X, ties="efron")
    ref = CoxPHFitter().fit(df, "T", "E")
    for i, nm in enumerate(NAMES):
        assert mine["beta"][i] == pytest.approx(ref.params_[nm], abs=1e-4)
        assert mine["se"][i] == pytest.approx(ref.standard_errors_[nm],
                                              rel=1e-4)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_tie_handling_matches_statsmodels_under_heavy_ties(case, ties):
    """THE DISCRIMINATING CASE.

    Breslow and Efron agree closely on continuous time, so a validation run
    only there would let a broken Breslow pass. Rounding to whole months forces
    ties onto most event days, which is where the two methods actually differ
    -- and is also the realistic case, since trials record time in days.
    """
    T, E, X, _df = case
    Tc = np.round(T)
    mine = S.cox_ph(Tc, E, X, ties=ties)
    ref = sm.PHReg(Tc, X, status=E, ties=ties).fit()
    for i in range(X.shape[1]):
        assert mine["beta"][i] == pytest.approx(ref.params[i], abs=1e-6)


def test_breslow_is_attenuated_relative_to_efron_and_so_is_the_reference(case):
    """The documented claim, checked against a reference rather than asserted.

    Breslow uses the full risk set for every death in a tied set, which inflates
    the denominator and biases coefficients TOWARD ZERO. If that is real, the
    reference implementation must show it too -- otherwise the effect is in my
    arithmetic, not in the method.
    """
    T, E, X, _df = case
    Tc = np.round(T)
    mine_b = S.cox_ph(Tc, E, X, ties="breslow")["beta"]
    mine_e = S.cox_ph(Tc, E, X, ties="efron")["beta"]
    ref_b = sm.PHReg(Tc, X, status=E, ties="breslow").fit().params
    ref_e = sm.PHReg(Tc, X, status=E, ties="efron").fit().params
    for i in range(X.shape[1]):
        assert abs(mine_b[i]) < abs(mine_e[i])
        assert abs(ref_b[i]) < abs(ref_e[i])


def test_robust_sandwich_matches(case):
    T, E, X, df = case
    fit = S.cox_ph(T, E, X, ties="efron")
    mine = INF.robust_variance(T, E, X, np.asarray(fit["beta"]),
                               np.asarray(fit["cov"]), cluster=None)
    ref = CoxPHFitter().fit(df, "T", "E", robust=True)
    for i, nm in enumerate(NAMES):
        assert mine["se"][i] == pytest.approx(ref.standard_errors_[nm],
                                              rel=1e-4)


def test_clustered_sandwich_matches(case):
    """The one most easily got subtly wrong: scores are summed WITHIN cluster
    before the outer product, not after."""
    T, E, X, df = case
    site = np.arange(len(T)) % 20
    fit = S.cox_ph(T, E, X, ties="efron")
    mine = INF.robust_variance(T, E, X, np.asarray(fit["beta"]),
                               np.asarray(fit["cov"]), cluster=site)
    ref = CoxPHFitter().fit(df.assign(site=site), "T", "E", cluster_col="site")
    assert mine["n_clusters"] == 20
    for i, nm in enumerate(NAMES):
        assert mine["se"][i] == pytest.approx(ref.standard_errors_[nm],
                                              rel=1e-4)


def test_brookmeyer_crowley_bounds_match(case):
    """REGRESSION TEST FOR THE BUG THIS AUDIT FOUND.

    The upper bound used to be the last event time inside the acceptance
    region. The supremum of the confidence set is the NEXT event time, because
    the KM curve holds its value on [t_i, t_i+1) and every t in that half-open
    interval is in the set. The interval was one event time too narrow, which
    is the anti-conservative direction.
    """
    T, E, _X, _df = case
    t, s, nr, ne, _se = S.kaplan_meier(T, E)
    mine = INF.median_survival_ci(t, s, nr, ne)
    kmf = KaplanMeierFitter().fit(T, E)
    ref = median_survival_times(kmf.confidence_interval_).to_numpy()[0]
    assert mine["median"] == pytest.approx(kmf.median_survival_time_)
    assert mine["lo"] == pytest.approx(ref[0])
    assert mine["hi"] == pytest.approx(ref[1])


def test_the_upper_bound_is_the_next_event_time_not_the_last_accepted(case):
    """States the convention directly, so the regression above cannot be
    'fixed' back by someone who reads only the failing number."""
    T, E, _X, _df = case
    t, s, nr, ne, _se = S.kaplan_meier(T, E)
    mine = INF.median_survival_ci(t, s, nr, ne)
    hi = mine["hi"]
    assert hi in set(t.tolist())
    idx = int(np.searchsorted(t, hi))
    # the accepted time is the one BEFORE the reported bound
    assert idx >= 1
    assert t[idx] == pytest.approx(hi)
