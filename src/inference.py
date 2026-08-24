"""Median-survival intervals, robust variance, and the exact tie likelihood.

FOUR NAMED GAPS
---------------
"No confidence interval on median survival."
"No bootstrap or robust standard errors, and no clustering."
"The PH test is a simplified Grambsch-Therneau -- unscaled residuals."
"Exact tie handling is not implemented."

WHY A MEDIAN NEEDS A DIFFERENT INTERVAL FROM EVERYTHING ELSE
--------------------------------------------------------------
A Kaplan-Meier median is a quantile of a STEP FUNCTION. It is not a mean, it has
no standard error in the usual sense, and its sampling distribution is not
symmetric -- the upper bound is routinely much further from the point estimate
than the lower one, and is often infinite when the curve never reaches 0.5.

So `+/- 1.96 SE` is not merely imprecise here, it is the wrong shape. The
BROOKMEYER-CROWLEY interval inverts the test instead: the confidence set is
every time t at which the hypothesis S(t) = 0.5 would not be rejected, using the
Greenwood variance on a transformed scale. It handles the infinite upper bound
correctly, by returning it.

THE LOG-LOG TRANSFORM IS NOT DECORATION
-----------------------------------------
The interval is built on log(-log S(t)) rather than on S(t) directly, because a
plain-scale interval can extend past 0 or 1 -- and a confidence interval that
includes "survival probability 1.04" tells the reader the method was wrong, not
the data. The transform maps (0,1) onto the whole real line, so the interval
cannot escape.

ROBUST AND CLUSTERED VARIANCE
------------------------------
The model-based Cox variance assumes the model is correctly specified. The
sandwich (Lin-Wei) estimator does not: it is consistent for the variance of
beta-hat even under misspecification, and it is what makes a hazard ratio from a
model with a known assumption violation reportable at all.

CLUSTERING matters more than robustness in trial data and is more often
forgotten. Multi-centre trials, repeated events, and family-based designs all
produce correlated observations, and the model-based SE assumes independence.
It is ALWAYS TOO SMALL under positive within-cluster correlation, which means
the error is anti-conservative: narrower intervals and more significant results
than the data support. `robust_variance(cluster=...)` sums the score residuals
within cluster before forming the sandwich.

WHAT THIS IS NOT
----------------
The exact tie likelihood here enumerates permutations and is only used for small
tied sets -- it is factorial in the tie size, which is why Efron exists. No
frailty models, no time-varying coefficients, no penalised likelihood, no
firth correction for monotone likelihood.
"""

from __future__ import annotations

from itertools import permutations
from math import exp, log, sqrt

import numpy as np


# ---------------------------------------------------------------------------
# median survival with a Brookmeyer-Crowley interval
# ---------------------------------------------------------------------------

def greenwood_variance(times, surv, n_risk, n_event):
    """Var(log S(t)) by Greenwood's formula, cumulative."""
    out, running = [], 0.0
    for r, d in zip(n_risk, n_event):
        if r > d > 0:
            running += d / (r * (r - d))
        elif d > 0 and r == d:
            running = float("inf")
        out.append(running)
    return np.array(out)


def median_survival_ci(times, surv, n_risk, n_event, alpha=0.05):
    """Median with a Brookmeyer-Crowley interval on the log-log scale.

    THE CONFIDENCE SET IS EVERY t AT WHICH S(t) = 0.5 WOULD NOT BE REJECTED.
    That is what makes it handle the two cases a symmetric interval cannot:

      * an INFINITE upper bound, when the curve never drops far enough for the
        test to reject at any later time. Reported as None, not as the largest
        observed time -- which would be a made-up number presented as an
        estimate.
      * an UNDEFINED median, when the curve never reaches 0.5 at all. Also
        None, and distinguishable from the first case by the point estimate
        also being None.
    """
    times = np.asarray(times, dtype=float)
    surv = np.asarray(surv, dtype=float)
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _z(alpha)

    var_log_s = greenwood_variance(times, surv, n_risk, n_event)

    # point estimate: first time survival drops to or below 0.5
    below = np.where(surv <= 0.5)[0]
    median = float(times[below[0]]) if len(below) else None

    lo = None
    hi_idx = None
    for i, (t, s) in enumerate(zip(times, surv)):
        if not (0 < s < 1):
            continue
        v = var_log_s[i]
        if not np.isfinite(v) or v <= 0:
            continue
        # log(-log S) scale: Var(log(-log S)) = Var(log S) / (log S)^2
        ll = log(-log(s))
        se_ll = sqrt(v) / abs(log(s))
        target = log(-log(0.5))
        if abs(ll - target) <= z * se_ll:
            if lo is None:
                lo = float(t)
            hi_idx = i

    # THE UPPER BOUND IS THE NEXT EVENT TIME, NOT THE LAST ACCEPTED ONE.
    # The confidence set is a set of TIMES, and the KM curve is a step function
    # holding its value on the half-open interval [t_i, t_{i+1}). So if S(t_i)
    # is in the acceptance region, EVERY t in [t_i, t_{i+1}) is in the set, and
    # the supremum of the set is t_{i+1}.
    #
    # Reporting t_i instead understates the upper bound by one event-time step.
    # That is the ANTI-CONSERVATIVE direction -- a narrower interval than the
    # data support -- which is why it is worth a comment and not just a fix.
    # Found by differencing against lifelines: the lower bound agreed exactly on
    # every seed while the upper was short by exactly one step on every seed,
    # and an error that is asymmetric like that is a convention, not noise.
    #
    # The LOWER bound needs no such adjustment, and the asymmetry is the same
    # fact seen from the other end: the step at t_i makes the set closed below
    # (t_i itself is in it) and open above.
    if hi_idx is None:
        hi = None
    elif hi_idx + 1 < len(times):
        hi = float(times[hi_idx + 1])
    else:
        # The set runs to the last event time; nothing after it bounds the set.
        hi = None

    # An upper bound equal to the last event time is only a real bound if the
    # curve actually fell below 0.5 after it. Otherwise the set is open above.
    if hi is not None and median is not None and hi >= times[-1] and surv[-1] > 0.5:
        hi = None
    return {"median": median, "lo": lo, "hi": hi, "alpha": alpha,
            "method": "Brookmeyer-Crowley, log(-log) scale",
            "note": ("hi=None means the upper bound is not finite: the "
                     "confidence set is open above, which is the honest answer "
                     "when the curve never falls far enough to reject S(t)=0.5 "
                     "at any later time. Reporting the largest observed time "
                     "instead would present a made-up number as an estimate."
                     if hi is None else None)}


def _z(alpha):
    """Inverse normal CDF by bisection. No scipy."""
    lo, hi = -10.0, 10.0
    target = 1 - alpha / 2
    for _ in range(200):
        mid = (lo + hi) / 2
        # Phi via erf
        from math import erf
        if 0.5 * (1 + erf(mid / sqrt(2))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# robust and clustered variance
# ---------------------------------------------------------------------------

def score_residuals(time, event, X, beta):
    """Per-subject score residuals for the Cox partial likelihood.

    The building block of the sandwich. Each subject's residual is their
    contribution to the score at the fitted beta; summing their squares gives
    the meat of the sandwich, and summing WITHIN CLUSTER first gives the
    clustered version.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != len(time):
        X = X.T
    n, p = X.shape
    beta = np.asarray(beta, dtype=float).reshape(p)

    theta = np.exp(X @ beta)
    order = np.argsort(time)
    resid = np.zeros((n, p))

    for i in range(n):
        if not event[i]:
            continue
        at_risk = time >= time[i]
        s0 = theta[at_risk].sum()
        s1 = (theta[at_risk, None] * X[at_risk]).sum(axis=0)
        zbar = s1 / s0
        # the subject who failed contributes (x_i - zbar)
        resid[i] += X[i] - zbar
        # everyone at risk contributes -(x_j - zbar) * theta_j / s0
        w = theta[at_risk] / s0
        resid[at_risk] -= w[:, None] * (X[at_risk] - zbar)
    return resid


def robust_variance(time, event, X, beta, cov, cluster=None):
    """Sandwich variance: cov @ (U'U) @ cov, clustered if asked.

    WHY CLUSTERING IS THE ONE THAT MATTERS. The model-based SE assumes
    independent observations and is ALWAYS TOO SMALL under positive
    within-cluster correlation. That error is anti-conservative -- narrower
    intervals, more significance than the data support -- which is the
    direction that gets a finding published and then fails to replicate.

    Multi-centre trials, repeated events and family designs all produce it, and
    it is more often forgotten than misspecification is.
    """
    resid = score_residuals(time, event, X, beta)
    if cluster is not None:
        cluster = np.asarray(cluster)
        summed = []
        for c in np.unique(cluster):
            summed.append(resid[cluster == c].sum(axis=0))
        resid = np.array(summed)
    meat = resid.T @ resid
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    v = cov @ meat @ cov
    se = np.sqrt(np.diag(v))
    return {"cov": v, "se": se,
            "n_clusters": (len(np.unique(cluster)) if cluster is not None
                           else len(resid)),
            "clustered": cluster is not None}


# ---------------------------------------------------------------------------
# exact tie handling
# ---------------------------------------------------------------------------

def exact_tie_loglik(theta_risk, theta_tied):
    """log of the exact marginal likelihood contribution for one tied set.

    Averages over every ORDERING of the tied failures, which is what Efron
    approximates and Breslow ignores. Factorial in the tie size -- 8 tied
    deaths is 40,320 permutations -- which is exactly why Efron exists and why
    this is offered only for small sets.

    Included because "Efron approximates the exact likelihood" is a claim, and
    a project that makes it should be able to compute the thing being
    approximated on a case small enough to check.
    """
    d = len(theta_tied)
    if d == 0:
        return 0.0
    if d > 8:
        raise ValueError(
            f"{d} tied deaths is {d}! orderings; the exact likelihood is "
            f"factorial in the tie size, which is the reason Efron exists. "
            f"Use ties='efron'.")
    total = 0.0
    base = float(np.sum(theta_risk))
    for perm in permutations(range(d)):
        term = 1.0
        remaining = base
        for idx in perm:
            term *= theta_tied[idx] / remaining
            remaining -= theta_tied[idx]
        total += term
    return log(total)


def compare_tie_methods(theta_risk, theta_tied):
    """Exact vs Efron vs Breslow on one tied set, as a log-likelihood.

    All three are the SAME QUANTITY -- the log contribution of one tied event
    time to the partial likelihood -- so they are directly comparable.

    `exact_tie_loglik` already includes the numerators (each permutation term
    is a product of theta/remaining), so it needs no separate numerator added.
    Getting that wrong is easy and produces three numbers on different scales
    that still look plausible next to each other.

    Breslow uses the full risk set for every death, so its denominator is too
    large and its contribution too small -- the direction that attenuates
    coefficients toward the null. Efron subtracts the average already-failed
    mass. The exact value should sit closer to Efron than to Breslow, and this
    is where that is checked rather than asserted.
    """
    d = len(theta_tied)
    s0 = float(np.sum(theta_risk))
    td = float(np.sum(theta_tied))
    num = float(np.sum(np.log(theta_tied)))

    breslow = num - d * log(s0)
    efron = num - sum(log(s0 - (l / d) * td) for l in range(d))
    exact = exact_tie_loglik(theta_risk, theta_tied)

    return {"exact": exact, "efron": efron, "breslow": breslow,
            "efron_error": abs(efron - exact),
            "breslow_error": abs(breslow - exact),
            "closer": "efron" if abs(efron - exact) < abs(breslow - exact)
                      else "breslow",
            "n_tied": d, "n_permutations": _factorial(d)}


def _factorial(n):
    out = 1
    for k in range(2, n + 1):
        out *= k
    return out


# ---------------------------------------------------------------------------
# Grambsch-Therneau, properly: SCALED Schoenfeld residuals
# ---------------------------------------------------------------------------

def scaled_schoenfeld(res, cov, beta):
    """Scale the raw Schoenfeld residuals: s* = beta + d * V @ s.

    WHY SCALING IS NOT COSMETIC. A raw Schoenfeld residual is on the scale of
    the COVARIATE; the scaled one is on the scale of the COEFFICIENT, and its
    expectation at time t is beta(t). That is what makes the plot readable and
    the test interpretable: a trend in s* is literally a trend in the hazard
    ratio, and the slope is in log-hazard units.

    The unscaled version in `survival.ph_test` correlates residuals with time
    and gets the DIRECTION right, but its statistic is not the
    Grambsch-Therneau one and cannot be quoted as such -- which the reference
    audit measured before this existed: the ratio between them ranged over more
    than an order of magnitude.

    `d` is the number of events, `V` the covariance of beta.
    """
    res = np.asarray(res, dtype=float)
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    beta = np.asarray(beta, dtype=float)
    return beta[None, :] + len(res) * (res @ cov.T)


def _transform_time(times, how):
    """cox.zph's time transforms. The CHOICE CHANGES THE ANSWER.

    A test against raw time is dominated by the longest follow-up, where the
    risk set is smallest and the residuals noisiest. `rank` and `km` spread the
    weight over the observed event times instead, which is why they are the
    usual defaults -- and why quoting a p-value without saying which transform
    produced it is not quoting anything.
    """
    times = np.asarray(times, dtype=float)
    if how == "identity":
        return times
    if how == "log":
        return np.log(times)
    if how == "rank":
        return np.argsort(np.argsort(times)).astype(float)
    raise ValueError("unknown time transform %r; use rank, identity or log"
                     % how)


def scaled_ph_test(time, event, X, beta, cov, transform="rank"):
    """The real Grambsch-Therneau test, per covariate and globally.

    Validated to agree with `lifelines.statistics.proportional_hazard_test`
    exactly -- see `validate_reference.py`. `survival.ph_test` is kept as the
    cheap screen it always was; this is the one whose number can be quoted.

    Returns per-covariate chi2/p on 1 df, plus a GLOBAL test on p df. The
    global one matters because testing three covariates separately at 0.05 and
    reporting the smallest is a multiple-comparison problem, and the usual
    reason a PH violation gets "found".
    """
    from math import erfc, sqrt

    import survival as _S

    res_t, res = _S.schoenfeld_residuals(time, event, X, beta)
    n_events = len(res_t)
    if n_events < 5:
        return {"per_covariate": [], "n_events": n_events,
                "transform": transform, "global": None}

    beta = np.asarray(beta, dtype=float)
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    sstar = scaled_schoenfeld(res, cov, beta)

    g = _transform_time(res_t, transform)
    g = g - g.mean()
    sg2 = float(np.sum(g ** 2))

    per = []
    for j in range(sstar.shape[1]):
        num = float(g @ (sstar[:, j] - beta[j])) ** 2
        den = n_events * float(cov[j, j]) * sg2
        chi2 = num / den if den > 0 else 0.0
        p = erfc(sqrt(chi2 / 2)) if chi2 > 0 else 1.0
        per.append({"chi2": chi2, "p": p, "df": 1})

    # GLOBAL test: the quadratic form over all covariates at once.
    u = g @ (sstar - beta[None, :])              # p-vector
    try:
        inv = np.linalg.inv(cov)
        gchi2 = float(u @ inv @ u) / (n_events * sg2)
    except np.linalg.LinAlgError:                 # pragma: no cover
        gchi2 = float("nan")
    df = sstar.shape[1]
    gp = _chi2_sf(gchi2, df) if np.isfinite(gchi2) else float("nan")

    return {"per_covariate": per, "n_events": n_events,
            "transform": transform,
            "global": {"chi2": gchi2, "p": gp, "df": df},
            "scaled_residuals": sstar, "times": res_t}


def _chi2_sf(x, df):
    """Upper tail of a chi-square. Series/continued-fraction, no scipy.

    Written out rather than imported because `src/` has no third-party
    dependency -- the reference libraries are used to AUDIT this code, never to
    provide it.
    """
    from math import exp, lgamma, log

    if x <= 0:
        return 1.0
    a = df / 2.0
    xx = x / 2.0
    if xx < a + 1.0:
        # lower incomplete gamma by series
        term = 1.0 / a
        total = term
        n = 0
        while n < 1000:
            n += 1
            term *= xx / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        lower = total * exp(-xx + a * log(xx) - lgamma(a))
        return max(0.0, min(1.0, 1.0 - lower))
    # upper incomplete gamma by continued fraction (Lentz)
    tiny = 1e-300
    b = xx + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    upper = h * exp(-xx + a * log(xx) - lgamma(a))
    return max(0.0, min(1.0, upper))
