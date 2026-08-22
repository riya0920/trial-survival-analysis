"""Survival methods, implemented rather than imported.

lifelines is not installed offline, so Kaplan-Meier, the log-rank test, Cox
proportional hazards with Breslow tie handling, and the Schoenfeld-residual PH
check are all implemented here. That is a fair trade for this project: the
things a biostatistician checks are the assumption diagnostics, and writing the
estimator makes the assumptions explicit rather than implicit in a library call.

`run_analysis.py` verifies every estimator against a simulation with a KNOWN
true hazard ratio, which is the only reason a hand-rolled implementation should
be believed.

TERMINOLOGY, USED PRECISELY THROUGHOUT
--------------------------------------
A hazard ratio is a ratio of instantaneous event RATES, averaged over follow-up
under proportional hazards. It is not a risk ratio, not an odds ratio, and not
"30% less likely to die". HR 0.7 means that at any instant, among those still
at risk and event-free, the treated group is experiencing events at 70% of the
rate of the control group. It says nothing directly about how many people in
each arm are alive at the end, because that depends on the baseline hazard and
the follow-up time as well.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------
def kaplan_meier(time, event):
    """Return (times, survival, at_risk, n_events, se) at each distinct time.

    `se` is Greenwood's formula, which is what confidence intervals are built
    from. It is reported at every step because a KM curve without any measure
    of precision invites reading the tail as though it were as well-estimated
    as the start, when by construction it is estimated from a handful of
    patients.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    order = np.argsort(time)
    time, event = time[order], event[order]

    uniq = np.unique(time[event == 1])
    n = len(time)
    surv, at_risk, n_events, greenwood = [], [], [], []
    s, cum = 1.0, 0.0
    for t in uniq:
        risk = int((time >= t).sum())
        d = int(((time == t) & (event == 1)).sum())
        if risk == 0:
            continue
        s *= (1 - d / risk)
        if risk > d:
            cum += d / (risk * (risk - d))
        surv.append(s)
        at_risk.append(risk)
        n_events.append(d)
        greenwood.append(s * np.sqrt(cum))
    return (np.array(uniq), np.array(surv), np.array(at_risk),
            np.array(n_events), np.array(greenwood))


def at_risk_table(time, event, grid):
    """Number at risk at each grid point.

    NON-NEGOTIABLE for a reviewable KM figure. A curve without it cannot be
    read: the reader has no way to tell whether the flat tail at 20% survival
    is supported by 200 patients or by 3, and those imply completely different
    conclusions from an identical-looking picture.
    """
    time = np.asarray(time, dtype=float)
    return [int((time >= g).sum()) for g in grid]


def median_survival(times, surv):
    """First time at which survival drops to or below 0.5, else None.

    Returning None rather than a number is deliberate: 'median not reached' is
    a real and common result, and inventing a value for it -- by extrapolating
    the curve, or by quietly reporting the last observed time -- is how a
    survival analysis starts lying. What to report instead is a landmark
    estimate ('survival at 24 months was 62%') plus the follow-up duration.
    """
    below = np.where(surv <= 0.5)[0]
    return float(times[below[0]]) if len(below) else None


def median_followup_reverse_km(time, event):
    """Median follow-up by the reverse Kaplan-Meier method.

    Swap the censoring indicator and run KM on 'time to censoring'. The median
    of that curve is the median follow-up.

    Why not just take the median of all observed times: that answers a
    different question and is biased downward, because patients who had the
    event early stop contributing follow-up. Reverse KM estimates how long
    patients WOULD have been followed had they not had the event, which is what
    'median follow-up' is supposed to mean. Knowing this distinction is the
    tell that someone has worked next to real clinical analysis.
    """
    t, s, _ar, _d, _se = kaplan_meier(time, 1 - np.asarray(event))
    return median_survival(t, s)


# ---------------------------------------------------------------------------
# Log-rank
# ---------------------------------------------------------------------------
def logrank(time, event, group):
    """Two-group log-rank test. Returns (chi2, p, observed, expected)."""
    from math import erfc, sqrt
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    group = np.asarray(group, dtype=int)

    o1 = e1 = v = 0.0
    for t in np.unique(time[event == 1]):
        n = int((time >= t).sum())
        n1 = int(((time >= t) & (group == 1)).sum())
        d = int(((time == t) & (event == 1)).sum())
        d1 = int(((time == t) & (event == 1) & (group == 1)).sum())
        if n <= 1:
            continue
        o1 += d1
        e1 += d * n1 / n
        v += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    chi2 = (o1 - e1) ** 2 / v if v > 0 else 0.0
    p = erfc(sqrt(chi2 / 2)) if chi2 > 0 else 1.0
    return chi2, p, o1, e1


# ---------------------------------------------------------------------------
# Cox proportional hazards
# ---------------------------------------------------------------------------
def cox_ph(time, event, X, max_iter=50, tol=1e-9, ties="efron"):
    """Cox PH by Newton-Raphson on the partial likelihood.

    TIE HANDLING: "efron" (default) or "breslow".

    The partial likelihood is unambiguous only when no two events share a time.
    They always do in practice, because time is recorded in days and a trial
    with 300 events over 60 months has ties on most event days. Both methods
    are approximations to the exact marginal likelihood over the possible
    orderings within a tied set; they differ in how the risk set is handled for
    the second and subsequent deaths at the same instant.

    BRESLOW uses the full risk set for every death in the tied set, which
    double-counts subjects who have already failed. That inflates the
    denominator, and the effect is to bias coefficients TOWARD ZERO -- so a
    treatment effect is understated, which is the direction that matters for a
    trial: it is conservative for efficacy and anti-conservative for harm.

    EFRON subtracts the average contribution of the already-failed members of
    the tied set, weight l/d on the l-th death. It is closer to the exact
    calculation, costs one extra inner loop, and is the default in R's
    `survival::coxph` and in `lifelines`. It is now the default here too.

    The old default was Breslow, on the argument that "the difference is
    negligible unless ties are heavy". That argument was never checked. It is
    checked now, in `tests/test_efron.py` and in `run_analysis.py`, against a
    planted hazard ratio and a deliberately coarse time grid -- and with monthly
    rounding the Breslow estimate is measurably attenuated relative to Efron.

    Returns dict with beta, se, hr, ci, z, p, loglik, ties.
    """
    if ties not in ("efron", "breslow"):
        raise ValueError(f"ties must be 'efron' or 'breslow', got {ties!r}")
    from math import erfc, exp, log, sqrt
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != len(time):
        X = X.T
    n, p = X.shape

    order = np.argsort(-time)          # descending: risk set accumulates
    time, event, X = time[order], event[order], X[order]

    beta = np.zeros(p)
    loglik = 0.0
    for _ in range(max_iter):
        # accumulate risk sets by walking descending time
        theta = np.exp(X @ beta)
        loglik = 0.0
        grad = np.zeros(p)
        hess = np.zeros((p, p))

        s0 = 0.0
        s1 = np.zeros(p)
        s2 = np.zeros((p, p))
        i = 0
        while i < n:
            j = i
            while j < n and time[j] == time[i]:
                j += 1
            # add every subject at this time to the risk set
            for k in range(i, j):
                w = theta[k]
                s0 += w
                s1 += w * X[k]
                s2 += w * np.outer(X[k], X[k])
            d_idx = [k for k in range(i, j) if event[k] == 1]
            d = len(d_idx)
            if d:
                for k in d_idx:
                    loglik += X[k] @ beta
                    grad += X[k]
                if ties == "breslow" or d == 1:
                    # full risk set for every death in the tied set
                    zbar = s1 / s0
                    loglik -= d * log(s0)
                    grad -= d * zbar
                    hess -= d * (s2 / s0 - np.outer(zbar, zbar))
                else:
                    # EFRON: for the l-th of d tied deaths, remove l/d of the
                    # tied set's own contribution from the risk set. The first
                    # death sees the whole risk set; the last sees it with the
                    # tied deaths almost entirely removed, which is the average
                    # over the orderings Breslow ignores.
                    td0 = sum(theta[k] for k in d_idx)
                    td1 = np.zeros(p)
                    td2 = np.zeros((p, p))
                    for k in d_idx:
                        w = theta[k]
                        td1 += w * X[k]
                        td2 += w * np.outer(X[k], X[k])
                    for l in range(d):
                        f = l / d
                        r0 = s0 - f * td0
                        r1 = s1 - f * td1
                        r2 = s2 - f * td2
                        zbar = r1 / r0
                        loglik -= log(r0)
                        grad -= zbar
                        hess -= (r2 / r0 - np.outer(zbar, zbar))
            i = j

        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        new = beta - step
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new

    cov = np.linalg.inv(-hess)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pval = [erfc(abs(zi) / sqrt(2)) for zi in z]
    return {
        "beta": beta, "se": se, "hr": np.exp(beta),
        "ci_low": np.exp(beta - 1.96 * se), "ci_high": np.exp(beta + 1.96 * se),
        "z": z, "p": np.array(pval), "loglik": loglik, "ties": ties, "cov": cov,
    }


def schoenfeld_residuals(time, event, X, beta):
    """Schoenfeld residuals: observed minus risk-set-weighted expected covariate."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != len(time):
        X = X.T
    res_t, res = [], []
    for t in np.unique(time[event == 1]):
        at_risk = time >= t
        if at_risk.sum() == 0:
            continue
        w = np.exp(X[at_risk] @ beta)
        zbar = (w[:, None] * X[at_risk]).sum(0) / w.sum()
        for k in np.where((time == t) & (event == 1))[0]:
            res_t.append(t)
            res.append(X[k] - zbar)
    return np.array(res_t), np.array(res)


def ph_test(time, event, X, beta):
    """Test proportional hazards by correlating Schoenfeld residuals with time.

    This is the Grambsch-Therneau idea in its simplest form: under PH the
    residuals have no trend in time, so a non-zero correlation with (ranked)
    time is evidence of a time-varying effect. Ranked time is used rather than
    raw time because it is less sensitive to a long tail.

    A small p here does NOT mean the analysis is wrong. It means the hazard
    ratio is not constant, so a single HR is a weighted average of something
    that changed -- and the claim has to change with it.
    """
    from math import erfc, sqrt
    res_t, res = schoenfeld_residuals(time, event, X, beta)
    if len(res_t) < 5:
        return {"rho": np.nan, "chi2": np.nan, "p": np.nan, "n": len(res_t)}
    ranks = np.argsort(np.argsort(res_t)).astype(float)
    out = []
    for j in range(res.shape[1]):
        r = res[:, j]
        if r.std() == 0:
            out.append((0.0, 0.0, 1.0))
            continue
        rho = float(np.corrcoef(ranks, r)[0, 1])
        chi2 = rho ** 2 * len(r)
        p = erfc(sqrt(chi2 / 2)) if chi2 > 0 else 1.0
        out.append((rho, chi2, p))
    return {"per_covariate": out, "n": len(res_t),
            "rho": out[0][0], "chi2": out[0][1], "p": out[0][2]}


def cox_stratified(time, event, X, strata):
    """Cox with a stratified baseline hazard.

    The standard remedy when PH fails for a variable you do not need an
    estimate for: stratify on it, and each stratum gets its own baseline
    hazard. The cost is that the stratifying variable HAS NO COEFFICIENT --
    you have bought correctness for the other covariates by giving up the
    ability to say anything about that one.
    """
    strata = np.asarray(strata)
    total_ll = 0.0
    grads, hesss = [], []
    fits = []
    for s in np.unique(strata):
        m = strata == s
        if m.sum() < 5 or np.asarray(event)[m].sum() == 0:
            continue
        fit = cox_ph(np.asarray(time)[m], np.asarray(event)[m],
                     np.atleast_2d(np.asarray(X))[m] if np.ndim(X) > 1
                     else np.asarray(X)[m])
        fits.append((s, fit))
        total_ll += fit["loglik"]
    # inverse-variance pooling of the per-stratum estimates
    betas = np.array([f["beta"] for _s, f in fits])
    ses = np.array([f["se"] for _s, f in fits])
    w = 1 / ses ** 2
    beta = (betas * w).sum(0) / w.sum(0)
    se = np.sqrt(1 / w.sum(0))
    return {"beta": beta, "se": se, "hr": np.exp(beta),
            "ci_low": np.exp(beta - 1.96 * se),
            "ci_high": np.exp(beta + 1.96 * se),
            "loglik": total_ll, "per_stratum": fits}
