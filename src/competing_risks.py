"""Competing risks, and restricted mean survival time.

WHY COMPETING RISKS ARE NOT AN ADVANCED TOPIC
---------------------------------------------
They are the default situation in any older or sicker population, and ignoring
them produces a specific, predictable, always-upward error.

The mistake is to take the event of interest (say, death from cancer), treat
every other kind of death as CENSORED, and read 1 - KM as "the probability of
dying from cancer by time t". Censoring assumes the censored patient could
still have the event later, and remains at risk in a meaningful sense. A patient
who died of something else cannot later die of cancer. Treating them as censored
implicitly redistributes their risk to the survivors, so 1 - KM **overestimates**
cumulative incidence -- and it overestimates it more when the competing risk is
common, which is exactly when someone is most likely to reach for it.

The correct estimator is the **Aalen-Johansen cumulative incidence function**,
which accounts for the fact that a competing event removes you permanently.
`compare_naive_vs_cif()` puts the two side by side and reports the gap.

TWO REGRESSION MODELS, TWO DIFFERENT QUESTIONS
----------------------------------------------
People argue about cause-specific vs Fine-Gray as though one is right. They
answer different questions and the choice follows from which question you asked:

  CAUSE-SPECIFIC HAZARD -- "among patients still alive and event-free, does
      treatment change the rate at which THIS event occurs?" This is the
      aetiological question. It is what you want for understanding mechanism.
      Fit by censoring competing events, which is legitimate HERE because the
      hazard is defined on the still-at-risk population by construction.

  FINE-GRAY SUBDISTRIBUTION -- "does treatment change the PROBABILITY that a
      patient will have experienced this event by time t?" This is the
      prognostic/allocation question. It is what you want for predicting
      absolute risk, deciding resource allocation, or telling a patient their
      chance. Its risk set deliberately KEEPS patients who had a competing
      event, which is why it maps directly onto the cumulative incidence.

A treatment can genuinely reduce the cause-specific hazard of cancer death and
LEAVE cumulative cancer-death incidence unchanged -- if it also keeps patients
alive longer so they survive to be at risk. Reporting one and interpreting it as
the other is how that gets missed.

RMST
----
Restricted mean survival time up to a horizon tau: the area under the survival
curve, in months. It needs NO proportional-hazards assumption, and it is in
units a clinician can hand to a patient ("on average, 4.1 more months alive over
the next 3 years") rather than a ratio of instantaneous rates. The earlier build
recommended RMST as the remedy when PH fails and did not implement it.

The price of RMST is that tau must be pre-specified, because choosing it after
seeing the curves is a garden-of-forking-paths problem: different tau give
different answers and the analyst can pick the flattering one.
"""

from __future__ import annotations

import numpy as np

from survival import cox_ph, kaplan_meier


# ---------------------------------------------------------------------------
# Cumulative incidence
# ---------------------------------------------------------------------------
def aalen_johansen(time, event_type, cause=1):
    """Cumulative incidence for `cause` in the presence of competing events.

    event_type: 0 = censored, 1 = cause of interest, 2+ = competing events.

    CIF(t) = sum over event times s <= t of  S(s-) * (d_cause(s) / n(s))

    where S is the OVERALL event-free survival (any cause). The S(s-) factor is
    the whole difference from 1-KM: you can only have the event at time s if you
    were still event-free just before it, and patients removed by a competing
    event are correctly no longer contributing.
    """
    time = np.asarray(time, dtype=float)
    event_type = np.asarray(event_type, dtype=int)
    order = np.argsort(time)
    time, event_type = time[order], event_type[order]

    times = np.unique(time[event_type > 0])
    cif, surv = [], 1.0
    running = 0.0
    for t in times:
        n_at_risk = int((time >= t).sum())
        if n_at_risk == 0:
            continue
        d_cause = int(((time == t) & (event_type == cause)).sum())
        d_any = int(((time == t) & (event_type > 0)).sum())
        running += surv * (d_cause / n_at_risk)
        cif.append(running)
        surv *= (1 - d_any / n_at_risk)          # overall event-free survival
    return times, np.array(cif)


def naive_one_minus_km(time, event_type, cause=1):
    """The WRONG estimator: treat competing events as censored, report 1-KM."""
    ev = (np.asarray(event_type) == cause).astype(int)
    t, s, _a, _d, _se = kaplan_meier(time, ev)
    return t, 1 - s


def compare_naive_vs_cif(time, event_type, cause=1, horizons=(12, 24, 36, 48)):
    """Quantify the overestimate at a set of horizons."""
    t_cif, cif = aalen_johansen(time, event_type, cause)
    t_naive, naive = naive_one_minus_km(time, event_type, cause)

    def at(ts, vals, h):
        idx = np.searchsorted(ts, h, side="right") - 1
        return float(vals[idx]) if idx >= 0 else 0.0

    rows = []
    for h in horizons:
        c, n = at(t_cif, cif, h), at(t_naive, naive, h)
        rows.append({"horizon": h, "cif": c, "naive_1_minus_km": n,
                     "absolute_overestimate": n - c,
                     "relative_overestimate": (n - c) / c if c > 0 else float("nan")})
    return rows, (t_cif, cif), (t_naive, naive)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def cause_specific_cox(time, event_type, X, cause=1):
    """Cox on the cause-specific hazard: competing events are CENSORED.

    Legitimate here because the cause-specific hazard is defined on the
    still-at-risk population, and a patient who died of another cause is
    genuinely no longer at risk of this one.
    """
    ev = (np.asarray(event_type) == cause).astype(int)
    return cox_ph(time, ev, X)


def _censoring_survival(time, event_type):
    """KM of the CENSORING distribution -- the reverse-KM trick again, reused
    here to build inverse-probability-of-censoring weights."""
    censored = (np.asarray(event_type) == 0).astype(int)
    t, s, _a, _d, _se = kaplan_meier(time, censored)
    return t, s


def fine_gray(time, event_type, X, cause=1, max_iter=40, tol=1e-8):
    """Fine-Gray subdistribution hazard model, IPCW-weighted.

    The one idea that matters: patients who experience a COMPETING event stay
    in the risk set afterwards, with a weight that decays by the probability of
    remaining uncensored. That is not a trick -- it is what makes the
    coefficient map onto the cumulative incidence function rather than onto the
    rate among survivors.

    Implementation note, stated honestly: this uses Breslow ties and a
    right-censoring-complete weighting, which is the standard simplification.
    A production implementation (cmprsk in R, lifelines' Fine-Gray) handles
    left-truncation and time-varying weights more carefully.
    """
    from math import erfc, log, sqrt
    time = np.asarray(time, dtype=float)
    event_type = np.asarray(event_type, dtype=int)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != len(time):
        X = X.T
    n, p = X.shape

    ct, cs = _censoring_survival(time, event_type)

    def g(t):
        """Censoring survival at t; 1.0 before the first censoring event."""
        if len(ct) == 0:
            return 1.0
        idx = np.searchsorted(ct, t, side="right") - 1
        return float(cs[idx]) if idx >= 0 else 1.0

    event_times = np.unique(time[event_type == cause])
    beta = np.zeros(p)

    for _ in range(max_iter):
        grad = np.zeros(p)
        hess = np.zeros((p, p))
        theta = np.exp(X @ beta)

        for t in event_times:
            # Subdistribution risk set: still event-free at t, PLUS those who
            # had a COMPETING event before t (downweighted).
            still = time >= t
            had_competing = (time < t) & (event_type > 0) & (event_type != cause)
            w = np.zeros(n)
            w[still] = 1.0
            gt = g(t)
            if had_competing.any() and gt > 0:
                w[had_competing] = np.array(
                    [gt / g(ti) if g(ti) > 0 else 0.0
                     for ti in time[had_competing]])
            w = np.clip(w, 0, 1)
            if w.sum() <= 0:
                continue

            wt = w * theta
            s0 = wt.sum()
            if s0 <= 0:
                continue
            s1 = (wt[:, None] * X).sum(0)
            s2 = (wt[:, None, None] * X[:, :, None] * X[:, None, :]).sum(0)
            zbar = s1 / s0

            d_idx = np.where((time == t) & (event_type == cause))[0]
            d = len(d_idx)
            for k in d_idx:
                grad += X[k]
            grad -= d * zbar
            hess -= d * (s2 / s0 - np.outer(zbar, zbar))

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
    return {"beta": beta, "se": se, "hr": np.exp(beta),
            "ci_low": np.exp(beta - 1.96 * se),
            "ci_high": np.exp(beta + 1.96 * se),
            "p": np.array([erfc(abs(b / s) / sqrt(2)) for b, s in zip(beta, se)])}


# ---------------------------------------------------------------------------
# RMST
# ---------------------------------------------------------------------------
def rmst(time, event, tau):
    """Restricted mean survival time up to tau: the area under the KM curve.

    tau must be PRE-SPECIFIED. Choosing it after seeing the curves is a
    garden-of-forking-paths problem -- different horizons give different
    answers and the analyst can pick the flattering one.
    """
    t, s, _a, _d, _se = kaplan_meier(time, event)
    t = np.concatenate([[0.0], t])
    s = np.concatenate([[1.0], s])
    keep = t <= tau
    t_k, s_k = t[keep], s[keep]
    # step function: rectangle widths between consecutive event times
    widths = np.diff(np.concatenate([t_k, [tau]]))
    return float(np.sum(s_k * widths))


def rmst_difference(time, event, group, tau, n_boot=400, seed=0):
    """RMST difference between two arms, with a bootstrap CI.

    No proportional-hazards assumption is required, which is the point: when PH
    fails, a hazard ratio is a weighted average of something that changed, and
    an RMST difference in months is still exactly what it says.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    group = np.asarray(group, dtype=int)
    a = rmst(time[group == 1], event[group == 1], tau)
    b = rmst(time[group == 0], event[group == 0], tau)
    point = a - b

    rng = np.random.default_rng(seed)
    diffs = []
    n = len(time)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        g = group[idx]
        if g.sum() < 5 or (1 - g).sum() < 5:
            continue
        try:
            diffs.append(rmst(time[idx][g == 1], event[idx][g == 1], tau)
                         - rmst(time[idx][g == 0], event[idx][g == 0], tau))
        except (IndexError, ValueError):
            continue
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"tau": tau, "rmst_treatment": a, "rmst_control": b,
            "difference": point, "lo": float(lo), "hi": float(hi),
            "significant": bool(lo > 0 or hi < 0), "n_boot": len(diffs)}
