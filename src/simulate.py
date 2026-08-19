"""Simulated trial data with a KNOWN true hazard ratio.

Simulation with known ground truth is the strongest available option here and
the spec says so. It is not a fallback for lacking a real dataset -- it is what
makes the methods VERIFIABLE. On Rotterdam or SUPPORT you can run a Cox model
and get a plausible number; here you can ask whether the estimator recovered
the coefficient you wrote down, which is a different and much stronger claim
for a hand-rolled implementation.

TWO DATASETS ARE GENERATED
--------------------------

`simulate_trial()` -- a clean randomised trial with a known HR, exponential
    baseline hazard, administrative censoring at end of study, and random
    dropout. Used to verify that KM, log-rank and Cox recover the truth.

`simulate_immortal_time()` -- the centrepiece. A cohort where treatment is
    given at a RANDOM TIME AFTER baseline, and treatment has NO EFFECT
    WHATSOEVER (true HR = 1.00). The naive analysis -- group by "ever
    treated" and compare from baseline -- will nonetheless show a large
    apparent benefit, because to be treated at day 90 you must first survive
    to day 90. That survival time is 'immortal': it is guaranteed
    event-free by the design of the analysis, and the naive grouping credits
    it to the treatment.

    This is not a toy. It is the mechanism behind a long series of
    observational findings that later failed in trials, and it is the
    shibboleth in the spec: a data scientist who knows it has worked adjacent
    to real clinical analysis, and one who does not produces confident
    garbage.
"""

from __future__ import annotations

import numpy as np

# The truths the analysis has to recover.
TRUE_HR_TREATMENT = 0.70
TRUE_HR_AGE_PER_DECADE = 1.25
TRUE_HR_STAGE = 1.80
BASELINE_HAZARD = 0.045          # per month
STUDY_MONTHS = 48.0

# For the immortal-time cohort: treatment genuinely does nothing.
TRUE_HR_IMMORTAL = 1.00


def simulate_trial(n=900, seed=31):
    """A randomised trial. Returns a dict of arrays plus the truth."""
    rng = np.random.default_rng(seed)

    arm = rng.integers(0, 2, n)                        # 1 = treatment
    age = np.clip(rng.normal(62, 11, n), 30, 90)
    stage = rng.integers(0, 2, n)                      # 1 = advanced
    sex = rng.integers(0, 2, n)
    ecog = rng.integers(0, 3, n)

    log_hr = (np.log(TRUE_HR_TREATMENT) * arm
              + np.log(TRUE_HR_AGE_PER_DECADE) * (age - 62) / 10
              + np.log(TRUE_HR_STAGE) * stage)
    hazard = BASELINE_HAZARD * np.exp(log_hr)
    event_time = rng.exponential(1 / hazard)

    # dropout (random censoring) and administrative censoring at study end
    dropout = rng.exponential(1 / 0.010, n)
    enrol = rng.uniform(0, 12, n)                      # staggered entry
    admin = STUDY_MONTHS - enrol

    obs = np.minimum(np.minimum(event_time, dropout), admin)
    event = (event_time <= np.minimum(dropout, admin)).astype(int)

    return {
        "time": obs, "event": event, "arm": arm, "age": age, "stage": stage,
        "sex": sex, "ecog": ecog, "enrol": enrol,
        "truth": {"hr_treatment": TRUE_HR_TREATMENT,
                  "hr_age_per_decade": TRUE_HR_AGE_PER_DECADE,
                  "hr_stage": TRUE_HR_STAGE,
                  "baseline_hazard": BASELINE_HAZARD},
    }


def simulate_immortal_time(n=1200, seed=37):
    """A cohort where treatment has NO effect and is given at a random later time.

    Returns baseline covariates, the true event time, the treatment time (inf
    for the never-treated), and the observed follow-up.

    The generation is deliberately blunt: `hazard` does not depend on treatment
    at all. Any apparent benefit in the analysis is therefore entirely an
    artefact of how the analysis was set up, with no possibility of it being a
    real effect the simulation smuggled in.
    """
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(64, 10, n), 35, 92)
    stage = rng.integers(0, 2, n)

    log_hr = (np.log(TRUE_HR_AGE_PER_DECADE) * (age - 64) / 10
              + np.log(TRUE_HR_STAGE) * stage)
    hazard = BASELINE_HAZARD * np.exp(log_hr)          # NO treatment term
    event_time = rng.exponential(1 / hazard)
    dropout = rng.exponential(1 / 0.008, n)
    admin = np.full(n, STUDY_MONTHS)
    censor_time = np.minimum(dropout, admin)

    # Treatment is initiated at a random time; ~55% are ever treated, and they
    # can only be treated if they are still alive and in follow-up when the
    # opportunity arrives.
    would_start = rng.exponential(1 / 0.035, n)
    ever_offered = rng.random(n) < 0.80
    treat_time = np.where(ever_offered, would_start, np.inf)
    observed_end = np.minimum(event_time, censor_time)
    treated = treat_time < observed_end                # actually got it in time
    treat_time = np.where(treated, treat_time, np.inf)

    obs = observed_end
    event = (event_time <= censor_time).astype(int)

    return {
        "time": obs, "event": event, "age": age, "stage": stage,
        "treat_time": treat_time, "ever_treated": treated.astype(int),
        "truth": {"hr_treatment": TRUE_HR_IMMORTAL,
                  "hr_age_per_decade": TRUE_HR_AGE_PER_DECADE,
                  "hr_stage": TRUE_HR_STAGE},
    }


def to_counting_process(data):
    """Split each subject at their treatment time into (start, stop] intervals.

    This is the correct handling of a time-dependent exposure, and it is the
    whole fix. Before treatment starts a subject contributes UNTREATED
    person-time; after it starts they contribute TREATED person-time. Nobody
    contributes treated person-time before they were treated, which is exactly
    what the naive ever-treated grouping does.

    Returns arrays suitable for a Cox model on (start, stop, event, treated).
    """
    starts, stops, events, treated, age, stage = [], [], [], [], [], []
    for i in range(len(data["time"])):
        t_end = data["time"][i]
        tt = data["treat_time"][i]
        if tt < t_end:
            starts.append(0.0)
            stops.append(tt)
            events.append(0)
            treated.append(0)
            age.append(data["age"][i])
            stage.append(data["stage"][i])

            starts.append(tt)
            stops.append(t_end)
            events.append(int(data["event"][i]))
            treated.append(1)
            age.append(data["age"][i])
            stage.append(data["stage"][i])
        else:
            starts.append(0.0)
            stops.append(t_end)
            events.append(int(data["event"][i]))
            treated.append(0)
            age.append(data["age"][i])
            stage.append(data["stage"][i])
    return {"start": np.array(starts), "stop": np.array(stops),
            "event": np.array(events), "treated": np.array(treated),
            "age": np.array(age), "stage": np.array(stage)}


def cox_counting_process(cp, max_iter=50, tol=1e-9):
    """Cox model on (start, stop] intervals, Breslow ties.

    Written separately from `survival.cox_ph` because the risk set is different:
    at event time t the risk set is every interval with start < t <= stop, not
    every subject with time >= t. That single change is what makes a
    time-dependent covariate legitimate.
    """
    from math import erfc, log, sqrt
    start, stop = cp["start"], cp["stop"]
    event = cp["event"]
    X = np.column_stack([cp["treated"], (cp["age"] - 64) / 10, cp["stage"]])
    p = X.shape[1]
    beta = np.zeros(p)

    event_times = np.unique(stop[event == 1])
    for _ in range(max_iter):
        grad = np.zeros(p)
        hess = np.zeros((p, p))
        loglik = 0.0
        theta = np.exp(X @ beta)
        for t in event_times:
            at_risk = (start < t) & (stop >= t)
            if not at_risk.any():
                continue
            w = theta[at_risk]
            Xr = X[at_risk]
            s0 = w.sum()
            s1 = (w[:, None] * Xr).sum(0)
            s2 = (w[:, None, None] * Xr[:, :, None] * Xr[:, None, :]).sum(0)
            zbar = s1 / s0
            d_idx = np.where((stop == t) & (event == 1))[0]
            d = len(d_idx)
            for k in d_idx:
                loglik += X[k] @ beta
                grad += X[k]
            loglik -= d * log(s0)
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
            "p": np.array([erfc(abs(b / s) / sqrt(2)) for b, s in zip(beta, se)]),
            "loglik": loglik}


# ---------------------------------------------------------------------------
# Competing risks
# ---------------------------------------------------------------------------
# Truths for the competing-risks cohort. Treatment reduces the CAUSE-SPECIFIC
# hazard of the event of interest and does NOTHING to the competing hazard --
# the configuration that separates the two regression models, because keeping
# patients alive longer leaves them at risk for longer and blunts the effect on
# cumulative incidence.
TRUE_HR_CAUSE_SPECIFIC = 0.60
TRUE_HR_COMPETING = 1.00
BASELINE_CAUSE = 0.030          # per month, event of interest
BASELINE_COMPETING = 0.028      # per month, competing death


def simulate_competing_risks(n=1600, seed=53):
    """Two competing causes. Returns event_type: 0 censored, 1 cause, 2 competing.

    The competing hazard is deliberately LARGE relative to the cause of
    interest, because that is when 1-KM goes most wrong -- and it is the
    realistic case in an older population, where most people die of something
    other than the disease being studied.
    """
    rng = np.random.default_rng(seed)
    arm = rng.integers(0, 2, n)
    age = np.clip(rng.normal(70, 9, n), 45, 95)

    h_cause = (BASELINE_CAUSE * np.exp(np.log(TRUE_HR_CAUSE_SPECIFIC) * arm
                                       + 0.020 * (age - 70)))
    h_comp = (BASELINE_COMPETING * np.exp(np.log(TRUE_HR_COMPETING) * arm
                                          + 0.055 * (age - 70)))

    t_cause = rng.exponential(1 / h_cause)
    t_comp = rng.exponential(1 / h_comp)
    admin = np.full(n, STUDY_MONTHS)
    dropout = rng.exponential(1 / 0.004, n)
    censor = np.minimum(admin, dropout)

    obs = np.minimum(np.minimum(t_cause, t_comp), censor)
    event_type = np.where((t_cause <= t_comp) & (t_cause <= censor), 1,
                 np.where((t_comp < t_cause) & (t_comp <= censor), 2, 0))

    return {"time": obs, "event_type": event_type, "arm": arm, "age": age,
            "truth": {"hr_cause_specific": TRUE_HR_CAUSE_SPECIFIC,
                      "hr_competing": TRUE_HR_COMPETING,
                      "baseline_cause": BASELINE_CAUSE,
                      "baseline_competing": BASELINE_COMPETING}}
