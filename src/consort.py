"""CONSORT participant flow.

WHAT CONSORT IS FOR, AND WHY A SIMULATED ONE IS ALMOST A LIE
------------------------------------------------------------
The CONSORT diagram exists to make participant loss visible. Its value is
entirely in the numbers it is uncomfortable to publish: how many were screened
and excluded, how many were randomised and then never treated, how many
discontinued and for what reason, and how many were analysed. A trial that
loses 30% of one arm to withdrawal has a result that means something different
from one that loses 3%, and the diagram is where that becomes unavoidable.

None of that is observed here. This trial is simulated, so every attrition
number below is one I CHOSE. A generated CONSORT diagram cannot demonstrate
that a trial retained its participants; it can only demonstrate that the code
knows the shape of the accounting.

So what is actually being built is the INVARIANT, which does generalise:

    every randomised participant appears in exactly one terminal state, and
    the arithmetic closes at every stage.

That is the property a real CONSORT diagram must satisfy and the one that gets
violated in practice -- by participants counted in two discontinuation
categories, by an analysis population larger than the randomised population
after a protocol deviation is reclassified, by a denominator that quietly
changes between the efficacy and safety tables. `validate()` refuses to emit a
diagram whose numbers do not close, which is the only part of this file that
would still be worth having on real data.

THE ITT / PER-PROTOCOL DISTINCTION
----------------------------------
Reported explicitly because it is where the analysis population is most often
misstated. INTENTION TO TREAT analyses every randomised participant in the arm
they were ASSIGNED, regardless of what they received; it preserves
randomisation and is the primary analysis for efficacy in a superiority trial.
PER PROTOCOL analyses only those who adhered, which breaks randomisation --
adherence is a post-randomisation outcome, and conditioning on it reintroduces
exactly the confounding the trial was designed to remove.

Per-protocol is not "the cleaner analysis". It is the biased one, and it is
reported here as a sensitivity analysis with that stated.

WHAT THIS IS NOT
----------------
No screening/enrolment log, no eligibility criteria, no site structure, no
randomisation schedule or allocation concealment, no blinding, no protocol
document, and no adverse-event accounting -- the safety population is a
different denominator again and is not modelled.
"""

from __future__ import annotations


class ConsortError(ValueError):
    """Raised when the participant accounting does not close."""


def build_flow(assessed, excluded_reasons, arms):
    """Assemble a CONSORT flow.

    `arms` is {arm_label: {"allocated": n, "not_treated": n,
                           "discontinued": {reason: n}, "analysed_itt": n,
                           "analysed_pp": n}}
    """
    excluded = sum(excluded_reasons.values())
    randomised = assessed - excluded
    flow = {"assessed": assessed, "excluded": excluded,
            "excluded_reasons": dict(excluded_reasons),
            "randomised": randomised, "arms": {}}
    for label, a in arms.items():
        disc = dict(a.get("discontinued", {}))
        flow["arms"][label] = {
            "allocated": a["allocated"],
            "not_treated": a.get("not_treated", 0),
            "discontinued": disc,
            "discontinued_total": sum(disc.values()),
            "completed": (a["allocated"] - a.get("not_treated", 0)
                          - sum(disc.values())),
            "analysed_itt": a["analysed_itt"],
            "analysed_pp": a["analysed_pp"],
        }
    return flow


def validate(flow):
    """Every randomised participant in exactly one terminal state.

    Returns a list of problems. Empty means the arithmetic closes.

    Each check corresponds to a way real trial reports break, not to a way this
    generator might:

      * allocation not summing to randomised -- someone randomised and then
        dropped from the diagram entirely, which is the single most serious
        CONSORT failure because it is invisible in the published numbers;
      * ITT larger than allocated -- an analysis population that grew, usually
        from a reclassification applied after unblinding;
      * per-protocol larger than ITT -- impossible by construction, and a sign
        that two different denominators have been mixed;
      * negative completion -- discontinuations exceeding allocation, from
        double-counting a participant in two reasons.
    """
    problems = []
    total_alloc = sum(a["allocated"] for a in flow["arms"].values())
    if total_alloc != flow["randomised"]:
        problems.append(
            f"allocation does not close: {total_alloc} allocated across arms "
            f"but {flow['randomised']} randomised. A participant randomised "
            f"and then absent from the diagram is the most serious CONSORT "
            f"failure, because nothing downstream shows it.")
    if flow["assessed"] - flow["excluded"] != flow["randomised"]:
        problems.append("screening does not close: assessed - excluded != "
                        "randomised")
    for label, a in flow["arms"].items():
        if a["completed"] < 0:
            problems.append(
                f"{label}: discontinuations ({a['discontinued_total']}) plus "
                f"never-treated ({a['not_treated']}) exceed allocation "
                f"({a['allocated']}) -- a participant is counted in two "
                f"terminal states")
        if a["analysed_itt"] > a["allocated"]:
            problems.append(
                f"{label}: ITT population ({a['analysed_itt']}) exceeds "
                f"allocation ({a['allocated']}). An analysis population cannot "
                f"grow after randomisation.")
        if a["analysed_pp"] > a["analysed_itt"]:
            problems.append(
                f"{label}: per-protocol ({a['analysed_pp']}) exceeds ITT "
                f"({a['analysed_itt']}), which is impossible by construction "
                f"and means two denominators have been mixed")
    return problems


def render_text(flow):
    """The diagram, as text. ASCII because a PNG is not more auditable."""
    L = []
    L.append(f"{'Assessed for eligibility':<44} n = {flow['assessed']:,}")
    L.append(f"{'  Excluded':<44} n = {flow['excluded']:,}")
    for reason, n in flow["excluded_reasons"].items():
        L.append(f"{'    ' + reason:<44} n = {n:,}")
    L.append(f"{'Randomised':<44} n = {flow['randomised']:,}")
    L.append("")
    for label, a in flow["arms"].items():
        L.append(f"  {label}")
        L.append(f"{'    Allocated to intervention':<44} n = {a['allocated']:,}")
        L.append(f"{'      Did not receive intervention':<44} "
                 f"n = {a['not_treated']:,}")
        for reason, n in a["discontinued"].items():
            L.append(f"{'      Discontinued: ' + reason:<44} n = {n:,}")
        L.append(f"{'    Completed follow-up':<44} n = {a['completed']:,}")
        L.append(f"{'    Analysed (ITT)':<44} n = {a['analysed_itt']:,}")
        L.append(f"{'    Analysed (per protocol)':<44} n = {a['analysed_pp']:,}")
        L.append("")
    return "\n".join(L)
