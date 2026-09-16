"""
Stage 6.4: Prescriptive Decision Layer - the cheapest, clearest
capability gap identified early in this project's research. Takes
SOH, RUL, second-life grade (session 25's method, GRADE thresholds
reused unchanged from run_stage4_step3_grading.py), and degradation-
mode signature (session 23's method, run_degradation_mode_analysis.py)
and produces ONE plain-language recommendation from a small, fixed
set, with the reasoning stated explicitly as a list of the specific
rules that fired - not a black box, and not a new model: every input
is already a real, separately-computed signal this project already
produces; this layer only combines them transparently.

NOT deployed to the live app in this stage - implemented and verified
only, per instruction.
"""

from __future__ import annotations
from dataclasses import dataclass, field

RECOMMENDATIONS = [
    "Continue normal use",
    "Monitor closely",
    "Candidate for second-life",
    "Recommend retirement",
]

SOH_PRIMARY_EV = 80.0
SOH_SECOND_LIFE = 50.0
RUL_LOW_CYCLES = 50
HIGH_RISK_MODES = {"LAM-leaning signature (peak height collapsed, position relatively stable)",
                    "mixed LLI+LAM-leaning signature"}


@dataclass
class Recommendation:
    action: str
    reasoning: list[str] = field(default_factory=list)


def recommend(soh: float, rul: float | None, second_life_grade: str,
              degradation_mode: str | None = None) -> Recommendation:
    reasoning = [f"SOH={soh:.1f}%, RUL={rul if rul is not None else 'unknown'} cycles, "
                 f"second-life grade='{second_life_grade}', degradation mode='{degradation_mode or 'unknown'}'"]

    # rule 1: hard SOH floor - matches session 25's own "Recycle only" threshold
    if soh < SOH_SECOND_LIFE:
        reasoning.append(f"SOH {soh:.1f}% is below the {SOH_SECOND_LIFE:.0f}% second-life floor "
                          f"(session 25's own 'Recycle only' threshold) - retirement is the only "
                          f"safe recommendation regardless of any other signal.")
        return Recommendation("Recommend retirement", reasoning)

    # rule 2: second-life band
    if soh < SOH_PRIMARY_EV or second_life_grade == "Second-life candidate":
        reasoning.append(f"SOH {soh:.1f}% falls in the second-life band "
                          f"[{SOH_SECOND_LIFE:.0f}%, {SOH_PRIMARY_EV:.0f}%) and/or session 25's grading "
                          f"already labels it '{second_life_grade}' - candidate for second-life "
                          f"redeployment rather than continued primary use.")
        if degradation_mode in HIGH_RISK_MODES:
            reasoning.append(f"CAVEAT: degradation-mode signature ('{degradation_mode}') indicates an "
                              f"active LAM-leaning (loss-of-active-material) mechanism, which this "
                              f"project's own research (B0018/B0044/B0045) has repeatedly found "
                              f"associated with FASTER, harder-to-predict further fade than a pure-LLI "
                              f"signature would suggest - flag for tighter-than-usual second-life "
                              f"monitoring, not a blanket 'safe as-is' second-life certification.")
        return Recommendation("Candidate for second-life", reasoning)

    # rule 3: SOH >= 80% (session 25's "Primary EV use" band) - still check RUL/degradation mode
    reasoning.append(f"SOH {soh:.1f}% is in the 'Primary EV use' band (>= {SOH_PRIMARY_EV:.0f}%, "
                      f"session 25's grading agrees: '{second_life_grade}').")
    flags = []
    if rul is not None and rul < RUL_LOW_CYCLES:
        flags.append(f"RUL={rul} cycles is low (< {RUL_LOW_CYCLES}) despite the high SOH grade - "
                      f"a real risk of near-term rapid decline, not visible from SOH alone.")
    if degradation_mode in HIGH_RISK_MODES:
        flags.append(f"degradation-mode signature ('{degradation_mode}') indicates an active LAM-"
                      f"leaning mechanism - this project's own history (B0018 specifically) has an "
                      f"established, documented case of a battery graded 'Primary EV use' on predicted "
                      f"SOH alone that was ACTUALLY already in the second-life band on its true SOH - "
                      f"this exact combination (high predicted SOH + LAM-leaning signature) is the "
                      f"known failure signature, not a hypothetical one.")
    if flags:
        reasoning.append("However, the following risk flag(s) fired despite the high SOH grade:")
        reasoning += [f"  - {f}" for f in flags]
        reasoning.append("Recommendation downgraded to closer monitoring rather than trusting the "
                          "SOH grade alone, given this project's own documented precedent for exactly "
                          "this combination of signals being unreliable.")
        return Recommendation("Monitor closely", reasoning)

    reasoning.append("No risk flags fired - SOH grade, RUL, and degradation mode are all consistent "
                      "with continued normal use.")
    return Recommendation("Continue normal use", reasoning)
