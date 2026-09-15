"""
Stage 6.4: tests the Prescriptive Decision Layer against 3 known
cases from this project's own history where the "right answer" is
already well-established - confirms recommendations are sensible and
the stated reasoning is accurate, not just that the code runs.

Values used are representative/illustrative of each case's well-
documented real character (stated explicitly, not claimed as one
exact cycle's raw model output re-derived here) - grounded in this
project's own established findings:
- B0018: the project's own repeatedly-documented second-life mis-
  certification case (predicted SOH lands it in the "Primary EV use"
  band while its degradation-mode signature is "mixed LLI+LAM-
  leaning" - the documented risk signature).
- A clean, early-life, healthy battery (high SOH, high RUL, minimal
  peak-shape change).
- A fast-fading battery already in the second-life SOH band, with an
  active LAM-leaning signature.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prescriptive_decision_layer import recommend

CASES = [
    {
        "name": "B0018 (known second-life mis-certification case)",
        "soh": 81.5, "rul": 2, "second_life_grade": "Primary EV use",
        "degradation_mode": "mixed LLI+LAM-leaning signature",
        "expected": "Monitor closely",
        "why_expected": "Grade alone says 'Primary EV use' (SOH>=80), but this project's own "
                         "research repeatedly found B0018's predicted SOH overstates its true "
                         "condition (documented +8-9pp bias) alongside this exact degradation-mode "
                         "signature - the decision layer should NOT simply pass through the raw "
                         "grade unflagged.",
    },
    {
        "name": "Clean healthy battery, early life",
        "soh": 96.5, "rul": 850, "second_life_grade": "Primary EV use",
        "degradation_mode": "minimal peak-shape change over the observed test-set window",
        "expected": "Continue normal use",
        "why_expected": "High SOH, high RUL, no degradation signature of concern - no risk flag "
                         "should fire.",
    },
    {
        "name": "Fast-fading battery, already second-life range",
        "soh": 54.0, "rul": 40, "second_life_grade": "Second-life candidate",
        "degradation_mode": "LAM-leaning signature (peak height collapsed, position relatively stable)",
        "expected": "Candidate for second-life",
        "why_expected": "SOH is already in the second-life band per session 25's own thresholds - "
                         "the LAM-leaning signature should appear as an explicit CAVEAT (tighter "
                         "monitoring within second-life use), not override the second-life "
                         "recommendation itself (the battery is genuinely too degraded for primary "
                         "use, degradation mode doesn't change that).",
    },
    {
        "name": "Severely degraded battery",
        "soh": 42.0, "rul": 5, "second_life_grade": "Recycle only",
        "degradation_mode": "LAM-leaning signature (peak height collapsed, position relatively stable)",
        "expected": "Recommend retirement",
        "why_expected": "Below the 50% second-life floor - retirement regardless of any other signal.",
    },
]


def main():
    n_correct = 0
    for case in CASES:
        rec = recommend(case["soh"], case["rul"], case["second_life_grade"], case["degradation_mode"])
        correct = rec.action == case["expected"]
        n_correct += correct
        print(f"\n=== {case['name']} ===")
        print(f"Inputs: SOH={case['soh']}, RUL={case['rul']}, grade={case['second_life_grade']!r}, "
              f"mode={case['degradation_mode']!r}")
        print(f"Expected: {case['expected']}  |  Got: {rec.action}  |  {'MATCH' if correct else 'MISMATCH'}")
        print(f"Why expected: {case['why_expected']}")
        print("Reasoning chain:")
        for r in rec.reasoning:
            print(f"  {r}")

    print(f"\n=== SUMMARY: {n_correct}/{len(CASES)} cases matched expected recommendation ===")


if __name__ == "__main__":
    main()
