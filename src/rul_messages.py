"""Plain-text RUL wording for the app, kept free of Streamlit so tests can check the exact sentences.

RUL is shown only for unflagged NASA/MIT browse batteries. Everything else (other sources, flagged batteries, uploads)
gets the neutral not-shown message. Figures: PAPER_RESULTS.md (in-domain RUL R2 0.374; CALCE MAE 421.5 cycles, R2 -566.35).
"""

RUL_SHOWN_CAPTION = ("Computed offline from the stored per-cycle table and verified identical to the live computation "
                     "(max difference 0 cycles over 28 pairs). In-domain RUL R2 is about 0.37.")
RUL_LIVE_CAPTION = ("Computed live from this battery's raw cycle curve with the deployed joint model. "
                    "In-domain RUL R2 is about 0.37.")
RUL_NOT_SHOWN_TEXT = ("RUL is not shown for this battery. The deployed RUL model fails on batteries unlike NASA/MIT "
                      "(on CALCE: MAE about 421 cycles, R2 about -566), so SOH is the headline output.")
RUL_NOT_SHOWN_FLAGGED_PREFIX = "This battery was flagged as unfamiliar. "
RUL_NOT_SHOWN_UPLOAD_PREFIX = "RUL is not offered for uploaded data. "


def rul_state(ctx: dict) -> str:
    """'shown' | 'flagged' | 'upload' | 'other' for a prediction context."""
    if ctx.get("is_upload"):
        return "upload"
    if not ctx.get("rul_hidden", ctx.get("out_of_domain", True)) and ctx.get("rul_pred") is not None:
        return "shown"
    return "flagged" if ctx.get("out_of_domain") else "other"


def rul_not_shown_message(state: str) -> str:
    if state == "flagged":
        return RUL_NOT_SHOWN_FLAGGED_PREFIX + RUL_NOT_SHOWN_TEXT
    if state == "upload":
        return RUL_NOT_SHOWN_UPLOAD_PREFIX + RUL_NOT_SHOWN_TEXT
    return RUL_NOT_SHOWN_TEXT


def rul_shown_caption(ctx: dict) -> str:
    return RUL_SHOWN_CAPTION if ctx.get("precomputed_fallback") else RUL_LIVE_CAPTION
