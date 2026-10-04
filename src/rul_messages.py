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


# ---- "Why isn't RUL shown here?" expander (plain text + one table; no LLM) ----------------------------------------------------------------
EVALUATED_EXTERNAL = ("CALCE", "Oxford", "HUST", "XJTU")
WHY_TITLE = "Why isn't RUL shown here?"
WHY_TRAINED = ("The RUL model was trained only on NASA and MIT cells, and it is only reliable on familiar NASA/MIT batteries.")
WHY_FLAGGED = "This battery was flagged as unfamiliar by the trust check, so RUL is hidden."
WHY_NOT_EVALUATED = "RUL was not evaluated on this source. All four external sources that were tested failed, so we do not show one."
WHY_REASONS = (
    "The model learned an average lifetime and predicts roughly the same remaining life whatever the battery. Example, MIT battery b1c4: "
    "the true RUL counts down from 1224 to 0 cycles, while the predictions stay in a narrow band around 600 cycles (middle half: 503 to 724).",
    "On CALCE its predictions average -401 cycles, which is impossible for a remaining life.",
    "The deployed site also has no raw curves for these sources, so RUL cannot be computed live there.",
)
WHY_INSTEAD = ("Use the SOH value with its 90% interval and the trust state instead, and check against a measured capacity when possible.")


def rul_crossdomain_table(out_dir):
    """Rows for the report's RUL cross-domain table, read from the same CSVs (stage4_step2b_summary.csv for in-domain,
    finalpass_item5c_rul_crossdomain.csv for the four external sources). In-domain MAE is not recorded in either file."""
    import pandas as pd
    out_dir = __import__("pathlib").Path(out_dir)
    s = pd.read_csv(out_dir / "stage4_step2b_summary.csv").set_index("metric")["value"]
    x = pd.read_csv(out_dir / "finalpass_item5c_rul_crossdomain.csv").set_index("dataset")
    rows = [{"Dataset": "In-domain (NASA+MIT test batteries)", "MAE (cycles)": None, "RMSE (cycles)": float(s["rul_joint_rmse"]), "R2": float(s["rul_joint_r2"])}]
    for ds in EVALUATED_EXTERNAL:
        rows.append({"Dataset": ds, "MAE (cycles)": float(x.loc[ds, "rul_mae"]), "RMSE (cycles)": float(x.loc[ds, "rul_rmse"]), "R2": float(x.loc[ds, "rul_r2"])})
    return pd.DataFrame(rows)


def why_not_shown_content(ctx: dict, dataset: str | None) -> dict | None:
    """None when RUL is shown. Otherwise {"paragraphs": [...], "highlight": dataset-or-None, "show_table": True}."""
    state = rul_state(ctx)
    if state == "shown":
        return None
    paras = [WHY_TRAINED]
    if state == "flagged":
        paras.append(WHY_FLAGGED)
    highlight = dataset if dataset in EVALUATED_EXTERNAL else None
    if highlight is None and state != "flagged":
        paras.append(WHY_NOT_EVALUATED)
    paras.extend(WHY_REASONS)
    paras.append(WHY_INSTEAD)
    return {"paragraphs": paras, "highlight": highlight}
