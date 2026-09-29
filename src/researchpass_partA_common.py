"""
Shared utilities for Part A of this research pass (5 external-method
upgrades tested against the project's CURRENT DEPLOYED baseline, which
as of today is dataset-aware routing between two already-trained,
already-verified XGBoost-fusion models - see DEVELOPMENT_LOG.md's
"Dataset-aware routing" entry). Reused by items 1/3/4 (tabular HI+
fusion feature pipeline) so all three score against the IDENTICAL
routed reference numbers and reuse the IDENTICAL data-loading/eval
code, the same discipline stage1_common.py/stage7_common.py already
established for earlier stages. Items 2/5 use stage7_common's raw
per-cycle tensor pipeline instead (different representation, imported
directly from stage7_common in those items' own scripts).

Loads the two ALREADY-TRAINED, already-verified production models
(models/xgb_soh_fusion.json, models/_experimental_xgb_soh_fusion_
extended_reformulation.json) - never retrains them. Imputation medians
are recomputed fresh from the train-pool split for whichever feature
set is needed (base 8-feature vs. extended 8-feature), the same
np.nanmedian(X[train_mask]) computation fit_xgb performs internally -
not loaded from a cached file, so this stays correct even if a feature
list changes, and costs under a second.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from stage1_common import (
    canonical_feature_cols, load_nasa_mit_pool, battery_split_masks,
    fusion_cols, build_calce_merged, add_reformulated_duration_features,
    OUT_DIR, PROC_DIR, ROOT,
)
from stage5_extended_reformulation import add_scv_matd_viect_reformulated, extended_canonical_feature_cols

MODELS_DIR = ROOT / "models"

# The CURRENT deployed, ROUTED baseline (per DATASET IDENTITY - the
# project's own already-shipped routing in src/live_inference.py):
# NASA/MIT and XJTU use the base model; CALCE/Oxford/HUST use the
# extended-reformulation model. These are each model's own independently
# -verified numbers (audit_true_deployed_baseline.csv /
# stage5_extended_reformulation_eval.csv), mixed exactly the way a real
# user of the live app sees them today, dataset by dataset. THIS is the
# comparison point for every item in this research pass, per instruction.
ROUTED_BASELINE = {
    "in-domain (fixed split)": 0.974,
    "CALCE": 0.740,
    "Oxford": 0.953,
    "HUST": 0.800,
    "XJTU": -1.062,
}
EXTENDED_ROUTED_DATASETS = {"CALCE", "Oxford", "HUST"}  # matches live_inference.py exactly


def load_base_pool_and_split():
    """NASA+MIT pool (Stage 1.1 duration-reformulated), train/test masks,
    and the full (all-dataset) hi table - needed by every tabular item to
    build both the base and extended feature representations."""
    merged_nm, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged_nm)
    return merged_nm, hi_full, train_mask, test_mask, split


def load_extended_pool_and_split():
    """Same pool, with Stage 1.1's duration reformulation AND Stage 5's
    SCV/MATD/VIECT/MET reformulation applied on top - the representation
    the ALREADY-DEPLOYED extended model (CALCE/Oxford/HUST route)
    actually uses (mirrors run_stage5_extended_reformulation_eval.py's
    own step 1 exactly: duration reformulation first, then extended)."""
    hi_full = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_full = add_reformulated_duration_features(hi_full)
    hi_full_ext = add_scv_matd_viect_reformulated(hi_full)
    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    nasa_mit_ext = hi_full_ext[hi_full_ext["dataset"].isin(["NASA", "MIT"])]
    merged_ext = pd.merge(nasa_mit_ext, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    train_mask, test_mask, split = battery_split_masks(merged_ext)
    return merged_ext, hi_full_ext, hi_full, train_mask, test_mask, split


def load_all_heldout_base(hi_full: pd.DataFrame) -> dict:
    """CALCE/Oxford/HUST/XJTU tables in the BASE (Stage-1.1-only)
    representation - CALCE built fresh via build_calce_merged (own fusion
    embeddings), Oxford/HUST/XJTU read from the Stage 5.1 precomputed
    parquet files (already merged with fusion embeddings there)."""
    return {
        "CALCE": build_calce_merged(hi_full),
        "Oxford": pd.read_parquet(PROC_DIR / "stage5_1_oxford_merged.parquet"),
        "HUST": pd.read_parquet(PROC_DIR / "stage5_1_hust_merged.parquet"),
        "XJTU": pd.read_parquet(PROC_DIR / "stage5_1_xjtu_merged.parquet"),
    }


def load_all_heldout_extended(hi_full_unreformulated: pd.DataFrame) -> dict:
    """Same 4 sets, with the extended SCV/MATD/VIECT/MET reformulation
    applied - the representation the already-deployed extended model
    uses for CALCE/Oxford/HUST."""
    base = load_all_heldout_base(hi_full_unreformulated)
    return {name: add_scv_matd_viect_reformulated(df) for name, df in base.items()}


def base_feature_cols():
    return canonical_feature_cols(reformulated=True) + ["cycle_idx"]


def extended_feature_cols():
    return extended_canonical_feature_cols(canonical_feature_cols(reformulated=True)) + ["cycle_idx"]


def fit_medians(merged: pd.DataFrame, train_mask: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    cols = feature_cols + fusion_cols()
    X = merged[cols].to_numpy(dtype=float, copy=True)
    X = np.where(np.isinf(X), np.nan, X)
    return np.nanmedian(X[train_mask], axis=0)


def load_base_model() -> XGBRegressor:
    m = XGBRegressor()
    m.load_model(str(MODELS_DIR / "xgb_soh_fusion.json"))
    return m


def load_extended_model() -> XGBRegressor:
    m = XGBRegressor()
    m.load_model(str(MODELS_DIR / "_experimental_xgb_soh_fusion_extended_reformulation.json"))
    return m


def build_X(df: pd.DataFrame, feature_cols: list[str], medians: np.ndarray) -> np.ndarray:
    cols = feature_cols + fusion_cols()
    X = df[cols].to_numpy(dtype=float, copy=True)
    X = np.where(np.isinf(X), np.nan, X)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])
    return X


def score(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "n": int(len(y_true)),
    }


def verdict(r2: float, dataset: str) -> str:
    base = ROUTED_BASELINE[dataset]
    if r2 > base + 1e-9:
        return "WIN"
    if abs(r2 - base) < 1e-6:
        return "TIE"
    return "LOSS"


def print_and_save_results(results: list[dict], out_name: str, item_label: str):
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / out_name, index=False)
    print(f"\n=== {item_label}: RESULTS vs. CURRENT ROUTED DEPLOYED BASELINE ===")
    print(df.to_string(index=False))
    return df
