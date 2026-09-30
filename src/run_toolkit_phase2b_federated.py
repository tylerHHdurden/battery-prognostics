"""
Toolkit Phase 2B: federated multi-source learning.

Question this phase answers: does a federated formulation (no raw rows
ever leaving their source) recover the Phase 2 candidate's pooled-
centralized performance - and does it fix the NASA/MIT "crowded out"
problem Phase 2 found, without the catastrophic collapse the naive
equal-total-weight balanced retrain (Phase 2d) produced?

NATIVE XGBOOST FEDERATED LEARNING - CONFIRMED IMPRACTICAL, NOT ASSUMED:
directly probed (`collective.init(dmlc_communicator="federated", ...)`
with a real 2-worker tracker, not just a trivial world_size=1 call, which
DOES succeed vacuously without ever touching the plugin) - this
project's actual installed xgboost (3.3.0, the standard PyPI wheel) threw
"XGBoost is not compiled with federated learning support" from its own
C++ layer. That plugin requires building xgboost from source with gRPC
and `-DPLUGIN_FEDERATED=ON` - a real environment change out of proportion
to this phase's own scope, not attempted. **Falls through to Flower's
XGBoost strategy, per this item's own explicit instruction.**

FLOWER USAGE, disclosed precisely: uses Flower's OWN, real
`flwr.server.strategy.fedxgb_bagging.aggregate` function (imported
directly from the installed `flwr==1.39.0` package, not reimplemented) -
the exact tree-JSON-merging primitive that class's `aggregate_fit` calls
internally. NOT using Flower's Ray-based simulation runtime
(`flwr.simulation.start_simulation`/`run_simulation`, both of which
require the separate `ray` package and spin up actor processes) - that
full-stack simulation changes nothing about the federation ALGORITHM
itself (which is exactly this same tree-merging function either way) and
adds a real-risk, heavy dependency (Ray's Windows support is uneven) for
zero algorithmic difference in what's being measured here. The
round/client loop below is driven directly in-process instead; the "no
raw rows leave a client" property is enforced by architecture (each
client-round function receives only ITS OWN client's DMatrix and returns
ONLY serialized tree bytes) AND by a real runtime assertion at the exact
point client output crosses into server-side aggregation
(`_cross_client_boundary`), not just by that architectural claim alone.

CLIENT-WEIGHTING TRANSLATION, a genuine judgment call, disclosed not
hidden: bagging-federated XGBoost aggregates by literally concatenating
each client's newly-trained trees into one ensemble - there is no
gradient-averaging step where a per-client "weight" scalar has an
obvious meaning (unlike FedAvg for neural nets). The natural lever for a
client's influence on the final ensemble is how many of the ensemble's
trees that client contributed. So "sample-weighted" / "uniform" /
"tempered (n^0.5)" are implemented here as the NUMBER OF TREES each
client contributes per global round, normalized so the same TOTAL tree
budget applies across all three schemes (a fair comparison of
redistribution, not of total model capacity):
  - uniform:        trees_c = local_round for every present client
  - sample-weighted: trees_c proportional to that client's own row count
  - tempered (n^0.5): trees_c proportional to sqrt(that client's row count)

EVALUATION PROTOCOL: this project's own established LODO family-holdout
methodology (`src/run_finalpass3_check1_lodo_family_holdout.py`'s own
FAMILIES sibling-grouping convention and its `metrics`/`bootstrap_ci`
functions, imported directly - not re-derived), extended with Tongji
(Phase 1(b), confirmed no identified sibling in this pool) and using the
Phase 2 candidate's own retrained fusion embeddings
(`fusion_embeddings_multisource.csv`) and identical hyperparameters
(same feature set, n_estimators=500, max_depth=6, learning_rate=0.03,
subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, the same
monotone_constraints pattern) - so "federated vs centralized pooled vs
NASA+MIT-only" is a fair, apples-to-apples comparison, not confounded by
a different feature set or capacity.

Per target: federated (x3 weighting schemes) vs centralized pooled vs
NASA+MIT-only, R2/RMSE/MAE with battery-level bootstrap CIs, and the gap
to centralized stated explicitly. Additionally: the tempered (n^0.5)
weighting is run ONCE in the CENTRALIZED model too (via XGBoost's own
per-row sample_weight, group total weight ∝ n_group^0.5), to test
directly whether it fixes NASA/MIT without Phase 2d's catastrophic
collapse (equal-TOTAL-weight, i.e. group total weight ∝ n_group^0).
"""
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from flwr.server.strategy.fedxgb_bagging import aggregate as fedxgb_bagging_aggregate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import canonical_feature_cols, add_reformulated_duration_features, build_calce_merged, PROC_DIR, OUT_DIR
from run_finalpass3_check1_lodo_family_holdout import metrics, bootstrap_ci, N_BOOT, SEED as LODO_SEED

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

BATTERYLIFE_SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                        "stanford", "stanford_2", "isu_ilcc", "tongji"]
ALL_SOURCES = ["NASA", "MIT", "CALCE", "Oxford", "HUST", "XJTU"] + BATTERYLIFE_SOURCES

# Extends run_finalpass3_check1_lodo_family_holdout.py's own FAMILIES dict
# with tongji - confirmed no identified sibling in this pool (a single
# BatteryLife sub-source integrated in Phase 1(b), not sharing lab/
# protocol origin with any other listed source, same standard applied to
# every other singleton source in the original dict).
FAMILIES = {
    "NASA": {"NASA"}, "MIT": {"MIT"}, "CALCE": {"CALCE"}, "Oxford": {"Oxford"},
    "HUST": {"HUST"}, "XJTU": {"XJTU"}, "ul_pur": {"ul_pur"}, "hnei": {"hnei"},
    "snl": {"snl"}, "rwth": {"rwth"}, "isu_ilcc": {"isu_ilcc"}, "tongji": {"tongji"},
    "mich": {"mich", "mich_exp"}, "mich_exp": {"mich", "mich_exp"},
    "stanford": {"stanford", "stanford_2"}, "stanford_2": {"stanford", "stanford_2"},
}
# 14 distinct client groups (siblings merged into one client each).
CLIENT_GROUPS = sorted({tuple(sorted(v)) for v in FAMILIES.values()})

SEED = 42
N_ESTIMATORS_TARGET = 500  # matches the Phase 2 candidate exactly
LOCAL_ROUND_BASE = 1       # uniform-scheme trees-per-client-per-round
EMBED_DIM = 16


def _multi_tree_bagging_aggregate(bst_prev_org: bytes | None, bst_curr_org: bytes) -> bytes:
    """Generalizes flwr.server.strategy.fedxgb_bagging.aggregate() (still
    imported and used directly elsewhere in this module, not replaced)
    to a client payload spanning MULTIPLE local boosting rounds at once,
    not just one.

    Flower's own `aggregate()` reads `num_parallel_tree` (normally 1)
    from the incoming payload to decide how many trees to pull in - it
    assumes exactly one new tree per call, and silently drops the rest
    of a multi-round slice otherwise (see train_federated's own comment
    for how this was caught: a synthetic 2-client probe with very
    different trees_per_client produced bit-identical predictions and
    identical num_boosted_rounds() before this was found). Calling
    flwr's aggregate() once per individual tree instead (the "obviously
    correct" fix) is itself expensive here: each call round-trips the
    ENTIRE growing global model through json.loads/json.dumps, so an
    O(total_trees) per-call cost paid `total_trees` times is O(n^2)
    overall - and the whole reason this phase's federated runs are slow
    to begin with (see this run's own first, buggy attempt: ~135 minutes
    wall time). This function appends ALL trees present in the incoming
    payload in ONE json.loads/json.dumps round-trip instead of one per
    tree - verified bit-exact equivalent to calling flwr's own
    aggregate() once per tree in sequence (same final predictions,
    same num_boosted_rounds(); only raw JSON byte layout differs,
    which is immaterial - re-verify this equivalence directly if this
    function is ever touched, not just trusted from this comment)."""
    if not bst_prev_org:
        return bst_curr_org
    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))
    tree_num_prev = int(bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"])
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    n_new = len(trees_curr)
    bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(tree_num_prev + n_new)
    iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"]
    base = iteration_indptr[-1]  # fixed BEFORE the loop - the list below is mutated in place as we go,
    # so reading its own [-1] again mid-loop (as an earlier, buggy version of this function did) would
    # read back a value this same loop had just appended, double-counting the increment every step.
    for i in range(n_new):
        trees_curr[i]["id"] = tree_num_prev + i
        bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(trees_curr[i])
        bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)
        iteration_indptr.append(base + i + 1)
    return bytes(json.dumps(bst_prev), "utf-8")


def _verify_batched_aggregate_matches_flwr():
    """Runs once at script start (called from main(), not just trusted
    from a comment): checks `_multi_tree_bagging_aggregate` produces
    bit-exact-identical predictions to calling flwr's own real
    `aggregate()` once per tree in sequence - the actual equivalence
    claim this module's design rests on, exercised as a real assertion
    every run, matching this project's own established discipline (see
    e.g. test_online_conformal.py) of verifying an equivalence claim in
    code rather than asserting it only in a docstring.

    Simulates the REAL usage pattern this module actually needs, not
    just a single append onto empty state: 2 clients with DIFFERENT
    trees-per-round (1 and 3), across 3 SEPARATE rounds, so the global
    model is appended onto repeatedly (non-empty bst_prev_org every call
    after the first). An earlier version of this check only tested one
    append onto an empty global model and PASSED while a real, separate
    off-by-one bug (iteration_indptr accounting) was live in this same
    function - caught only once real multi-round accumulation was
    exercised (see DEVELOPMENT_LOG.md), which is exactly why this check
    now specifically targets that path."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.05, size=200)
    d = xgb.DMatrix(X, label=y)
    params = {"max_depth": 3, "objective": "reg:squarederror"}
    client_trees_per_round = {"A": 1, "B": 3}

    def run(use_batched: bool):
        global_bytes = None
        for _round in range(3):
            for client, n_trees in client_trees_per_round.items():
                bst = xgb.Booster(params=params, cache=[d])
                if global_bytes is not None:
                    bst.load_model(bytearray(global_bytes))
                n_before = bst.num_boosted_rounds()
                for _ in range(n_trees):
                    bst.update(d, bst.num_boosted_rounds())
                new_trees = bst[n_before:bst.num_boosted_rounds()]
                if use_batched:
                    payload = new_trees.save_raw("json")
                    global_bytes = _multi_tree_bagging_aggregate(global_bytes, payload) if global_bytes else payload
                else:
                    for i in range(n_trees):
                        one_tree = new_trees[i:i + 1]
                        payload = one_tree.save_raw("json")
                        global_bytes = fedxgb_bagging_aggregate(global_bytes, payload) if global_bytes else payload
        final = xgb.Booster(params=params)
        final.load_model(bytearray(global_bytes))
        return final

    ground_truth = run(use_batched=False)
    batched = run(use_batched=True)

    expected_total = 3 * sum(client_trees_per_round.values())
    assert ground_truth.num_boosted_rounds() == batched.num_boosted_rounds() == expected_total, \
        (f"tree count mismatch: ground_truth={ground_truth.num_boosted_rounds()}, "
         f"batched={batched.num_boosted_rounds()}, expected={expected_total}")
    pred_gt = ground_truth.predict(xgb.DMatrix(X))
    pred_batched = batched.predict(xgb.DMatrix(X))
    assert np.array_equal(pred_gt, pred_batched), \
        "_multi_tree_bagging_aggregate does NOT match flwr's own per-tree aggregate() - do not trust Phase 2B results"
    print("[phase2b] self-check PASSED: batched multi-tree aggregation is bit-exact equivalent to "
          "flwr's own per-tree aggregate(), across repeated multi-round, multi-client accumulation")


def _cross_client_boundary(payload) -> bytes:
    """The ONLY conduit through which a client's local training output is
    allowed to reach server-side aggregation. Asserts the payload is
    exactly serialized model bytes - never a raw array/DataFrame - so a
    client function accidentally returning its own local data (instead of
    trained tree bytes) fails loudly here, not silently downstream."""
    assert isinstance(payload, (bytes, bytearray)), \
        f"client boundary violation: expected serialized model bytes, got {type(payload)}"
    payload = bytes(payload)
    assert not hasattr(payload, "shape"), "client boundary violation: payload looks array-like"
    return payload


def _attach_candidate_fusion(df: pd.DataFrame, name: str, fusion_df: pd.DataFrame,
                             feature_cols_base: list[str], all_cols: list[str]) -> pd.DataFrame:
    """Replace whatever `fusion_*` columns a per-source parquet carries with the CANDIDATE
    encoder's own embeddings (fusion_embeddings_multisource.csv), joined on (battery_id, cycle_idx).

    BUG FIXED HERE (found 2026-09-30 while investigating "candidate encoder divergence"): this loader
    used to take `fusion_*` straight from the per-source merged parquets for Oxford/HUST/XJTU and all
    BatteryLife sources. Those columns are the OLD deployed encoder's embeddings - and for BatteryLife
    they were computed from RAW, un-normalized tensors (build_batterylife_hi_table.py encodes `X`
    without apply_channel_norm; raw dVdQ reaches ~4e9, so isu_ilcc/rwth embeddings reach 1e7-1e13,
    95-99.97% of their rows > 100). So Phase 2B's "same features as the Phase 2 candidate" was false:
    it silently mixed the candidate encoder (NASA/MIT/CALCE) with the old one (everything else)."""
    import encoder_provenance as ep
    ep.assert_store(PROC_DIR / "fusion_embeddings_multisource.csv", ep.CAND, f"_attach_candidate_fusion[{name}]")
    fcols = [f"fusion_{i}" for i in range(EMBED_DIM)]
    hi = df.drop(columns=[c for c in fcols if c in df.columns]).copy()
    hi["battery_id"] = hi["battery_id"].astype(str)
    emb = fusion_df[fusion_df["dataset"] == name][["battery_id", "cycle_idx"] + fcols].copy()
    emb["battery_id"] = emb["battery_id"].astype(str)
    merged = pd.merge(hi, emb, on=["battery_id", "cycle_idx"], how="inner")
    assert len(merged) > 0, f"{name}: no rows matched the candidate embeddings"
    return merged[all_cols + ["SOH", "battery_id"]].copy()


def load_pooled_data():
    """Same pooling as run_toolkit_phase2_multisource_retrain.py's own
    Phase 2 candidate build: 8 reformulated canonical HI features +
    cycle_idx + the CANDIDATE's own 16-dim fusion embeddings
    (fusion_embeddings_multisource.csv, the retrained encoder's output,
    NOT the old deployed encoder's fusion_embeddings.csv)."""
    feature_cols_base = canonical_feature_cols(reformulated=True)
    all_cols = feature_cols_base + ["cycle_idx"] + [f"fusion_{i}" for i in range(EMBED_DIM)]

    hi_full = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_full = add_reformulated_duration_features(hi_full)
    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings_multisource.csv")

    sources = {}
    nasa_mit = pd.merge(hi_full[hi_full["dataset"].isin(["NASA", "MIT"])],
                         fusion_df[fusion_df["dataset"].isin(["NASA", "MIT"])],
                         on=["dataset", "battery_id", "cycle_idx"], how="inner")
    sources["NASA"] = nasa_mit[nasa_mit["dataset"] == "NASA"][all_cols + ["SOH", "battery_id"]].copy()
    sources["MIT"] = nasa_mit[nasa_mit["dataset"] == "MIT"][all_cols + ["SOH", "battery_id"]].copy()

    calce_merged = build_calce_merged(hi_full)
    calce_merged["battery_id"] = calce_merged["battery_id"].astype(str)
    calce_fusion = fusion_df[fusion_df["dataset"] == "CALCE"][["battery_id", "cycle_idx"] + [f"fusion_{i}" for i in range(EMBED_DIM)]]
    calce_full = pd.merge(calce_merged[["battery_id", "cycle_idx"] + feature_cols_base + ["SOH"]],
                           calce_fusion, on=["battery_id", "cycle_idx"], how="inner")
    sources["CALCE"] = calce_full[all_cols + ["SOH", "battery_id"]].copy()

    for name, fname in [("Oxford", "stage5_1_oxford_merged.parquet"), ("HUST", "stage5_1_hust_merged.parquet"),
                         ("XJTU", "stage5_1_xjtu_merged.parquet")]:
        df = pd.read_parquet(PROC_DIR / fname)
        sources[name] = _attach_candidate_fusion(df, name, fusion_df, feature_cols_base, all_cols)

    for source in BATTERYLIFE_SOURCES:
        if source == "tongji":
            path = PROC_DIR / "batterylife_tongji_merged.parquet"
        else:
            path = PROC_DIR / f"batterylife_{source}_merged.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df.copy()
        df["battery_id"] = source + "::" + df["battery_id"].astype(str)
        sources[source] = _attach_candidate_fusion(df, source, fusion_df, feature_cols_base, all_cols)

    # Hard guard against the bug this function used to have (see _attach_candidate_fusion's docstring):
    # the candidate encoder's clipped-input outputs are provably bounded (worst-case corner < ~11 for
    # the saved weights, observed max 4.2 over all 419,250 stored rows) - anything larger means an
    # old-encoder/raw-X embedding got in.
    fcols = [f"fusion_{i}" for i in range(EMBED_DIM)]
    for name, df in sources.items():
        mx = float(np.nanmax(np.abs(df[fcols].to_numpy(dtype=float))))
        assert mx < 20.0, f"{name}: max |fusion| = {mx:.3g} - not the candidate encoder's bounded output"

    for name, df in sources.items():
        if not df["battery_id"].astype(str).str.startswith(name + "::").all():
            df["battery_id"] = name + "::" + df["battery_id"].astype(str)

    return sources, all_cols, feature_cols_base


def client_frame(sources, group, all_cols):
    return pd.concat([sources[s] for s in group if s in sources], ignore_index=True)


def build_dmatrix(df, all_cols, col_medians, sample_weight=None):
    X = df[all_cols].to_numpy(dtype=float, copy=True)
    X = np.where(np.isinf(X), np.nan, X)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    y = df["SOH"].to_numpy(dtype=float)
    return xgb.DMatrix(X, label=y, weight=sample_weight), X, y


def train_centralized(train_df, test_df, all_cols, monotone, col_medians, sample_weight=None):
    X_train, _, y_train = None, None, None
    Xtr = train_df[all_cols].to_numpy(dtype=float, copy=True)
    Xtr = np.where(np.isinf(Xtr), np.nan, Xtr)
    inds = np.where(np.isnan(Xtr))
    Xtr[inds] = np.take(col_medians, inds[1])
    ytr = train_df["SOH"].to_numpy(dtype=float)

    model = XGBRegressor(n_estimators=N_ESTIMATORS_TARGET, max_depth=6, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
                          reg_lambda=1.0, monotone_constraints=monotone)
    model.fit(Xtr, ytr, sample_weight=sample_weight)

    Xte = test_df[all_cols].to_numpy(dtype=float, copy=True)
    Xte = np.where(np.isinf(Xte), np.nan, Xte)
    inds2 = np.where(np.isnan(Xte))
    Xte[inds2] = np.take(col_medians, inds2[1])
    pred = model.predict(Xte)
    return pred


def _trees_per_client(client_rows: dict, scheme: str, n_clients: int) -> dict:
    """trees_c per round for each client, normalized so the total trees
    contributed per round == n_clients (i.e. the SAME total ensemble size
    across all three weighting schemes for a fair comparison - only the
    per-client REDISTRIBUTION differs)."""
    names = list(client_rows.keys())
    if scheme == "uniform":
        raw = {c: 1.0 for c in names}
    elif scheme == "sample_weighted":
        total = sum(client_rows.values())
        raw = {c: client_rows[c] / total * n_clients for c in names}
    elif scheme == "tempered":
        sqrt_sum = sum(np.sqrt(v) for v in client_rows.values())
        raw = {c: np.sqrt(client_rows[c]) / sqrt_sum * n_clients for c in names}
    else:
        raise ValueError(scheme)
    trees = {c: max(1, int(round(raw[c]))) for c in names}
    return trees


def train_federated(client_dfs: dict, all_cols, monotone, col_medians, scheme: str, n_rounds: int):
    """Bagging federated XGBoost, driven in-process. Each client's local
    training only ever sees ITS OWN DMatrix (built once, reused every
    round - never centralized/concatenated with any other client's
    data). Only serialized tree bytes cross into aggregation, via
    `_cross_client_boundary`."""
    client_rows = {c: len(df) for c, df in client_dfs.items()}
    n_clients = len(client_dfs)
    trees_per_client = _trees_per_client(client_rows, scheme, n_clients)

    client_dmatrix = {}
    for c, df in client_dfs.items():
        dm, _, _ = build_dmatrix(df, all_cols, col_medians)
        client_dmatrix[c] = dm

    params = {
        "objective": "reg:squarederror", "max_depth": 6, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
        "monotone_constraints": "(" + ",".join(str(m) for m in monotone) + ")",
        "tree_method": "hist",
    }

    # Each client-round contributes trees_per_client[c] new trees in ONE
    # batched append (`_multi_tree_bagging_aggregate`, verified bit-exact
    # equivalent to calling flwr's own per-tree `aggregate()` in
    # sequence - see that function's own docstring for how the naive
    # per-tree version's O(n^2) JSON-round-trip cost was found and why
    # this batching is needed, not just a nice-to-have).
    global_model_bytes = None
    for _ in range(n_rounds):
        for c in client_dfs:  # fixed, deterministic client order each round
            dtrain = client_dmatrix[c]
            bst = xgb.Booster(params=params, cache=[dtrain])
            if global_model_bytes is not None:
                bst.load_model(bytearray(global_model_bytes))
            n_before = bst.num_boosted_rounds()
            for _ in range(trees_per_client[c]):
                bst.update(dtrain, bst.num_boosted_rounds())
            new_trees = bst[n_before:bst.num_boosted_rounds()]
            client_payload = _cross_client_boundary(new_trees.save_raw("json"))
            global_model_bytes = _multi_tree_bagging_aggregate(
                global_model_bytes if global_model_bytes is None else _cross_client_boundary(global_model_bytes),
                client_payload,
            )

    final_bst = xgb.Booster(params=params)
    final_bst.load_model(bytearray(global_model_bytes))
    total_trees = sum(trees_per_client.values()) * n_rounds
    return final_bst, trees_per_client, total_trees


def evaluate_booster(bst, test_df, all_cols, col_medians):
    Xte = test_df[all_cols].to_numpy(dtype=float, copy=True)
    Xte = np.where(np.isinf(Xte), np.nan, Xte)
    inds2 = np.where(np.isnan(Xte))
    Xte[inds2] = np.take(col_medians, inds2[1])
    dtest = xgb.DMatrix(Xte)
    return bst.predict(dtest)


def main():
    t0 = time.time()
    print("=== Toolkit Phase 2B: federated multi-source learning ===", flush=True)
    _verify_batched_aggregate_matches_flwr()
    print(f"[phase2b] {len(CLIENT_GROUPS)} client groups: {CLIENT_GROUPS}", flush=True)

    sources, all_cols, feature_cols_base = load_pooled_data()
    n_fusion = EMBED_DIM
    monotone = tuple([0] * len(feature_cols_base) + [-1] + [0] * n_fusion)

    n_rounds = max(1, round(N_ESTIMATORS_TARGET / len(CLIENT_GROUPS)))
    print(f"[phase2b] federated: {n_rounds} rounds x {len(CLIENT_GROUPS)} clients x "
          f"{LOCAL_ROUND_BASE} local tree(s)/round (uniform baseline) "
          f"= {n_rounds * len(CLIENT_GROUPS)} total trees (target {N_ESTIMATORS_TARGET})")

    # ---------- NASA+MIT-only baseline: trained ONCE, evaluated zero-retrain against every held-out source ----------
    nasa_mit_train = pd.concat([sources["NASA"], sources["MIT"]], ignore_index=True)
    X_nm = nasa_mit_train[all_cols].to_numpy(dtype=float, copy=True)
    X_nm = np.where(np.isinf(X_nm), np.nan, X_nm)
    nasa_mit_medians = np.nanmedian(X_nm, axis=0)
    inds = np.where(np.isnan(X_nm))
    X_nm[inds] = np.take(nasa_mit_medians, inds[1])
    y_nm = nasa_mit_train["SOH"].to_numpy(dtype=float)
    nasa_mit_model = XGBRegressor(n_estimators=N_ESTIMATORS_TARGET, max_depth=6, learning_rate=0.03,
                                   subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
                                   reg_lambda=1.0, monotone_constraints=monotone)
    nasa_mit_model.fit(X_nm, y_nm)
    print("[phase2b] NASA+MIT-only baseline trained (once, reused for every held-out target)")

    results = []
    weighting_schemes = ["sample_weighted", "uniform", "tempered"]

    for held_out in ALL_SOURCES:
        if held_out not in sources:
            continue
        family = FAMILIES[held_out]
        t_s = time.time()
        test_df = sources[held_out]

        # ---- pool of everything OUTSIDE the held-out family ----
        train_client_dfs = {}
        for group in CLIENT_GROUPS:
            if set(group) & family:
                continue  # this client group IS the held-out family (or shares a sibling with it)
            df = client_frame(sources, group, all_cols)
            if len(df):
                train_client_dfs["+".join(group)] = df

        train_bids = set(pd.concat(train_client_dfs.values())["battery_id"].unique())
        test_bids = set(test_df["battery_id"].unique())
        assert len(train_bids & test_bids) == 0, f"BUG: battery ID collision for held-out={held_out}"

        pooled_train = pd.concat(train_client_dfs.values(), ignore_index=True)
        X_pool = pooled_train[all_cols].to_numpy(dtype=float, copy=True)
        X_pool = np.where(np.isinf(X_pool), np.nan, X_pool)
        col_medians = np.nanmedian(X_pool, axis=0)

        y_test = test_df["SOH"].to_numpy(dtype=float)
        bids_test = test_df["battery_id"].to_numpy()

        # ---- centralized pooled (unweighted - matches the Phase 2 candidate exactly) ----
        pred_central = train_centralized(pooled_train, test_df, all_cols, monotone, col_medians)
        m_central = metrics(y_test, pred_central)
        ci_central = bootstrap_ci(y_test, pred_central, bids_test)

        # ---- centralized, tempered (n^0.5) sample weighting ----
        group_sizes = {name: len(df) for name, df in train_client_dfs.items()}
        sqrt_sum = sum(np.sqrt(v) for v in group_sizes.values())
        group_total_weight = {name: np.sqrt(group_sizes[name]) / sqrt_sum for name in group_sizes}
        row_to_group = pooled_train["battery_id"].astype(str).str.split("::").str[0]
        # map each row's own client-group name (group id used above is "src1+src2" joined form,
        # but rows only carry their OWN source prefix - resolve row -> its group's key)
        source_to_group_key = {}
        for name, group in zip(train_client_dfs.keys(), [g for g in CLIENT_GROUPS if "+".join(g) in train_client_dfs]):
            for s in group:
                source_to_group_key[s] = name
        row_group_key = row_to_group.map(source_to_group_key)
        sample_weight_tempered = row_group_key.map(
            lambda g: group_total_weight[g] / group_sizes[g]
        ).to_numpy(dtype=float)
        pred_central_tempered = train_centralized(pooled_train, test_df, all_cols, monotone, col_medians,
                                                    sample_weight=sample_weight_tempered)
        m_central_tempered = metrics(y_test, pred_central_tempered)

        # ---- NASA+MIT-only, zero-retrain ----
        pred_nm = evaluate_booster(nasa_mit_model.get_booster(), test_df, all_cols, nasa_mit_medians)
        m_nm = metrics(y_test, pred_nm)

        row = {
            "held_out_source": held_out, "n_test": len(y_test), "n_test_batteries": test_df["battery_id"].nunique(),
            "n_train": len(pooled_train), "n_clients": len(train_client_dfs),
            "centralized_r2": m_central["r2"], "centralized_r2_ci_lo": ci_central["r2"][0], "centralized_r2_ci_hi": ci_central["r2"][1],
            "centralized_rmse": m_central["rmse"], "centralized_mae": m_central["mae"],
            "centralized_tempered_r2": m_central_tempered["r2"], "centralized_tempered_rmse": m_central_tempered["rmse"],
            "nasa_mit_only_r2": m_nm["r2"], "nasa_mit_only_rmse": m_nm["rmse"], "nasa_mit_only_mae": m_nm["mae"],
        }

        for scheme in weighting_schemes:
            bst, trees_per_client, total_trees = train_federated(train_client_dfs, all_cols, monotone, col_medians,
                                                                   scheme, n_rounds)
            pred_fed = evaluate_booster(bst, test_df, all_cols, col_medians)
            m_fed = metrics(y_test, pred_fed)
            ci_fed = bootstrap_ci(y_test, pred_fed, bids_test)
            row[f"federated_{scheme}_r2"] = m_fed["r2"]
            row[f"federated_{scheme}_r2_ci_lo"] = ci_fed["r2"][0]
            row[f"federated_{scheme}_r2_ci_hi"] = ci_fed["r2"][1]
            row[f"federated_{scheme}_rmse"] = m_fed["rmse"]
            row[f"federated_{scheme}_mae"] = m_fed["mae"]
            row[f"federated_{scheme}_total_trees"] = total_trees
            row[f"federated_{scheme}_gap_to_centralized_r2"] = m_central["r2"] - m_fed["r2"]

        results.append(row)
        print(f"[phase2b] held_out={held_out}: centralized R2={m_central['r2']:.3f} | "
              f"fed(sample_weighted)={row['federated_sample_weighted_r2']:.3f} | "
              f"fed(uniform)={row['federated_uniform_r2']:.3f} | "
              f"fed(tempered)={row['federated_tempered_r2']:.3f} | "
              f"NASA+MIT-only={m_nm['r2']:.3f} | centralized_tempered={m_central_tempered['r2']:.3f} "
              f"({time.time()-t_s:.1f}s)", flush=True)

    results_df = pd.DataFrame(results)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(OUT_DIR / "toolkit_phase2b_federated_results.csv", index=False)

    nasa_row = results_df[results_df["held_out_source"] == "NASA"].iloc[0]
    mit_row = results_df[results_df["held_out_source"] == "MIT"].iloc[0]
    print("\n[phase2b] NASA/MIT tempered-centralized check (does n^0.5 group weighting fix "
          "NASA/MIT without Phase 2d's collapse?):")
    print(f"    NASA: unweighted centralized R2={nasa_row['centralized_r2']:.3f}, "
          f"tempered centralized R2={nasa_row['centralized_tempered_r2']:.3f}, "
          f"NASA+MIT-only(routed) R2={nasa_row['nasa_mit_only_r2']:.3f}")
    print(f"    MIT:  unweighted centralized R2={mit_row['centralized_r2']:.3f}, "
          f"tempered centralized R2={mit_row['centralized_tempered_r2']:.3f}, "
          f"NASA+MIT-only(routed) R2={mit_row['nasa_mit_only_r2']:.3f}")
    other = results_df[~results_df["held_out_source"].isin(["NASA", "MIT"])]
    n_other_tempered_collapsed = int((other["centralized_tempered_r2"] < 0).sum())
    print(f"    Other {len(other)} sources: {n_other_tempered_collapsed} went negative under "
          f"tempered centralized weighting (Phase 2d's equal-weight collapse hit 15/16)")

    print(f"\n[phase2b] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
