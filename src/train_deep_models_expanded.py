"""
Dataset Expansion Phase 1: VLSTM/CNN-LSTM/PiFormer retrained on the
expanded NASA+MIT pool (all 34 NASA batteries + all 185 MIT cells, vs.
the original 4+28=32). Additive - train_deep_models.py and every file it
writes (vlstm_soh.pt, cnn_lstm_soh.pt, piformer_soh.pt,
deep_models_metrics.csv, deep_models_{test,train}_preds.csv,
channel_norm_stats.json, battery_split.json) are completely untouched;
this script writes its own parallel `*_expanded` files, including its
OWN channel_norm_stats_expanded.json (fit on the EXPANDED fit-battery
split, not reused from the original - the whole point of re-fitting on
more data is that the fit-set distribution itself may differ).

Same architecture, same hyperparameters (40 epochs, batch=64, lr=1e-3,
patience=8) as the original run - per instruction, unchanged unless
there's a clear reason to. No such reason was found for THIS script;
the compute-time consequence of ~5.9x more cycles is accepted as the
cost of the experiment, not worked around by cutting the epoch budget.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
from sequence_features import (
    build_dataset_tensors, CHANNEL_NAMES,
    compute_channel_norm_stats, apply_channel_norm,
)
from train_deep_models import make_xy, train_one_model
from models.vlstm import VLSTM
from models.cnn_lstm import CNNLSTM
from models.piformer import PiFormer
from expanded_pool_exclusions import EXCLUDED_BATTERIES

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
(PROC_DIR / "predictions").mkdir(exist_ok=True, parents=True)
(ROOT / "models").mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

# B0052 excluded separately (too_few_cycles, see
# data/processed/phase1_expanded_failures.csv); the 10 EXCLUDED_BATTERIES
# NASA IDs (see expanded_pool_exclusions.py - physically impossible SOH
# values from a degenerate early-cycle capacity baseline) are filtered
# out below, not listed here, so this stays a single source of truth.
_ALL_NASA_RAW = [
    "B0005", "B0006", "B0007", "B0018",
    "B0025", "B0026", "B0027", "B0028", "B0029", "B0030", "B0031", "B0032",
    "B0033", "B0034", "B0036", "B0038", "B0039", "B0040", "B0041", "B0042",
    "B0043", "B0044", "B0045", "B0046", "B0047", "B0048", "B0049", "B0050",
    "B0051", "B0052", "B0053", "B0054", "B0055", "B0056",
]
ALL_NASA_CELLS = [c for c in _ALL_NASA_RAW if c not in EXCLUDED_BATTERIES and c != "B0052"]


_TENSOR_CACHE_PATH = PROC_DIR / "_expanded_battery_tensors_cache.pkl"


def load_all_battery_tensors_expanded():
    """Same contract as train_deep_models.load_all_battery_tensors, but
    over the full 34-NASA / 185-MIT pool instead of 4/28.

    Disk-cached (pickle) - added after this exact tensor-load step cost
    ~17-18 minutes on EACH of 3 separate runs this session, all because
    an external process teardown (not a code bug) killed the training
    script before it finished, forcing a full raw-data reread every
    single retry while the actual training progress was lost anyway.
    Caching this ~800MB-ish in-memory structure to disk means a future
    interrupted-and-resumed run only pays this cost ONCE, not once per
    restart - a genuine resilience fix earned by repeated real pain, not
    a speculative optimization. Cache key is implicit (no invalidation
    logic): this is a one-shot dataset-expansion pipeline, not a script
    that runs repeatedly against changing inputs, so a stale cache is
    not a realistic risk here - delete the file manually if the
    underlying battery list/exclusions ever change and this needs to be
    rebuilt from raw data again."""
    if _TENSOR_CACHE_PATH.exists():
        t0 = time.time()
        import pickle
        with open(_TENSOR_CACHE_PATH, "rb") as f:
            out = pickle.load(f)
        print(f"[deep-exp] loaded {len(out)} battery tensors from cache "
              f"({_TENSOR_CACHE_PATH.name}) in {time.time()-t0:.1f}s - skipped the "
              f"~17min raw-data reload")
        return out

    out = {}
    t0 = time.time()
    for cid in ALL_NASA_CELLS:
        cycles = list(iterate_nasa_cycles(cid))
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is not None:
            out[cid] = (X.astype(np.float32), soh.astype(np.float32), rul.astype(np.float32), "NASA")
        print(f"[deep-exp] loaded NASA/{cid}: {X.shape if X is not None else None}")

    with open(PROC_DIR / "mit_full_cells.json") as f:
        mit_full_raw = json.load(f)
    mit_full = [e for e in mit_full_raw if e["global_id"] not in EXCLUDED_BATTERIES]
    print(f"[deep-exp] MIT cells: {len(mit_full_raw)} total, {len(mit_full)} after excluding "
          f"{len(mit_full_raw) - len(mit_full)} with degenerate SOH baselines (see expanded_pool_exclusions.py)")
    for i, entry in enumerate(mit_full):
        cycles = list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is not None:
            out[entry["global_id"]] = (X.astype(np.float32), soh.astype(np.float32),
                                        rul.astype(np.float32), "MIT")
        if (i + 1) % 20 == 0:
            print(f"[deep-exp] loaded {i+1}/{len(mit_full)} MIT tensors "
                  f"({time.time()-t0:.0f}s elapsed)")
    print(f"[deep-exp] tensor loading done in {time.time()-t0:.1f}s, {len(out)} batteries")
    try:
        import pickle
        t1 = time.time()
        with open(_TENSOR_CACHE_PATH, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[deep-exp] cached battery tensors to {_TENSOR_CACHE_PATH.name} "
              f"({time.time()-t1:.1f}s) - future runs skip the raw-data reload")
    except Exception as e:
        print(f"[deep-exp] WARNING: failed to write tensor cache ({e}) - "
              f"not fatal, just means the next run pays the reload cost again")
    return out


def predict(model, X, chunk_size: int = 4096):
    """Chunked inference - a real bug caught on this expanded-pool run:
    the original (unchunked) version worked fine on the original
    32-battery scale's small arrays and on this run's ~29,489-row TEST
    set, but crashed with `RuntimeError: ... not enough memory: you
    tried to allocate 79203200000 bytes` (~79GB) when called on the
    ~123,755-row combined fit+val set for train-set predictions.
    Root cause: PiFormer's multi-head attention materializes a
    (batch, heads, seq, seq) score tensor - O(batch) memory that VLSTM/
    CNN-LSTM's recurrent/conv forward passes don't have - so it scales
    fine at 29K rows but not at 124K. Chunking produces numerically
    IDENTICAL predictions (no dependency between rows in any of these
    3 architectures - each cycle is scored independently), just bounded
    peak memory regardless of how large X is.

    Also defensively calls .eval() here (belt-and-suspenders on top of
    load_or_train's explicit call below) - a second real bug this
    session found the missing-.eval() failure mode corrupts CNN-LSTM's
    BatchNorm-dependent predictions specifically; guarding it here too
    means no future caller of predict() can reintroduce it."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), chunk_size):
            batch = torch.tensor(X[i:i + chunk_size])
            outs.append(model(batch).squeeze(-1).numpy())
    raw = np.concatenate(outs)
    return raw * model.y_std_ + model.y_mean_


def main():
    t0 = time.time()
    battery_data = load_all_battery_tensors_expanded()
    print(f"[deep-exp] loaded {len(battery_data)} batteries in {time.time()-t0:.1f}s")

    split = json.loads((PROC_DIR / "battery_split_expanded.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]

    n_val_batteries = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val_batteries:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    print(f"[deep-exp] fit batteries: {len(fit_ids)}, val batteries: {len(val_ids)}, "
          f"test batteries: {len(test_ids)}")

    X_fit, y_fit, rul_fit, _, _, _ = make_xy(battery_data, fit_ids)
    X_val, y_val, rul_val, _, _, _ = make_xy(battery_data, val_ids)
    X_test, y_test, rul_test, ds_test, bid_test, cyc_test = make_xy(battery_data, test_ids)
    print(f"[deep-exp] fit cycles={len(X_fit)}, val cycles={len(X_val)}, test cycles={len(X_test)}")

    # Resume support (added after this exact run crashed mid-way on a
    # memory bug in the train-predictions step below, AFTER all 3 models
    # had already finished their expensive training - VLSTM 10154.2s,
    # CNNLSTM 1683.1s, PiFormer 8290.7s, ~5.6h combined). Reusing the
    # already-trained, already-verified checkpoints + norm stats rather
    # than retraining from scratch: identical result (deterministic given
    # the same fit_ids/seed, already confirmed reproducible epoch-for-
    # epoch across the two attempts at this run), zero wasted compute.
    norm_stats_path = PROC_DIR / "channel_norm_stats_expanded.json"
    if norm_stats_path.exists():
        norm_stats = json.loads(norm_stats_path.read_text())
        print("[deep-exp] reusing existing channel_norm_stats_expanded.json (already computed this run)")
    else:
        norm_stats = compute_channel_norm_stats(X_fit)
        with open(norm_stats_path, "w") as f:
            json.dump(norm_stats, f, indent=2)
    for i, s in enumerate(norm_stats):
        print(f"[deep-exp] channel {i} ({CHANNEL_NAMES[i]}) norm: clip=[{s['lo']:.3g},{s['hi']:.3g}] "
              f"mean={s['mean']:.3g} std={s['std']:.3g}")
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_val = apply_channel_norm(X_val, norm_stats)
    X_test = apply_channel_norm(X_test, norm_stats)

    results = {}
    all_metrics = []

    def load_or_train(name, model, ckpt_name, X_fit_m, X_val_m, hist_name):
        ckpt_path = ROOT / "models" / ckpt_name
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path))
            # Real bug caught on THIS resume (root-caused, not papered
            # over): a freshly-constructed nn.Module defaults to
            # .train() mode, and load_state_dict() does NOT change that.
            # CNN-LSTM's nn.BatchNorm1d layers use live BATCH statistics
            # in train() mode instead of the stored running_mean/
            # running_var - meaning its checkpoint's WEIGHTS were loaded
            # correctly but its predictions came out wrong anyway
            # (TEST R2 0.9666->0.8837, a genuine silent corruption, not
            # a rounding difference). VLSTM (no BatchNorm) and PiFormer
            # (LayerNorm, mode-independent) were both unaffected by the
            # same missing call, which is exactly why only CNN-LSTM's
            # numbers looked wrong - the identical architecture-specific
            # signature as this project's very first CNN-LSTM
            # investigation, just a different root cause this time.
            model.eval()
            # y_mean_/y_std_ aren't part of state_dict() (same gotcha
            # session 5 already documented) - recomputed identically
            # from y_fit, the exact tensor train_one_model would have
            # used, so de-standardization matches the original run.
            model.y_mean_, model.y_std_ = float(y_fit.mean()), float(y_fit.std() + 1e-8)
            print(f"[deep-exp] {name}: loaded existing checkpoint {ckpt_name}, skipping retraining")
            return model
        t1 = time.time()
        model, hist = train_one_model(name, model, X_fit_m, y_fit, X_val_m, y_val)
        print(f"[deep-exp] {name} trained in {time.time()-t1:.1f}s")
        torch.save(model.state_dict(), ckpt_path)
        pd.DataFrame(hist).to_csv(PROC_DIR / "predictions" / hist_name, index=False)
        return model

    vlstm = load_or_train("VLSTM", VLSTM(input_size=1, hidden_size=32, n_targets=1),
                           "vlstm_soh_expanded.pt", X_fit[:, :, 0:1], X_val[:, :, 0:1],
                           "vlstm_expanded_history.csv")
    results["VLSTM"] = predict(vlstm, X_test[:, :, 0:1])

    cnn_lstm = load_or_train("CNNLSTM", CNNLSTM(), "cnn_lstm_soh_expanded.pt",
                              X_fit, X_val, "cnn_lstm_expanded_history.csv")
    results["CNNLSTM"] = predict(cnn_lstm, X_test)

    piformer = load_or_train("PiFormer", PiFormer(), "piformer_soh_expanded.pt",
                              X_fit, X_val, "piformer_expanded_history.csv")
    results["PiFormer"] = predict(piformer, X_test)

    for name, pred in results.items():
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        print(f"[deep-exp] {name} TEST RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
        all_metrics.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})

    out = pd.DataFrame({
        "dataset": ds_test, "battery_id": bid_test, "cycle_idx": cyc_test,
        "SOH": y_test, "RUL": rul_test,
        "y_pred_VLSTM": results["VLSTM"], "y_pred_CNNLSTM": results["CNNLSTM"],
        "y_pred_PiFormer": results["PiFormer"],
    })
    out.to_csv(PROC_DIR / "predictions" / "deep_models_expanded_test_preds.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(PROC_DIR / "predictions" / "deep_models_expanded_metrics.csv", index=False)

    ids_fitval = fit_ids + val_ids
    _, ytr, rultr, dstr, bidtr, cyctr = make_xy(battery_data, ids_fitval)
    Xtr = np.concatenate([X_fit, X_val])
    assert len(Xtr) == len(ytr)
    tr_pred_v = predict(vlstm, Xtr[:, :, 0:1])
    tr_pred_c = predict(cnn_lstm, Xtr)
    tr_pred_p = predict(piformer, Xtr)
    out_train = pd.DataFrame({
        "dataset": dstr, "battery_id": bidtr, "cycle_idx": cyctr,
        "SOH": ytr, "RUL": rultr,
        "y_pred_VLSTM": tr_pred_v, "y_pred_CNNLSTM": tr_pred_c, "y_pred_PiFormer": tr_pred_p,
    })
    out_train.to_csv(PROC_DIR / "predictions" / "deep_models_expanded_train_preds.csv", index=False)

    print(f"[deep-exp] ALL DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
