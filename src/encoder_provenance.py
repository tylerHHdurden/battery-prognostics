"""
Encoder provenance guard (2026-09-30).

Two encoders exist in this project and their embeddings are NOT interchangeable:
  OLD  = "old_v1"        models/ica_encoder.pt + data/processed/channel_norm_stats.json
                         (NASA+MIT-trained; what the old routed models, the deployed OC-SVM, fusion_embeddings.csv,
                         the Stage-5.1 Oxford/HUST/XJTU parquets, CALCE's build_calce_merged, and - after the 2026-09-30
                         fix - the BatteryLife parquets carry)
  CAND = "candidate_v1"  models/_candidate_ica_encoder.pt + data/processed/candidate_channel_norm_stats.json
                         (16-source; what the multisource candidate model/OC-SVM, fusion_embeddings_multisource.csv, the
                         trust profiles and the live upload path use)

Every model file and every embedding store gets a sidecar `<file>.meta.json` recording the encoder it expects /
carries (tag + md5 of that encoder's weights and norm stats, so a retrained encoder without a re-tag also fails).
Loaders call `assert_encoder_match(...)` and it raises loudly - never a warning - when a model is paired with
embeddings from the other encoder. This exists because that pairing happened silently twice: the BatteryLife
parquets held raw-X (un-normalized) embeddings, and the Phase 2B/trust-report pool mixed the two encoders.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD = "old_v1"
CAND = "candidate_v1"

ENCODER_FILES = {
    OLD: (ROOT / "models" / "ica_encoder.pt", ROOT / "data" / "processed" / "channel_norm_stats.json"),
    CAND: (ROOT / "models" / "_candidate_ica_encoder.pt", ROOT / "data" / "processed" / "candidate_channel_norm_stats.json"),
}


class EncoderMismatch(AssertionError):
    pass


_TEXT_SUFFIXES = {".json", ".csv", ".txt", ".md"}


def _md5(path: Path) -> str:
    """md5 of a file. Text files (json/csv/txt/md) are hashed with CRLF normalised to LF: on Windows with core.autocrlf the same
    committed file is CRLF on disk, on Linux (Streamlit Cloud) it is LF, and a raw hash differs between the two (this caused an
    EncoderMismatch on the live app on 2026-10-01). Binary files (.pt, .pkl, .parquet) are hashed as-is."""
    path = Path(path)
    h = hashlib.md5()
    if path.suffix.lower() in _TEXT_SUFFIXES:
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
        return h.hexdigest()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encoder_md5(tag: str) -> str:
    """md5 over the encoder's weights file AND its normalization stats (both define the embedding)."""
    w, s = ENCODER_FILES[tag]
    return hashlib.md5((_md5(w) + _md5(s)).encode()).hexdigest()


def sidecar(path) -> Path:
    p = Path(path)
    return p.with_name(p.name + ".meta.json")


def write_meta(path, role: str, encoder_tag: str, note: str = "") -> dict:
    """role: 'model' (expects this encoder's embeddings), 'embeddings' (carries them), or 'encoder'."""
    assert encoder_tag in ENCODER_FILES, encoder_tag
    meta = {"role": role, "encoder_tag": encoder_tag, "encoder_md5": encoder_md5(encoder_tag), "note": note}
    tmp = sidecar(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(sidecar(path))  # atomic
    return meta


def read_meta(path) -> dict:
    sc = sidecar(path)
    if not sc.exists():
        raise EncoderMismatch(f"no encoder-provenance sidecar for {path} (expected {sc.name}) - refusing to guess "
                              f"which encoder it belongs to")
    return json.loads(sc.read_text())


def tag_of(path) -> str:
    meta = read_meta(path)
    if meta["encoder_md5"] != encoder_md5(meta["encoder_tag"]):
        raise EncoderMismatch(f"{path}: sidecar says encoder {meta['encoder_tag']} md5 {meta['encoder_md5']} but that "
                              f"encoder on disk now has md5 {encoder_md5(meta['encoder_tag'])} - encoder changed without "
                              f"re-tagging/rebuilding this file")
    return meta["encoder_tag"]


def assert_encoder_match(model_path, embeddings, context: str = "") -> str:
    """model_path: a model file (sidecar role 'model'). embeddings: an embedding-store PATH or a bare tag string.
    Raises EncoderMismatch unless they are the same encoder. Returns the shared tag."""
    m = tag_of(model_path)
    e = embeddings if embeddings in ENCODER_FILES else tag_of(embeddings)
    if m != e:
        raise EncoderMismatch(f"{context}: model {Path(model_path).name} expects encoder {m} but the embeddings are from "
                              f"{e} ({embeddings if embeddings not in ENCODER_FILES else 'tag'}) - refusing to score")
    return m


def assert_store(path, expected_tag: str, context: str = "") -> None:
    t = tag_of(path)
    if t != expected_tag:
        raise EncoderMismatch(f"{context}: {Path(path).name} carries {t} embeddings, expected {expected_tag}")


def tag_all_current_files() -> list[tuple[str, str, str]]:
    """Writes sidecars for every model / embedding store whose encoder has been VERIFIED (see
    outputs/toolkit_embedding_provenance_table.csv). Returns (file, role, tag) rows. Canonical BatteryLife parquets are
    deliberately NOT tagged here - they are tagged by swap_in_corrected_batterylife_parquets.py when swapped in."""
    m, d = ROOT / "models", ROOT / "data" / "processed"
    plan = [
        (m / "ica_encoder.pt", "encoder", OLD), (m / "_candidate_ica_encoder.pt", "encoder", CAND),
        (m / "xgb_soh_fusion.json", "model", OLD),
        (m / "_experimental_xgb_soh_fusion_extended_reformulation.json", "model", OLD),
        (m / "ocsvm_model.pkl", "model", OLD), (m / "ocsvm_scaler.pkl", "model", OLD),
        (m / "_candidate_multisource.json", "model", CAND), (m / "_candidate_multisource_balanced.json", "model", CAND),
        (m / "_candidate_ocsvm.pkl", "model", CAND), (m / "_candidate_ocsvm_scaler.pkl", "model", CAND),
        (m / "_source_profiles.pkl", "model", CAND),
        (m / "_builtin_battery_trust.csv", "embeddings", CAND),
        (d / "fusion_embeddings.csv", "embeddings", OLD),
        (d / "fusion_embeddings_multisource.csv", "embeddings", CAND),
        (d / "candidate_fusion_train_range.json", "embeddings", CAND),
        (d / "stage5_1_oxford_merged.parquet", "embeddings", OLD),
        (d / "stage5_1_hust_merged.parquet", "embeddings", OLD),
        (d / "stage5_1_xjtu_merged.parquet", "embeddings", OLD),
        (d / "old_encoder_embeddings_batterylife_corrected.parquet", "embeddings", OLD),
    ]
    rows = []
    for path, role, tag in plan:
        if path.exists():
            write_meta(path, role, tag)
            rows.append((str(path.relative_to(ROOT)), role, tag))
    return rows
