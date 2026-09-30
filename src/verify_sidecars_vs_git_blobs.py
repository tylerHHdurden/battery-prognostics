"""Every encoder-provenance sidecar must agree with hashes computed from the COMMITTED (LF) blobs, not just the working-tree files.
Also checks that encoder_provenance._md5 still normalises CRLF for text files (defence in depth)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import encoder_provenance as ep  # noqa: E402


def blob(rel):
    return subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, check=True).stdout


def blob_md5(rel):
    b = blob(rel)
    return hashlib.md5(b.replace(b"\r\n", b"\n") if Path(rel).suffix in {".json", ".csv", ".txt", ".md"} else b).hexdigest()


expected = {}
for tag, (w, s) in ep.ENCODER_FILES.items():
    wr, sr = str(w.relative_to(ROOT)).replace("\\", "/"), str(s.relative_to(ROOT)).replace("\\", "/")
    expected[tag] = hashlib.md5((blob_md5(wr) + blob_md5(sr)).encode()).hexdigest()
    assert expected[tag] == ep.encoder_md5(tag), f"{tag}: worktree-based md5 != committed-blob md5"
n_ok, bad = 0, []
for base in (ROOT / "models", ROOT / "data" / "processed"):
    for f in sorted(base.glob("*.meta.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if m.get("encoder_tag") in expected and "encoder_md5" in m:
            if m["encoder_md5"] == expected[m["encoder_tag"]]: n_ok += 1
            else: bad.append(f.name)
# defence in depth: CRLF and LF versions of the same text hash identically
tmp = ROOT / "outputs" / "_crlf_probe.json"
tmp.write_bytes(b'{"a": 1,\r\n "b": 2}\r\n'); h1 = ep._md5(tmp)
tmp.write_bytes(b'{"a": 1,\n "b": 2}\n'); h2 = ep._md5(tmp); tmp.unlink()
print(f"sidecars valid against committed blobs: {n_ok}; invalid: {bad}; CRLF-normalised hashing works: {h1 == h2}")
sys.exit(0 if not bad and h1 == h2 and n_ok > 0 else 1)
