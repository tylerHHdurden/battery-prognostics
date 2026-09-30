"""For every output CSV rewritten by the corrected-embedding reruns, compare git HEAD (BEFORE, raw-X) vs disk (AFTER):
per-file count of numeric cells that changed and the largest absolute / relative change, plus the top movers."""
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
files = subprocess.run(["git", "status", "--short", "outputs"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
files = [l[3:].strip() for l in files if l.startswith(" M") and l.strip().endswith(".csv")]
rows, movers = [], []
for f in files:
    try:
        old = pd.read_csv(io.BytesIO(subprocess.run(["git", "show", f"HEAD:{f}"], cwd=ROOT, capture_output=True, check=True).stdout))
        new = pd.read_csv(ROOT / f)
    except Exception as e:
        rows.append({"file": f, "note": f"unreadable: {e}"}); continue
    if old.shape != new.shape or list(old.columns) != list(new.columns):
        rows.append({"file": f, "note": f"shape/columns changed {old.shape} -> {new.shape}"}); continue
    num = [c for c in old.columns if pd.api.types.is_numeric_dtype(old[c]) and pd.api.types.is_numeric_dtype(new[c]) and old[c].dtype != bool and new[c].dtype != bool]
    changed, maxabs, maxrel = 0, 0.0, 0.0
    key = old.columns[0]
    for c in num:
        d = (new[c] - old[c]).abs()
        m = d > 1e-9
        changed += int(m.sum())
        if m.any():
            maxabs = max(maxabs, float(d[m].max()))
            rel = (d / old[c].abs().clip(lower=1e-9))[m]
            maxrel = max(maxrel, float(rel.max()))
            i = d.idxmax()
            movers.append({"file": Path(f).name, "column": c, "row": str(old.loc[i, key]), "before": old.loc[i, c], "after": new.loc[i, c], "abs_change": float(d[i])})
    rows.append({"file": Path(f).name, "numeric_cells": int(old[num].notna().sum().sum()), "cells_changed": changed, "max_abs_change": round(maxabs, 4), "max_rel_change": round(maxrel, 3), "note": ""})
df = pd.DataFrame(rows)
df.to_csv(ROOT / "outputs" / "toolkit_rerun_change_summary.csv", index=False)
pd.DataFrame(movers).sort_values("abs_change", ascending=False).to_csv(ROOT / "outputs" / "toolkit_rerun_top_movers.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(df.to_string(index=False))
