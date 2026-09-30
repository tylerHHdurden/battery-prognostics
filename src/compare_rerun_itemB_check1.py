"""Before/after for the two LODO-type reruns on corrected embeddings (Item B, and lodo_check1 family holdout).
BEFORE = the committed (raw-X) file in git HEAD; AFTER = the corrected rerun on disk. Writes a new CSV only."""
import io
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def head(path):
    out = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
    return pd.read_csv(io.BytesIO(out))


rows = []
# Item B
b0 = head("outputs/finalpass2_itemB_lodo_results.csv").set_index("held_out_source")
b1 = pd.read_csv(ROOT / "outputs/finalpass2_itemB_lodo_results.csv").set_index("held_out_source")
for s in b1.index:
    if s not in b0.index: continue
    rows.append({"analysis": "Item B LODO (R2)", "source": s, "before_r2": b0.loc[s, "lodo_r2"], "after_r2": b1.loc[s, "lodo_r2"],
                 "before_mae": b0.loc[s, "lodo_mae"], "after_mae": b1.loc[s, "lodo_mae"],
                 "before_beats_nasa_mit": b0.loc[s, "lodo_beats_nasa_mit_only"], "after_beats_nasa_mit": b1.loc[s, "lodo_beats_nasa_mit_only"]})
# check1 family holdout: old file (unchanged in git) vs the corrected rerun
c0 = pd.read_csv(ROOT / "outputs/finalpass3_check1_lodo_family_holdout.csv")
c1 = pd.read_csv(ROOT / "outputs/toolkit_lodo_family_holdout_rerun_corrected.csv")
print("check1 old cols:", list(c0.columns))
print("check1 new cols:", list(c1.columns))
df = pd.DataFrame(rows)
df["r2_change"] = df["after_r2"] - df["before_r2"]
df["verdict_flipped"] = df["before_beats_nasa_mit"].notna() & df["after_beats_nasa_mit"].notna() & (df["before_beats_nasa_mit"].astype(str) != df["after_beats_nasa_mit"].astype(str))
df.to_csv(ROOT / "outputs/toolkit_rerun_before_after_itemB.csv", index=False)
pd.set_option("display.width", 250)
print(df.round(3).to_string(index=False))
