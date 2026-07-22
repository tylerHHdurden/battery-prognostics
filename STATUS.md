# Dataset acquisition status — 2026-07-21

## 1. NASA PCoE — DONE

- Source: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
- The zip contains 6 sub-zips grouped by cell IDs; B0005/B0006/B0007/B0018 are
  all in `1. BatteryAgingARC-FY08Q4.zip`.
- Extracted to `data/raw/nasa/B0005_B0006_B0007_B0018/{B0005,B0006,B0007,B0018}.mat`
- Loaded with `scipy.io.loadmat(..., simplify_cells=True)`.

**Structure** (`src/load_nasa.py`): each `.mat` has a top-level struct named
after the cell (e.g. `B0005`) with a `cycle` field — a sequence of
charge / discharge / impedance operations:

| cell | total logged ops | charge | discharge | impedance |
|---|---|---|---|---|
| B0005 | 616 | 170 | 168 | 278 |
| B0006 | 616 | 170 | 168 | 278 |
| B0007 | 616 | 170 | 168 | 278 |
| B0018 | 319 | 134 | 132 | 53 |

- `charge`/`discharge` entries carry `Voltage_measured`, `Current_measured`,
  `Temperature_measured`, `Time`, plus load/charge-specific fields; discharge
  also carries a single `Capacity` (Ah) value — the actual capacity-fade
  signal.
- `impedance` entries carry complex-valued EIS arrays (`Battery_impedance`,
  `Re`, `Rct`, etc.)
- Example capacity fade: B0005 goes from 1.86 Ah (cycle 1) to 1.33 Ah (last
  discharge), consistent with the well-known NASA aging trend.

No caveats — this dataset loaded cleanly on the first attempt.

## 2. CALCE (CS2_35, CS2_36, CS2_37) — DONE

- Source: https://calce.umd.edu/battery-data → Prismatic Cells → CS2 Battery
  → Type 2 ("Cycled at constant current of 1C")
- Direct zips: `https://web.calce.umd.edu/batteries/data/CS2_{35,36,37}.zip`
- Each zip is a folder of dated Arbin-exported `.xlsx` files (one per test
  session), each with an `Info` sheet (free-text header, not loaded) and a
  `Channel_x-xxx` data sheet.
- Loaded and concatenated per cell with `src/load_calce.py` (pandas +
  openpyxl), sorted by `Date_Time`.

**Structure**:

| cell | rows | session files | cycle range | date range |
|---|---|---|---|---|
| CS2_35 | 266,914 | 25 | 1–50 (per session) | 2010-08-16 → 2011-02-03 |
| CS2_36 | 269,656 | 26 | 1–50 (per session) | 2010-08-16 → 2011-02-02 |
| CS2_37 | 290,998 | 27 | 1–50 (per session) | 2010-08-16 → 2011-02-02 |

Columns: `Data_Point, Test_Time(s), Date_Time, Step_Time(s), Step_Index,
Cycle_Index, Current(A), Voltage(V), Charge_Capacity(Ah),
Discharge_Capacity(Ah), Charge_Energy(Wh), Discharge_Energy(Wh), dV/dt(V/s),
Internal_Resistance(Ohm), Is_FC_Data, AC_Impedance(Ohm), ACI_Phase_Angle(Deg)`

**Caveats worth knowing before you build features on this:**
- `Cycle_Index` **resets to 1 at the start of every session file** — it is
  not a global cycle counter. `load_calce.py` groups by
  `(source_file, Cycle_Index)` in date order to build a true per-cycle
  sequence.
- `Discharge_Capacity(Ah)`/`Charge_Capacity(Ah)` are **cumulative within a
  session file**, not per-cycle — they climb across cycles and reset to 0 at
  the next file. The loader's capacity-fade summary takes a `diff()` within
  each file to recover true per-cycle capacity.
- Verified fade trend after the fix: CS2_35 1.14→0.30 Ah (932 cycles),
  CS2_36 1.14→0.17 Ah (973 cycles), CS2_37 1.13→0.19 Ah (1038 cycles) —
  plausible EOL-range fade for a ~1.1 Ah rated cell, all three cells
  consistent with each other.

## 3. MIT / Stanford / TRI (Severson et al., Nature Energy 2019) — DONE, beyond scope

Time-boxed to 45 minutes; used ~20 of them.

- `pip install beep` worked cleanly (Python 3.14, one harmless warning about
  `monty`'s `yaml` extra).
- Ran the **official quickstart** from
  https://tri-amdd.github.io/beep/Python%20tutorials/1%20-%20quickstart/ :
  downloaded the one documented example cell
  (`2017-05-12_6C-50per_3_6C_CH36.csv` + its metadata CSV, via
  `data.matr.io/1/api/v1/file/{id}/download`), loaded it with
  `beep.structure.cli.auto_load`, and validated it — **works end-to-end**:
  919,051 rows, 15 columns, `is_valid=True`.
  - One gotcha: BEEP's `auto_load` requires an **absolute path**, or it
    raises `ValueError: ... is not absolute!`.
  - Another: the `/download` API endpoint 301-redirects to a static asset
    path — `curl` needs `-L` to follow it (data.matr.io is now a static
    S3/CloudFront snapshot of what used to be a live Girder-based backend;
    there is no working live API behind it anymore, see below).

**Went looking for more file IDs than the single quickstart example, per your
instructions.** Result:
- **data.matr.io's own API has no working list/search endpoint.** Every
  `/api/v1/{collection,folder,item,resource/search}` request 301-redirects
  back to the SPA's client-side router — the site is a static export, not a
  live Girder server. Only exact, previously-known `/file/{id}` and
  `/file/{id}/download` paths resolve (they were apparently snapshotted
  as static objects at those exact URLs).
- **BEEP's own GitHub repo** (`TRI-AMDD/beep`) only ever mentions the 2 file
  IDs from the quickstart doc — nothing more in code, tests, or issues.
- **Found the real payoff in `microsoft/BatteryML`** (a third-party repo,
  found via a GitHub code search for other public repos referencing
  `data.matr.io/1/api/v1/file`): its
  `batteryml/preprocess/download.py` has a hardcoded manifest of **4 whole-batch
  `.mat` files**, each a v7.3/HDF5 MATLAB struct holding *every cell in that
  test batch* (not just one cell each) — i.e. the entire published dataset,
  not a handful of extra files:

  | file | data.matr.io file id | size |
  |---|---|---|
  | MATR_batch_20170512.mat | 5c86c0b5fa2ede00015ddf66 | 3.03 GB |
  | MATR_batch_20170630.mat | 5c86bf13fa2ede00015ddd82 | 2.01 GB |
  | MATR_batch_20180412.mat | 5c86bd64fa2ede00015ddbb2 | 3.24 GB |
  | MATR_batch_20190124.mat | 5dcef152110002c7215b2c90 | 2.60 GB |

  These correspond to the 3 original Severson et al. batches
  (2017-05-12, 2017-06-30, 2018-04-12) plus one later follow-up batch
  (2019-01-24), together covering **185 A123 LFP cells** (46 + 48 + 46 + 45)
  under various fast-charge protocols — this is effectively the complete
  MIT dataset, recovered entirely from a third-party repo's download script
  rather than any official manifest. All 4 files downloaded successfully
  (verified byte-for-byte against the Content-Length reported by
  data.matr.io) and all 4 were opened and inspected with `h5py` — cell
  counts, charge policies, cycle lives, and capacity-fade curves all look
  physically sane, e.g.:

  | batch | cells | example charge policy | example cycle life |
  |---|---|---|---|
  | 20170512 | 46 | 3.6C(80%)-3.6C | 1190 |
  | 20170630 | 48 | 1C(4%)-6C | 300 |
  | 20180412 | 46 | 5C(67%)-4C-newstructure | 1009 |
  | 20190124 | 45 | 4.8-5.2-5.2-4.16 | 857 |
- Did **not** find a file manifest in the Severson et al. Nature Energy
  supplementary materials themselves (didn't need to, once the BatteryML
  manifest turned up — stopped there to stay inside the time box).

**Schema** (via `microsoft/BatteryML`'s `preprocess_MATR.py`, confirmed by
directly opening the files with `h5py` — they're HDF5/v7.3, `scipy.io.loadmat`
cannot open them): top-level `batch` group, MATLAB object-reference arrays
`summary`, `cycle_life`, `policy_readable`, `cycles`, one entry per cell.
Per-cell `summary` has `IR, QCharge, QDischarge, Tavg, Tmin, Tmax,
chargetime, cycle`; per-cell `cycles` has per-cycle time series
`I, Qc, Qd, Qdlin, T, Tdlin, V, discharge_dQdV, t`.

**Bottom line: MIT data acquisition went well past "1-3 files" — all 4
official batch files (the whole dataset) plus the single documented
quickstart example were downloaded and loaded successfully within the time
box.** No manual follow-up needed for broader access; if anything is still
wanted beyond this, it would be the CALCE-style per-cell provenance metadata
(e.g. exact channel/test-stand IDs), which isn't in the batch files but isn't
needed for prognostics modeling either.

## What's left / not done in this pass
- `data/processed/` is empty — no cleaning/merging across datasets yet.
- No unified schema across NASA/CALCE/MIT (they use different units, cycle
  conventions, and file formats by design of this task — normalization is a
  follow-up step).
- All three CALCE cells (CS2_35/36/37) and all four MIT batch files are
  confirmed loading correctly as of this writing.
