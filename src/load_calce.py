"""
Load CALCE CS2 battery cell data (CS2_35, CS2_36, CS2_37) with pandas.

Source: https://calce.umd.edu/battery-data (Prismatic Cells > CS2 Battery >
Type 2, "Cycled at constant current of 1C")
Direct downloads: https://web.calce.umd.edu/batteries/data/CS2_{35,36,37}.zip

Each cell's zip contains one Arbin-exported .xlsx file per test session
(named by date, e.g. CS2_35_10_15_10.xlsx), each with two sheets:
    Info            - free-form test report header (not tabular)
    Channel_x-xxx   - the actual cycler log with columns:
        Data_Point, Test_Time(s), Date_Time, Step_Time(s), Step_Index,
        Cycle_Index, Current(A), Voltage(V), Charge_Capacity(Ah),
        Discharge_Capacity(Ah), Charge_Energy(Wh), Discharge_Energy(Wh),
        dV/dt(V/s), Internal_Resistance(Ohm), Is_FC_Data,
        AC_Impedance(Ohm), ACI_Phase_Angle(Deg)

This loader concatenates all session files for a cell into one DataFrame,
sorted by Date_Time, with a `source_file` column to keep provenance.
"""

import zipfile
from pathlib import Path

import pandas as pd

CALCE_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "calce"
CELL_IDS = ["CS2_35", "CS2_36", "CS2_37"]


def _extract_if_needed(cell_id: str) -> Path:
    extract_dir = CALCE_DIR / "extracted" / cell_id
    if not extract_dir.exists():
        with zipfile.ZipFile(CALCE_DIR / f"{cell_id}.zip") as zf:
            zf.extractall(extract_dir)
    return extract_dir


def load_cell(cell_id: str) -> pd.DataFrame:
    """Load and concatenate all Arbin xlsx session logs for one CALCE cell."""
    extract_dir = _extract_if_needed(cell_id)
    xlsx_files = sorted(extract_dir.rglob("*.xlsx"))

    frames = []
    for f in xlsx_files:
        xl = pd.ExcelFile(f)
        data_sheets = [s for s in xl.sheet_names if s.startswith("Channel")]
        for sheet in data_sheets:
            df = pd.read_excel(xl, sheet_name=sheet)
            df["source_file"] = f.name
            frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full["Date_Time"] = pd.to_datetime(full["Date_Time"])
    full = full.sort_values("Date_Time").reset_index(drop=True)
    return full


def describe_cell(cell_id: str) -> pd.DataFrame:
    df = load_cell(cell_id)
    print(f"\n=== {cell_id} ===")
    print(f"Rows: {len(df):,}  Session files: {df['source_file'].nunique()}")
    print(f"Date range: {df['Date_Time'].min()} -> {df['Date_Time'].max()}")
    print(f"Cycle range: {df['Cycle_Index'].min()} -> {df['Cycle_Index'].max()}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())

    # Cycle_Index resets to 1 at the start of every session file, AND
    # Discharge_Capacity(Ah) is cumulative *within* a session file rather than
    # per-cycle (it keeps climbing across cycles, then resets to 0 at the
    # next file). So the true per-cycle discharge capacity is the diff of the
    # per-cycle cumulative max, within each file, in date order.
    per_cycle = (
        df.groupby(["source_file", "Cycle_Index"], sort=False)
        .agg(cum_discharge=("Discharge_Capacity(Ah)", "max"),
             Date_Time=("Date_Time", "min"))
        .reset_index()
        .sort_values("Date_Time")
    )
    per_cycle["discharge_capacity"] = (
        per_cycle.groupby("source_file")["cum_discharge"].diff()
        .fillna(per_cycle["cum_discharge"])
    )
    discharge_caps = per_cycle[per_cycle["discharge_capacity"] > 0]["discharge_capacity"]
    print(f"\nDischarge capacity fade across {len(discharge_caps)} logged cycles: "
          f"{discharge_caps.iloc[0]:.4f} Ah -> {discharge_caps.iloc[-1]:.4f} Ah")
    return df


def main() -> None:
    for cell_id in CELL_IDS:
        describe_cell(cell_id)


if __name__ == "__main__":
    main()
