# battery-prognostics

Battery cycle-life / capacity-fade prognostics workspace, pulling together
three public li-ion cycling datasets: NASA PCoE, CALCE (CS2), and MIT/Stanford/TRI
(Severson et al., Nature Energy 2019).

## Structure

```
data/
  raw/
    nasa/    - NASA PCoE .mat files (B0005, B0006, B0007, B0018)
    calce/   - CALCE CS2 cell .xlsx sessions (CS2_35, CS2_36, CS2_37)
    mit/     - MIT/BEEP quickstart example + full-dataset batch .mat files
  processed/ - cleaned/merged tables (not yet populated)
src/
  load_nasa.py   - loads NASA .mat files, prints struct/cycle structure
  load_calce.py  - loads CALCE .xlsx sessions into pandas, per-cell
  load_mit.py    - BEEP quickstart pattern + raw HDF5 batch-file structure
notebooks/   - exploratory analysis (not yet populated)
outputs/     - figures, reports (not yet populated)
models/      - trained model artifacts (not yet populated)
.venv/       - Python 3.14 virtualenv (numpy, scipy, pandas, h5py, beep, ...)
requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

## Running the loaders

```bash
python src/load_nasa.py
python src/load_calce.py
python src/load_mit.py
```

See `STATUS.md` for current dataset status, sources, and known caveats.
