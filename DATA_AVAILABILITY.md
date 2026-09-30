# Data Availability Statement

Every dataset used in this project, its access method, and license/
terms of use as documented at the point each was integrated (this
project's own `DEVELOPMENT_LOG.md` and loader-script docstrings are the
source for every URL/DOI below - not re-derived or assumed here). None
of the raw data is redistributed in this repository (all `data/raw/`
paths are git-ignored) - every dataset is publicly re-downloadable from
its own original host at the URLs below.

| Dataset | Cells used | Source / access | License / terms (as stated by the host) |
|---|---|---|---|
| NASA PCoE Li-ion Battery Aging Dataset | B0005, B0006, B0007, B0018 (+15 more in the expanded pool) | `https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip` (NASA Ames PCoE repository) | NASA Open Data / public domain (U.S. Government work); no additional restriction stated by the host |
| NASA Randomized Battery Usage Data Set | (training-pool candidate, not a held-out eval set) | `https://phm-datasets.s3.amazonaws.com/NASA/11.+Randomized+Battery+Usage+Data+Set.zip`; cross-confirmed via NASA's own Zenodo deposit, DOI `10.5281/zenodo.15277374` | Same as above (official NASA PCoE / NASA-deposited Zenodo record) |
| MIT/Stanford/MIT-Stanford fast-charging dataset (Severson et al., *Nature Energy* 2019) | 28-cell curated subset (`mit_subset.json`), expanded later | `data.matr.io` (BEEP quickstart mirror; the live API has been non-functional since project start, worked around via the archived quickstart CSV files) | CC BY 4.0, per the dataset's own publication and hosting convention |
| CALCE CS2 battery data | CS2_35, CS2_36, CS2_37 | `https://calce.umd.edu/battery-data` (direct: `https://web.calce.umd.edu/batteries/data/CS2_{35,36,37}.zip`) | Free for research use per CALCE's own data-sharing page; no redistribution license file provided by the host - used here under that stated research-use permission |
| Oxford Battery Degradation Dataset 1 | 8 cells | `ora.ox.ac.uk` (Oxford Research Archive), 254 MB `.mat` file | ORA's standard open-access research-data terms (institutional repository; attribution expected) |
| HUST battery dataset (Ma, G. et al., *Energy & Environmental Science* 2022, vol. 15) | 77 cells | Mendeley Data, 1.19 GB zip | CC BY 4.0 (Mendeley Data's default license for this record) |
| XJTU battery dataset (Wang et al.) | 55 cells (47 used - 8 Batch-6/Sim_satellite cells excluded, shared SOH-labeling artifact, disclosed elsewhere in `DEVELOPMENT_LOG.md`) | Zenodo, 2.44 GB zip | Zenodo record's own stated license (CC BY 4.0 convention for this record type) |
| BatteryLife (integrated sub-sources: UL_PUR, HNEI, SNL, MICH, MICH_EXP, RWTH, Stanford, Stanford_2, ISU_ILCC) | 170 batteries across 9 of BatteryLife's 18 original raw sources (the other 9 - CALB, ISU-ILCC's own further splits, NA-ion, ZN-coin, etc. - not locally available, disclosed in Part B's own entry) | `https://zenodo.org/api/records/17756951/files/<Source>.zip/content` (BatteryLife's own Zenodo deposit); paper: arXiv:2502.18807 | Per BatteryLife's own Zenodo record terms (verified as a genuine, real, open-access deposit before use - not assumed) |

**Not integrated, confirmed real and open but out of this project's own
scope** (disclosed in the 18-item pass, Part B item 10): Stroebl et al.
(Samsung INR21700-50E, *Scientific Data* 2024, figshare, ~10GB
compressed) and Luh & Blank (KIT NMC/C-SiO, *Scientific Data* 2024,
Zenodo/RADAR4KIT DOI `10.35097/kww7jv8ajuvchcah`) - neither is among
BatteryLife's own integrated sources; both would require their own
separate integration effort.

**Derived/processed data**: every `data/processed/*.parquet` file in
this repository is generated from the raw sources above by this
project's own scripts (`src/build_*`, `src/run_stage*_feature*`, etc.)
- fully regenerable, not independently redistributed raw data.

**Models**: all trained model weights under `models/` (including every
`_experimental_*` file from this pass) are this project's own original
training artifacts, licensed under this repository's own terms - not
subject to any upstream dataset license beyond the training-data-use
permissions above.
