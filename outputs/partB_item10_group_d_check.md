# Part B, item 10: Group D dataset accessibility check

Verifies (via direct WebSearch, not assumed) whether BatteryLife's raw
sources include the two previously-blocked Group D datasets.

## BatteryLife's own integrated raw sources (confirmed directly, item 6)
CALB, CALCE, HNEI, HUST, ISU_ILCC, MATR, MICH, MICH_EXP, NA-ion, RWTH,
SDU, SNL, Stanford, Stanford_2, Tongji, UL_PUR, XJTU, ZN-coin (18 total,
per the Zenodo record's own file listing, confirmed in item 6).

## Dataset 1: Stroebl et al., 279-cell Samsung INR21700-50E

- **Confirmed real and independently accessible**: Stroebl, Petersohn et
  al., "A multi-stage lithium-ion battery aging dataset using various
  experimental design methodologies," *Scientific Data* 11:1020 (2024).
  279 cells, 93 aging conditions, ~10GB compressed / ~100GB uncompressed,
  hosted on figshare (linked from the Nature Scientific Data article).
  Analysis code: github.com/fst2112/Multi-Stage-Lithium-Ion-Battery-Aging-Dataset-Analysis.
- **NOT among BatteryLife's 18 integrated raw sources** - no name/
  institution match against the list above. (RWTH is the closest
  same-country candidate but cites a DIFFERENT, unrelated 2021 paper -
  "One-shot battery degradation trajectory prediction with deep
  learning" - confirmed via its own README, not the same dataset.)

## Dataset 2: Luh & Blank, 228-cell NMC/C-SiO

- **Confirmed real and independently accessible**: Luh & Blank
  (Karlsruhe Institute of Technology), "Comprehensive battery aging
  dataset: capacity and impedance fade measurements of a lithium-ion
  NMC/C-SiO cell," *Scientific Data*, DOI 10.1038/s41597-024-03831-x.
  228 cells, >3 billion data points, hosted on RADAR4KIT/Zenodo, DOI
  10.35097/kww7jv8ajuvchcah. Example scripts:
  github.com/energystatusdata/bat-age-data-scripts.
- **NOT among BatteryLife's 18 integrated raw sources** - no match.

## Verdict

Neither Group D dataset is part of BatteryLife's own raw-source
integration - folding BatteryLife in (items 6-9) does NOT close this
gap. Both remain real, independently open-access, and downloadable
(figshare and Zenodo/RADAR4KIT respectively) - genuinely accessible,
just not through BatteryLife. Integrating either would need its own
separate adapter-writing and download effort, out of this pass's scope
(not attempted here).
