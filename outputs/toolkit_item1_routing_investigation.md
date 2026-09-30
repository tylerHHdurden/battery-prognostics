# Item 1 routing investigation (read-only; numbers from outputs/toolkit_item1_routing_investigation.csv, recomputed R2/MAE, R2 matches finalpass_item1_labelfree_routing.csv exactly)

Routing was selected on held-out data (a deployment choice, not an unbiased result). "Deployed" = pre-candidate dataset-aware table (EXTENDED_ROUTED_DATASETS = CALCE/Oxford/HUST -> extended; XJTU/NASA/MIT -> base), the table Item 1 scored.
Caveat: src/live_inference.py currently has USE_MULTISOURCE_CANDIDATE = True, which routes CALCE/Oxford/HUST/XJTU to a separate multi-source model; that model is not part of Item 1's base-vs-extended comparison.

| dataset | deployed pick | label-free pick | base R2 / MAE | ext R2 / MAE | true winner (R2; MAE) | R2 margin (ext-base) / MAE margin (base-ext) | deployed right | label-free right |
|---|---|---|---|---|---|---|---|---|
| NASA | base | n/a | no held-out pair (train pool) | | not testable | | not testable | not testable |
| MIT | base | n/a | no held-out pair (train pool) | | not testable | | not testable | not testable |
| CALCE | extended | base | 0.568 / 10.441 | 0.740 / 6.514 | extended; extended | +0.172 / +3.927 | yes | no |
| Oxford | extended | base | -2.694 / 12.510 | 0.953 / 1.288 | extended; extended | +3.647 / +11.222 | yes | no |
| HUST | extended | base | -0.152 / 5.715 | 0.800 / 2.565 | extended; extended | +0.952 / +3.151 | yes | no |
| XJTU | base | base | -1.062 / 6.449 | -1.772 / 7.594 | base; base | -0.710 / -1.146 | yes | yes |

Non-app BatteryLife sources (INVALIDATED-scope, rerun on corrected embeddings; context only; 9 sources, not 7):
| source | deployed | label-free | base R2 / MAE | ext R2 / MAE | true winner | deployed right | label-free right |
|---|---|---|---|---|---|---|---|
| ul_pur | base | base | 0.044 / 3.542 | -0.605 / 7.005 | base | yes | yes |
| hnei | base | base | -0.101 / 14.798 | 0.785 / 7.306 | extended | no | no |
| snl | base | base | 0.056 / 6.210 | 0.442 / 4.850 | extended | no | no |
| mich | base | base | 0.533 / 8.053 | 0.754 / 5.737 | extended | no | no |
| mich_exp | base | base | 0.718 / 4.303 | 0.309 / 10.024 | base | yes | yes |
| rwth | base | base | -0.473 / 22.923 | 0.503 / 14.521 | extended | no | no |
| stanford | base | base | 0.275 / 15.590 | 0.809 / 3.749 | extended | no | no |
| stanford_2 | base | base | 0.247 / 15.649 | 0.806 / 3.829 | extended | no | no |
| isu_ilcc | base | base | 0.114 / 29.619 | 0.180 / 23.504 | extended | no | no |

R2 and MAE winners agree on all 13 sets. Deployed correct 6/13 (CALCE, Oxford, HUST, XJTU, ul_pur, mich_exp); label-free 3/13 (XJTU, ul_pur, mich_exp); extended is true winner on 10/13.

(1) Six-app table: no change proposed. Deployed picks are right on all four testable app datasets (CALCE/Oxford/HUST extended, XJTU base) by both R2 and MAE; NASA/MIT cannot be scored (train pool).
(2) The 8/13 -> 3/13 (rule), 8/13 -> 10/13 (extended winner) and 8/13 -> 6/13 (deployed) changes come from BatteryLife rows only: CALCE/Oxford/HUST/XJTU R2 values are identical in the first Item 1 commit (85dbe48) and now. All nine BatteryLife rows changed in R2; winners flipped on snl and mich (base -> extended); the rule pick flipped to base on hnei, snl, stanford, stanford_2, isu_ilcc.
(3) Routing was selected on held-out data, so these are deployment choices, not unbiased results.
