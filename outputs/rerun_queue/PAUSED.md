# PAUSED 2026-09-30 12:33 - user deadline interrupt (report due 2 PM). Resume ONLY after the user says the report is submitted.

## Queue (outputs/rerun_queue/run_queue.sh) - KILLED
[queue] START itemA 12:09:34
[queue] END   itemA rc=0 secs=643 12:20:17
[queue] START itemB 12:20:17
[queue] END   itemB rc=0 secs=538 12:29:15
[queue] START lodo_check1 12:29:15

itemA DONE (before/after posted). itemB DONE (538s, 12:29; before/after NOT yet posted). lodo_check1 was running and was KILLED mid-run -> rerun it. Remaining in order: lodo_check1, phase2c, phase3a, item1, item2, item3, item5a, item5d, partB7, partB8, partB9, check2, itemE, itemC (+ static-baseline recompute, deployed-OCSVM flag-rate recheck).

## Phase 2B rerun (corrected embeddings) - SUSPENDED (NtSuspendProcess), Windows PIDs 22972 + 11280. Folds done: 12/16. Resume: Get-Process -Id 22972,11280 then NtResumeProcess (or restart the script).

## regenerate_batterylife_fusion_columns.py - SUSPENDED, PIDs 24920 + 28536. Sources done (isu_ilcc done; only tongji remained):
[regen] ul_pur: 2245 rows | max|fusion| 8.47 | >10x p99.9: 0 | NaN rows 0 | diff vs indepe
[regen] hnei: 15155 rows | max|fusion| 9.39 | >10x p99.9: 0 | NaN rows 0 | diff vs indepen
[regen] snl: 38880 rows | max|fusion| 12.32 | >10x p99.9: 0 | NaN rows 5 | diff vs indepen
[regen] mich: 19881 rows | max|fusion| 7.81 | >10x p99.9: 0 | NaN rows 0 | diff vs indepen
[regen] mich_exp: 6545 rows | max|fusion| 10.44 | >10x p99.9: 0 | NaN rows 0 | diff vs ind
[regen] rwth: 22095 rows | max|fusion| 26.67 | >10x p99.9: 0 | NaN rows 0 | diff vs indepe
[regen] stanford: 5633 rows | max|fusion| 6.04 | >10x p99.9: 0 | NaN rows 21 | diff vs ind
[regen] stanford_2: 8465 rows | max|fusion| 6.05 | >10x p99.9: 0 | NaN rows 29 | diff vs i
[regen] isu_ilcc: 45229 rows | max|fusion| 3.94 | >10x p99.9: 0 | NaN rows 0 | diff vs ind

## Not yet done: step 1 tagging (encoder_provenance.tag_all_current_files), swap_in_corrected_batterylife_parquets.py, deferred guards in researchpass_partA_common/stage1_common; steps 3 (OOD redesign: run src/trust_operating_point.py first), 4, Phase 3B/4/5.

## Uncommitted code changes exist (app.py, src/live_inference.py, trust_report.py, phase2b loader, builder, new scripts) - NOT touched further during the report build.
