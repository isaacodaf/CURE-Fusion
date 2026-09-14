# Scientific tables

The thirteen tables cover detection, utility prediction, action selection and computational cost. Each CSV and Markdown view is regenerated from the included scientific operands by `export_tables.py`. The [machine-readable index](index.json) supplies exact source files, hashes, JSON pointers and rounding for each numeric cell.

| File | Table | Scientific table | Metric and units | Views |
| --- | --- | --- | --- | --- |
| 01 | Table 1 | Current SEW study and input contract | Counts, input contracts and fixed fitting protocol | [CSV](table_01_study_contract.csv) · [Markdown](table_01_study_contract.md) |
| 02 | Table 2 | Four-class development COCO box AP | AP (%), IoU 0.50:0.05:0.95; 992 reused development frames; one detector seed | [CSV](table_02_matched_detection.csv) · [Markdown](table_02_matched_detection.md) |
| 03 | Table 3 | Online tensor-to-output cost | Median and p95 milliseconds; peak allocator memory in GiB; batch one, FP32 | [CSV](table_03_online_efficiency.csv) · [Markdown](table_03_online_efficiency.md) |
| 04 | Table 4 | Complete current per-class endpoints | AP (%), IoU 0.50:0.05:0.95; every method and condition | [CSV](table_04_per_class_endpoints.csv) · [Markdown](table_04_per_class_endpoints.md) |
| 05 | Table 5 | Utility errors against constant predictors | RMSE / MAE in teacher-risk units; 2,521 development objects; mean/median fitted on 10,964 optimization objects, zero unfitted | [CSV](table_05_constant_errors.csv) · [Markdown](table_05_constant_errors.md) |
| 06 | Table 6 | Relative error reductions against constants | Percent error reduction; pointwise descriptive 95% intervals from 2,000 paired recording draws | [CSV](table_06_constant_contrasts.csv) · [Markdown](table_06_constant_contrasts.md) |
| 07 | Table 7 | Trained utility readouts | RMSE / MAE in teacher-risk units; 90% estimated-target-mean coverage (%) | [CSV](table_07_trained_readouts.csv) · [Markdown](table_07_trained_readouts.md) |
| 08 | Table 8 | Paired trained-readout error contrasts | Percent error reduction; descriptive 95% intervals; count of eight recording omissions with lower CURE error | [CSV](table_08_trained_contrasts.csv) · [Markdown](table_08_trained_contrasts.md) |
| 09 | Table 9 | Separate 2D fitting gates | Fixed optimization prefixes and updates; all-class geometric/AP gate; no generalization claim | [CSV](table_09_paired_2d_fitting.csv) · [Markdown](table_09_paired_2d_fitting.md) |
| 10 | Table 10 | Five-seed stateless fusion | Factual and natural-adverse AP (%), mean ± sample SD across five detector seeds; historical population | [CSV](table_10_historical_five_seed.csv) · [Markdown](table_10_historical_five_seed.md) |
| 11 | Table 11 | Frozen-controller routing sensitivity | First 64 fitting frames; AP (%) and equal-frame task risk; original/corrected camera temperature; no trained efficacy comparison | [CSV](table_11_routing_sensitivity.csv) · [Markdown](table_11_routing_sensitivity.md) |
| 12 | Table 12 | Original summary-based action study | COCO AP (%), mean ± sample SD across five controller seeds; fixed references have no SD | [CSV](table_12_summary_actions.csv) · [Markdown](table_12_summary_actions.md) |
| 13 | Table 13 | Query-set action study | COCO AP (%), mean ± sample SD across five controller seeds; fresh thinning belongs to this action bank | [CSV](table_13_queryset_actions.csv) · [Markdown](table_13_queryset_actions.md) |

Full-precision regenerated result tables, complete seed runs and omission analyses are in [raw/](raw/index.json).
