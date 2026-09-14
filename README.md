# CURE-Fusion

Counterfactual Sensor Utility and the Attribution to Routing Gap in Multimodal Detection

[Finished figures](figures/README.md) · [Scientific tables](tables/README.md)

Research code for a methodological study of Counterfactual Sensor Utility (CSU): signed removal attribution conditioned on a frozen teacher, factual object-risk operands and a declared intervention. CURE-Fusion provides the shared camera–radar detector interface used to test attribution prediction and routing separately. The study distinguishes learnable removal importance from the value of an executable routing action; the tested utility-guided controls do not establish a matched detection gain.

The package includes detector adapters, utility objectives, matched action selectors, trained readout controls, evaluation helpers, statistical analysis, all eight figures and all 13 scientific tables. The recorded populations, scientific results and execution boundaries remain those of the completed study.

## Quick start

The quickest route runs on CPU with Python 3.12. It checks the saved numerical results and regenerates all 13 tables; no GPU, dataset download or trained checkpoint is needed for this route.

```sh
git clone https://github.com/isaacodaf/CURE-Fusion.git
cd CURE-Fusion
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python reproduce_results.py --skip-figures --output outputs/quickstart
```

A successful run writes `outputs/quickstart/REPRODUCTION.json` with status `passed_saved_research_reproduction`. Use a new output directory for each run.

**[Full setup guide](SETUP.md)** covers Windows, figure regeneration, saved-detection evaluation, CUDA experiment requirements and troubleshooting. The [finished figures](figures/README.md) and [tables](tables/README.md) can also be viewed directly on GitHub.

## Browse finished results

All eight finished figures and all 13 scientific tables are included. No script needs to be run to view them. The four original camera examples remain in `results/sensor_pairs/` with their attribution.

<p><img src="figures/figure_01_architecture.png" width="400" alt="Counterfactual sensor utility architecture" /> <img src="figures/figure_06_paired_camera_radar.png" width="400" alt="Paired camera and radar inputs" /></p>

| Figure | Content | Exports |
| --- | --- | --- |
| 1 | Architecture | [PNG](figures/figure_01_architecture.png) · [PDF](figures/figure_01_architecture.pdf) · [SVG](figures/figure_01_architecture.svg) |
| 2 | Matched detection | [PNG](figures/figure_02_matched_detection.png) · [PDF](figures/figure_02_matched_detection.pdf) · [SVG](figures/figure_02_matched_detection.svg) |
| 3 | Per class detection | [PNG](figures/figure_03_per_class_detection.png) · [PDF](figures/figure_03_per_class_detection.pdf) · [SVG](figures/figure_03_per_class_detection.svg) |
| 4 | Utility readouts | [PNG](figures/figure_04_utility_readouts.png) · [PDF](figures/figure_04_utility_readouts.pdf) · [SVG](figures/figure_04_utility_readouts.svg) |
| 5 | Camera cases | [PNG](figures/figure_05_camera_cases.png) · [PDF](figures/figure_05_camera_cases.pdf) · [SVG](figures/figure_05_camera_cases.svg) |
| 6 | Paired camera radar | [PNG](figures/figure_06_paired_camera_radar.png) · [PDF](figures/figure_06_paired_camera_radar.pdf) · [SVG](figures/figure_06_paired_camera_radar.svg) |
| 7 | Queryset comparison | [PNG](figures/figure_07_queryset_comparison.png) · [PDF](figures/figure_07_queryset_comparison.pdf) · [SVG](figures/figure_07_queryset_comparison.svg) |
| 8 | Online cost | [PNG](figures/figure_08_online_cost.png) · [PDF](figures/figure_08_online_cost.pdf) · [SVG](figures/figure_08_online_cost.svg) |

[Table index: metrics, conditions, units and source mappings](tables/README.md)

| Table | TMLR table location and content | Views |
| --- | --- | --- |
| 01 | Table 1 (main): Current SEW study and input contract | [CSV](tables/table_01_study_contract.csv) · [Markdown](tables/table_01_study_contract.md) |
| 02 | Table 2 (main): Four-class development COCO box AP | [CSV](tables/table_02_matched_detection.csv) · [Markdown](tables/table_02_matched_detection.md) |
| 03 | Table 3 (main): Online tensor-to-output cost | [CSV](tables/table_03_online_efficiency.csv) · [Markdown](tables/table_03_online_efficiency.md) |
| 04 | Table 4 (Appendix A): Complete current per-class endpoints | [CSV](tables/table_04_per_class_endpoints.csv) · [Markdown](tables/table_04_per_class_endpoints.md) |
| 05 | Table 5 (Appendix C): Utility errors against constant predictors | [CSV](tables/table_05_constant_errors.csv) · [Markdown](tables/table_05_constant_errors.md) |
| 06 | Table 6 (Appendix C): Relative error reductions against constants | [CSV](tables/table_06_constant_contrasts.csv) · [Markdown](tables/table_06_constant_contrasts.md) |
| 07 | Table 7 (Appendix C): Trained utility readouts | [CSV](tables/table_07_trained_readouts.csv) · [Markdown](tables/table_07_trained_readouts.md) |
| 08 | Table 8 (Appendix C): Paired trained-readout error contrasts | [CSV](tables/table_08_trained_contrasts.csv) · [Markdown](tables/table_08_trained_contrasts.md) |
| 09 | Table 9 (Appendix D): Separate 2D fitting gates | [CSV](tables/table_09_paired_2d_fitting.csv) · [Markdown](tables/table_09_paired_2d_fitting.md) |
| 10 | Table 10 (Appendix E): Five-seed stateless fusion | [CSV](tables/table_10_historical_five_seed.csv) · [Markdown](tables/table_10_historical_five_seed.md) |
| 11 | Table 11 (Appendix F): Frozen-controller routing sensitivity | [CSV](tables/table_11_routing_sensitivity.csv) · [Markdown](tables/table_11_routing_sensitivity.md) |
| 12 | Table 12 (Appendix G): Original summary-based action study | [CSV](tables/table_12_summary_actions.csv) · [Markdown](tables/table_12_summary_actions.md) |
| 13 | Table 13 (Appendix G): Query-set action study | [CSV](tables/table_13_queryset_actions.csv) · [Markdown](tables/table_13_queryset_actions.md) |

The [figure index](figures/index.json) maps the final paper numbering to the plotting scripts’ preserved internal filenames. [Full-precision tables](tables/raw/index.json) retain complete conditions, seeds and mixed or null comparisons.

## Reproduce saved results

Use Python 3.12 and a virtual environment:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python reproduce_results.py --output outputs/reproduction
```

Choose a new output directory. The command verifies the package manifest, recomputes all 40 utility assessment cells, 12 constant comparisons and eight trained-readout comparisons from saved arrays, reproduces the fixed 2,000 paired recording draws, checks complete detection/action summaries, and writes CSV/JSON tables plus eight figures. It also regenerates and checks the 13 included publication table views and complete raw tables. `--skip-figures` runs the numerical and table portion alone.

All 16 detector endpoints, 64 class endpoints, 116 original-action runs, 116 richer-query runs, and the historical eleven-method comparison remain. Detection AP is read from the recorded complete results; this command does not re-evaluate detections or retrain models. Utility means, variances, errors and recording contrasts are recomputed. Its absolute FP64 comparison tolerance is 10⁻¹². All methods and conditions are retained, including mixed and null comparisons.

Exact PNG/PDF/SVG reproduction uses the pinned plotting packages plus Arial for Figure 1 and the Matplotlib-distributed DejaVu Sans fonts elsewhere. Arial is not bundled. If these fonts differ, the command keeps the generated files and fails exact-byte comparison. No network is used by the reproduction command.

The publication table views can also be regenerated separately:

```sh
.venv/bin/python export_tables.py --output outputs/publication_tables
```

The renderer derives numeric cells from the supplied results and protocols, using declared labels and display precision. It does not read manuscript files. Generated training outputs are separate from these published result assets.

## Evaluate supplied saved detections

```sh
.venv/bin/python -m pip install -r requirements-evaluation.txt
.venv/bin/python evaluate_saved.py \
  --ground-truth /data/ground_truth_coco.json \
  --ground-truth-sha256 EXACT_SHA256 \
  --predictions /data/predictions_coco.json \
  --predictions-sha256 EXACT_SHA256 \
  --output outputs/coco_metrics.json
```

This invokes pycocotools 2.0.11 with all supplied frames, four fixed categories and the original max-100 COCO accumulation. The fixed-four summary is unavailable when a class has no ground-truth support. It never changes scores, boxes, class membership or the evaluation population. Full saved predictions and evaluation annotations are external inputs. `analysis/analysis_common_v1.py` also contains the original physical-replica paired-sequence accumulator interface. That larger replay requires its complete external operands and the model package dependencies; its optional accelerated kernel additionally requires Numba. Neither that kernel nor a full detection bootstrap is run by the compact reproduction command.

## Model and training source

| Component | Source |
|---|---|
| Four-class D-FINE/CURE adapter and loss | `code/cure_fusion/dfine_cure_v2.py`, `dfine_cure_losses_v2.py` |
| Frame action utility and fixed-input rules | `csu_action_v1.py`, `csu_action_rules_v1.py` |
| Rich paired-query representation | `csu_action_queryset_v2.py`, `csu_queryset_inputs_v2.py` |
| Same-representation and Task-feature utility readouts | `csu_sensor_head_control_v1.py`, `csu_task_feature_comparison_v2.py` |
| Native RT-DETR fitting protocol | `rtdetr_native_admission_v1.py`, `rtdetr_native_overfit_v1.py` |

Paths after the first row are relative to `code/cure_fusion/`. Required local imports are included, including earlier helpers used by the final implementations. `SOURCE_MAP.json` records unchanged source hashes and narrowly extracted plotting/statistical functions. Public “sensor utility” terminology maps to the original `cev_*` tensor names; these scientific interfaces are preserved.

CUDA launch sources are in `code/scripts/`; scientific settings, split identities and required machine-readable metadata are included in `configs/`. The public reproduction protocols preserve every original scientific field. Their source-integrity inventories, including the richer study's shared-admission source list, cover the included research code and required execution metadata. Guard-only historical inventories are not execution dependencies. No historical experiment is modified by this portability adapter.

Preflight an experiment or inspect its input arguments:

```sh
python run_experiment.py --experiment detector --check-only
python run_experiment.py --experiment detector \
  --python /environments/cuda/bin/python -- --help
```

Other experiment names are `action`, `queryset`, `cure-feature-head`, `task-feature-head`, and `native-fitting`. The wrapper creates an isolated temporary execution layout from this package, supplies the hash-checked public protocol, and forwards explicit input/output arguments. It neither reads the original workspace nor requires private notes. Dataset/cache/checkpoint hashes, actual admission identities, class/split contracts, initialization, budgets, objectives and evaluator safeguards remain unchanged. Supply absolute external input/output paths because the execution directory is temporary. `configs/external_dependencies.json` lists required external artifacts and recorded runtime versions.

The completed primary study uses 6,180 fitting frames and 992 reused development frames, one detector seed and fixed 40-epoch endpoints. Action comparisons use five controller seeds with one detector seed. Trained-readout comparisons are post hoc, use one head seed, and retain each endpoint's own factual query correspondence. These populations and scopes are not changed by packaging. GPU training, native detector extraction, raw target generation and full saved-detection AP replay were not repeated during this release's CPU smoke test.

## Layout

```text
code/       Exact model, loss, input and recorded training sources
analysis/   Saved-output metrics and recording-level statistics
plotting/   Current scientific figures only
configs/    Frozen protocols, splits and external dependencies
results/    Compact numerical operands and four attributed sensor examples
figures/    Eight finished figures, each as PNG/PDF/SVG
tables/     Thirteen publication views and complete raw numeric tables
tests/      Data, arithmetic and local import-closure checks
```

The separation of source, configurations, dependencies and commands follows common official research repositories such as [D-FINE](https://github.com/Peterande/D-FINE) and [CRN](https://github.com/youngskkim/CRN). They remain distinct methods and external dependencies, not results reproduced here.

See `NOTICE.md` for dataset-example attribution and third-party software boundaries.
