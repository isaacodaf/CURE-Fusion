# Setup and reproduction

Start with the CPU quick start below. It reproduces calculations from the saved scientific operands. Training or reevaluating full detections requires separate datasets, checkpoints and other inputs; those files are not bundled.

## Choose a task

| Task | Requirements | Output |
| --- | --- | --- |
| Browse results | A browser | Eight figures and thirteen tables already in this repository |
| Recompute utility statistics and tables | Python 3.12 and `requirements.txt`; CPU | Checked JSON/CSV results and scientific tables |
| Regenerate all eight figures | Same packages, plus Arial | PNG, PDF and SVG outputs, checked against the saved exports |
| Evaluate your saved detections | Evaluation dependencies and complete COCO annotations/predictions | COCO box AP metrics |
| Rerun a recorded experiment | A suitable CUDA environment and the exact external inputs | New training/evaluation outputs under the selected frozen protocol |

## 1. Clone and install

On macOS or Linux:

```sh
git clone https://github.com/isaacodaf/CURE-Fusion.git
cd CURE-Fusion
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, use these environment commands after cloning:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the remaining commands with `.venv\Scripts\python.exe` in place of `python` if you have not activated that environment. Use the pinned package versions in [requirements.txt](requirements.txt). GPU experiment runtimes are separate from this CPU analysis environment.

## 2. Check the installation

```sh
python -m unittest discover -s tests -v
```

The tests check the release manifest, numerical helpers, local source dependencies, and figure/table assets. They do not train a detector.

## 3. Recompute saved results

```sh
python reproduce_results.py --skip-figures --output outputs/quickstart
```

Expected result: `outputs/quickstart/REPRODUCTION.json` reports `passed_saved_research_reproduction`. The `tables/` subdirectory contains regenerated numeric tables and a `publication/` subdirectory with the thirteen formatted views.

This command recomputes 40 utility assessment cells, 12 constant comparisons, eight trained-readout comparisons and the recorded 2,000 paired recording draws. It also checks the complete saved detection and action summaries. Utility error and detection AP are separate quantities: the command recomputes utility statistics, but reads detection AP from recorded results. It does not rerun detector inference or training.

Use a new output directory each time, such as `outputs/quickstart_2`. Existing outputs are deliberately preserved.

To export only the scientific tables:

```sh
python export_tables.py --output outputs/publication_tables
```

## 4. Regenerate figures

```sh
python reproduce_results.py --output outputs/full_reproduction
```

This adds all eight figures to `outputs/full_reproduction/figures/` and compares their bytes with the reference exports. Exact figure reproduction requires the pinned plotting packages, Arial for Figure 1, and Matplotlib's DejaVu Sans fonts for the other figures. Arial is not redistributed. Check whether Matplotlib can find it:

```sh
python -c "from matplotlib import font_manager; print(font_manager.findfont('Arial', fallback_to_default=False))"
```

If Arial is unavailable, install a licensed copy through your normal system font installer and restart Python. A different font or rendering environment may produce different figure bytes even when the data are unchanged. The original reviewed figures remain in [figures/](figures/README.md); use the numerical quick start if you only need the scientific calculations. The reproduction command itself makes no network requests.

## 5. Evaluate saved detections

```sh
python -m pip install -r requirements-evaluation.txt
python evaluate_saved.py --help
```

Supply complete COCO-format ground truth, predictions and their SHA-256 digests:

```sh
python evaluate_saved.py \
  --ground-truth /absolute/path/ground_truth_coco.json \
  --ground-truth-sha256 REPLACE_WITH_FILE_SHA256 \
  --predictions /absolute/path/predictions_coco.json \
  --predictions-sha256 REPLACE_WITH_FILE_SHA256 \
  --output outputs/coco_metrics.json
```

The paths and digests above are placeholders to replace. This evaluates all supplied frames under the fixed four-category contract using pycocotools. Full predictions and evaluation annotations are external inputs. Refer to [README.md](README.md) for the evaluation scope and unsupported-class behavior.

## 6. Prepare a CUDA experiment

Inspect the included source/protocol package without starting training:

```sh
python run_experiment.py --experiment detector --check-only
python run_experiment.py --experiment action --check-only
python run_experiment.py --experiment queryset --check-only
python run_experiment.py --experiment cure-feature-head --check-only
python run_experiment.py --experiment task-feature-head --check-only
python run_experiment.py --experiment native-fitting --check-only
```

A successful preflight checks the packaged source and execution layout. It does **not** certify that external data, checkpoints or a GPU are available.

[configs/external_dependencies.json](configs/external_dependencies.json) records checkpoint hashes, upstream revisions, feature/cache identities and the separate recorded runtimes. [configs/execution/index.json](configs/execution/index.json) maps experiment names to launchers and required metadata. The later controls and native RT-DETR fitting study used different Python/PyTorch environments; do not install them into one environment by guessing compatible versions.

Inspect a launcher's complete arguments with a configured CUDA interpreter:

```sh
python run_experiment.py --experiment detector \
  --python /absolute/path/to/cuda-environment/bin/python -- --help
```

Then supply the required external inputs and an explicit output directory. All input/output paths must be absolute because the wrapper builds a temporary execution directory. The example interpreter path is a placeholder. The wrapper preserves source-integrity, split, class, checkpoint, budget and evaluator checks; it does not download or manufacture the missing inputs. Full training is not a one-command reproduction from this compact release alone.

## Troubleshooting

| Message or symptom | What to do |
| --- | --- |
| `python3.12` not found | Install Python 3.12, then recreate the virtual environment. |
| Import error | Confirm that the active interpreter belongs to `.venv` and install `requirements.txt` with that interpreter. |
| Output directory already exists | Choose a different output path; keep the earlier results. |
| Manifest mismatch | Use an unmodified fresh clone. The manifest hashes describe the published files; intentional edits require a separately recorded new manifest. |
| Figure hash mismatch | Check the pinned packages and required fonts. Numerical reproduction can run separately with `--skip-figures`. |
| Missing CUDA/data/checkpoint input | Follow the selected experiment's recorded dependency contract. The CPU quick start does not require these inputs. |

## Data and software terms

See [NOTICE.md](NOTICE.md) for SEW example attribution, upstream dependencies and project-code licensing status. The four included scene examples are adapted under CC-BY-SA 4.0. Dataset and pretrained-model rights remain with their respective providers.
