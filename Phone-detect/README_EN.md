# PhoneGuard-P2: Mobile Phone Detection for Exam Monitoring

[中文](README.md) | **English**

PhoneGuard-P2 is a single-class mobile phone detection project for exam-room images and video frames. Built on a YOLOv8 baseline, it provides an end-to-end engineering workflow for safe preprocessing, training, evaluation, recovery, and submission inference, with an emphasis on small objects, difficult scenes, reproducibility, and strict data boundaries.

> The repository is intended to provide an auditable and reproducible research and engineering implementation. It does not promise any leaderboard score.

## Goals

- Locate phones accurately and return bounding boxes with confidence scores.
- Improve detection under small-object, occlusion, low-light, blur, compression, multi-scale, and multi-angle conditions.
- Reduce false positives from visually similar objects such as remote controls, earbud cases, and power banks.
- Balance precision, recall, and inference speed for near-real-time monitoring.
- Keep training, evaluation, checkpoint selection, and recovery traceable and verifiable.

Local evaluation follows the competition score:

```text
score = 100 × (0.5 × mAP@0.5 + 0.3 × Precision + 0.2 × Recall)
```

## Approach

PhoneGuard-P2 adds a P2 small-object head to the usual P3/P4/P5 detection scales. The system contains two training entry points and two engineering control mechanisms:

- **Baseline**: a straightforward YOLOv8 reference workflow; its `--pretrained` mode may download weights, so strict offline runs must explicitly select `--scratch`.
- **PhoneGuard-P2**: four-scale P2/P3/P4/P5 detection, custom augmentations, and controlled experiment orchestration.
- **Robust → Refine curriculum**: the first stage targets low light, blur, compression, limited occlusion, and scale variation; the second disables strong Mosaic/MixUp, lowers the learning rate, and uses a quality gate to reject regressions.
- **Reproducibility and recovery**: dataset fingerprints, checkpoint bindings, append-only evidence, and authenticated resume entry points protect experiment identity.

The two-stage gate reduces the risk of regressions and overfitting, but it cannot guarantee that overfitting never occurs or that a platform score will improve.

## Workflow and data boundary

```text
Local training archive ──→ safe preprocessing and train/validation split ──→ training and internal validation
                                                                                   ↓
Local test archive ──→ safe preprocessing into isolated processed/test ──→ final prediction ──→ submission JSON
```

The preprocessor may validate and prepare an isolated test set locally, but training, validation, threshold tuning, and checkpoint selection must never consume test samples; only the final prediction module may read `processed/test`. The formal schema-v2 training identity depends only on train, validation, and the training archive; it contains no test paths, names, sizes, hashes, or member lists.

Official data, private test sets, generated weights, run outputs, credentials, and submissions must remain local and must not be committed or redistributed. Treat the original local materials supplied by the organizer as the authority for competition and data constraints.

## Repository layout

| Path | Responsibility |
| --- | --- |
| `project/processdata/` | Safe extraction, validation, deterministic splitting, atomic replacement, and train-only views |
| `project/baseline/` | YOLOv8 baseline training, validation, and submission inference |
| `project/experiments/phoneguard_p2/` | Four-scale model, augmentations, single-stage and curriculum training, evaluation, selection, and recovery |
| `tests/` | Synthetic, unit, and security regression tests |
| `phoneguard-p2-size-aware-cancellation-20260828/` | Standalone train-only size-aware sampling and cooperative-cancellation hand-off |
| `requirements.txt` | Main project Python and CUDA dependencies |

## Environment setup

Windows PowerShell is recommended. Real GPU training requires a compatible NVIDIA GPU; `requirements.txt` currently selects the PyTorch CUDA 12.8 build.

```powershell
git clone https://github.com/swj-stone/phone-detect.git
Set-Location phone-detect

$RepoRoot = (Get-Location).Path
$EnvRoot = "${RepoRoot}-venv"
python --version  # Confirm that this reports Python 3.13.x
python -m venv $EnvRoot
& "$EnvRoot\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependency installation depends on Python, CUDA, the GPU driver, and wheel availability. Confirm that `torch.cuda.is_available()` returns `True` before starting a long GPU run.

## Quick verification

Run the main project test suite from the repository root:

```powershell
python -u -m unittest discover -s tests -p "test_*.py" -v
```

On a code-only checkout, tests that require local acceptance metadata are skipped by design. You can then run the tiny smoke test without formal data or a GPU:

```powershell
Set-Location project
python -u -m experiments.phoneguard_p2.smoke --tiny --device cpu
```

The tiny smoke test verifies code paths and artifact contracts only; it does not measure real model quality.

## Main entry points

Use the following commands to inspect supported arguments. The preprocessor and curriculum runner provide `--dry-run`; inspect `--help` for the other entry points instead of assuming that they support a read-only preflight:

```powershell
Set-Location project

python -u -m processdata.preprocess --help
python -u -m baseline.train --help
python -u -m baseline.evaluate --help
python -u -m baseline.predict --help

python -u -m experiments.phoneguard_p2.train --help
python -u -m experiments.phoneguard_p2.curriculum --help
python -u -m experiments.phoneguard_p2.evaluate --help
python -u -m experiments.phoneguard_p2.smoke --help
```

After an interrupted curriculum run, resume from the immutable manifest at the pipeline root:

```powershell
python -u -m experiments.phoneguard_p2.curriculum `
  --resume-pipeline "<results-root>/<pipeline-id>/pipeline_manifest.json"
```

Resume rechecks the dataset fingerprint, model identity, event chain, and checkpoint bytes and does not overwrite an existing execution.

## Outputs and reproducibility

Local run artifacts are stored under `project/results/` by default. A formal PhoneGuard-P2 pipeline normally contains:

- an immutable pipeline manifest and append-only event records;
- robust/refine checkpoints, bindings, and separate executions;
- local validation metrics, checkpoint rankings, and quality-gate decisions;
- the selected weights and their decision evidence.

These directories may contain private-data derivatives or large files and must remain untracked. Formal PhoneGuard-P2 workflows accept only trusted local resources and verified `.pt` files and do not auto-download; PyTorch checkpoints may contain pickle payloads. Baseline `--pretrained` is a separate entry point and may fetch weights from the network.

## Code navigation

- `project/experiments/phoneguard_p2/`: controlled training, recovery, evaluation, and artifact contracts.
- `project/processdata/`: safe preprocessing and train-view implementation.
- `project/baseline/`: the YOLOv8 reference workflow.
- `phoneguard-p2-size-aware-cancellation-20260828/`: a standalone train-only experiment hand-off.

Historical engineering notes may remain in the repository. Complete a privacy review before sharing or citing any scores, paths, digests, or hashes from them.

## Limitations

- Model quality depends on data quality, split policy, hardware, randomness, and training budget.
- Local validation and leaderboard distributions may differ.
- The quality gate can reject clearly degraded candidates, but it cannot eliminate every overfitting risk.
- This repository does not declare an open-source license; confirm the applicable permissions and competition rules before using the code or data.
