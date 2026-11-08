# Coordination Referee

**English** · [简体中文](README_CN.md)

Given a completed multi-agent run, identify the root coordination failure, decide whether the team achieved its goal, and locate the message where the failure began. This folder contains the retained **v84b** submission: original Python code, two trained ModernBERT checkpoints, tokenizers, a candidate-set model prior, public demo inputs, and a launcher. It can be moved out of this repository and run on its own.

## Official server result

| Score | Macro f1 | Robustness f1 | Success f1 | Fault turn hit2 |
|---:|---:|---:|---:|---:|
| **0.94328** | 0.94406 | 0.94326 | 0.94827 | 0.93194 |

These are the [server values reported by the participant](official_result.json). The public demo has no test labels or hidden robustness mask, so running it produces predictions, not this score.

## The competition

Each CSV row is one synthetic execution trace: the original goal, agent roster, communication topology, ordered messages, shared state, and artifacts. The competition provides 10,000 labeled training runs and 4,000 unlabeled test runs; the public and private test partitions each contain 2,000 runs. A handoff can fail and the team can still recover, so **goal completion is a separate target**. The [organizer's starter kit](https://github.com/dmayudin/coordination-referee/tree/690fbc90f611f79b0d21222b087f33a24705965f) includes a separate demo dataset with the same format.

| Output column | What to predict |
|---|---|
| `label` | One root cause: `clean`, `dropped_handoff`, `duplicated_work`, `deadlock`, `conflict`, `goal_drift`, or `runaway_loop`. |
| `success` | `1` if the original goal was achieved; `0` otherwise. |
| `fault_turn` | The turn at which the root fault first appeared, or `-1` for a clean run. |

The [official scoring rules](https://github.com/dmayudin/coordination-referee/blob/690fbc90f611f79b0d21222b087f33a24705965f/docs/metrics.md) weight seven-class Macro F1 (50%), Macro F1 on a hidden distribution-shift subset (25%), binary Success F1 (15%), and joint fault-turn hit@2 (10%). A hit requires **both** the correct fault class and an onset within two turns of the annotation; the denominator is all truly faulty runs.

## How the solution works

1. **Find evidence before classifying the run.** Parse assignments, deliveries, acknowledgements, repeated actions and conflicting state updates. Generate candidate onset turns for each fault type. LightGBM scores these candidates in five folds grouped by run. The strongest scores, their separation and their positions become run-level features. See [parsing](src/parse.py) and [candidate generation](src/feats.py).
2. **Combine run-level classifiers.** LightGBM and a temporal CNN model structural and sequential evidence. A five-fold CatBoost classifier contributes half of the class probability blend at this stage. The pipeline can add high-confidence pseudo-labels from the unlabeled test batch (threshold 0.90), so changing the test batch can change the predictions. A [fault arbiter](src/arbiter.py) separates competing root causes from secondary symptoms.
3. **Predict success and locate the onset.** A [candidate-set Transformer](src/csetnn.py) sees up to eight candidate turns per fault plus a null option. A zero-initialized periodic numerical embedding adds information about positions and scores without changing the initial token projection. Five models pool success in log-odds space and combine turn logits. The decoder picks the turn whose ±2 window covers the most probability mass.
4. **Use language evidence where it changes the decision.** [Two ModernBERT encoders](src/lm/lm_infer.py) read the ordered trace (up to 8,192 tokens each). Their geometric-mean class probabilities enter `log P_stage2 + 0.2 × log P_LM`. The LM updates the preceding decision only if fusion changes the stage-two argmax; a changed fault class also triggers relocalization. Its Success weight is **0**, and its turn pointers do not drive final localization. [Evidence-based corrections](src/resid_post.py) then adjust selected labels and turns. The complete flow is in [pipeline.py](src/pipeline.py).

## Reproduce the submission

Run commands **from this folder**. For the verified example, use **Python 3.13.9** (Windows x86-64); the historical Python 3.12.10 environment is also documented below. Archive checks and demo preparation use only the standard library. Model execution also needs a CUDA GPU with bfloat16 support, a compatible NVIDIA driver, and the Python packages below. Install dependencies before running offline. Budget roughly **2 GB of disk for the bundle, extracted files and CSVs**, plus a separate Python environment and training scratch space.

### 1. Verify the assets and prepare the environment

```sh
python reproduce.py verify
python reproduce.py prepare
python -m venv .venv
```

Activate the environment: `source .venv/bin/activate` on Linux, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install the reference CUDA build first:

```sh
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environment/requirements-smoke.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
```

Both capability checks should print `True`. The [smoke environment](environment/requirements-smoke.txt) lists the versions used in the GPU example below. To recreate the earlier research environment instead, create a separate **Python 3.12.10** environment, install CUDA PyTorch `2.7.1` from the same cu128 index, then run `python -m pip install -r requirements.txt` using the [historical pins](requirements.txt). Neither environment is the competition container. `prepare` extracts all 32 original archive members to `runtime/v84b/`; the [src/](src/) copy is for browsing. Run the launcher against the extracted original to use the weights and tokenizers.

### 2. Reproduce the verified GPU example (560 train / 35 test)

```sh
python examples/make_smoke.py --out work/smoke
python reproduce.py run --train work/smoke/train.csv --test work/smoke/test.csv --output work/smoke/predictions.csv --jobs 4 --lm-budget 8192
python tools/score_local.py --truth work/smoke/truth.csv --pred work/smoke/predictions.csv
```

This constructs 560 training rows and 35 separate inference rows. Use the default `supplied-only` profile: the bundled demo file also contains these held-out rows. Pass `truth.csv` only to the scorer. It reports Macro F1, Success F1, joint hit@2 and **local K** (`0.50 × Macro F1 + 0.15 × Success F1 + 0.10 × joint hit@2`, maximum 0.75). It does not estimate Robustness F1. If fewer than 20 pseudo-labels clear the confidence threshold, the original algorithm ends after the first pass. The launcher detects the missing LM-fusion stage and declines to publish that output as a complete v84b run.

The [recorded GPU run](examples/reference/run.json) processed both LM members, accepted 22 pseudo-labels, trained the five-fold CatBoost and five CSET members, and produced [35 reference predictions](examples/reference/gpu_predictions.csv). GPU kernels, library versions and batch settings may affect exact predictions on another system. The full demo inputs below were verified; the 4,000-row inference was not rerun for this delivery.

**Without a GPU**, you can still reproduce the input split and the [reference local metrics](examples/reference/local_metrics.json) using only the standard library:

```sh
python examples/make_smoke.py --out work/smoke
python tools/score_local.py --truth work/smoke/truth.csv --pred examples/reference/gpu_predictions.csv
```

### 3. Predict on all 4,000 public demo inputs

```sh
python tools/prepare_demo.py
python reproduce.py run --train work/demo/train.csv --test work/demo/test.csv --output work/demo/predictions.csv --jobs 4
python reproduce.py check --test work/demo/test.csv --output work/demo/predictions.csv
```

No data download is needed. The preparation script restores the organizer's 10,000 training rows from `extra_train.csv.gz` inside the ZIP and 4,000 unlabeled rows from `assets/demo-test.csv.gz`, checking both CSV hashes. The default `supplied-only` profile fits the non-LM components from the supplied training CSV, disables the optional demo-data merge (`CR_NO_EXTRA=1`), and trains CSET (`CR_CSET_MODE=train`). The two LM checkpoints are loaded for inference. The output must be a new file with one row per test ID and columns `run_id,label,success,fault_turn`.

On a GPU with limited memory, append `--lm-budget 8192` to `run`: the two language models run first with a smaller batch budget, then the other models run. Without this option, the original launcher starts the LM concurrently with a 65,536-token batch budget. Batch size may affect numerical rounding.

### 4. Use the competition platform files

If you have the [competition platform files](https://dsworks.ru/en/champ/aij26-referee/data), pass their paths in place of the demo paths. For the **unmodified archive's data policy**, run:

```sh
python reproduce.py run --profile archive-default --train platform/train.csv --test platform/test.csv --output work/platform_predictions.csv --jobs 4
```

`archive-default` enables the archive's optional merge of nonduplicate public demo rows and its automatic CSET mode. Omit `--profile archive-default` to train with only the supplied CSV. The server job configuration and hidden labels are not included; the commands reproduce the implementation, not a guaranteed leaderboard number. The original recipe and training membership for the LM checkpoints are unavailable, so their training from scratch and a leakage-free local estimate of their quality cannot be reproduced here.

## Files to keep together

| File or directory | Contents |
|---|---|
| [Original submission](assets/submission_v84b_v81_lm2.zip) | Exact v84b archive, including `solution.py`, CSET prior, two ModernBERT checkpoints, tokenizers, and compressed demo training data. SHA-256: `918ae44f7fbb01756594c165e19dd8969e7d599ec3dfbe1dea53952129bbd8c4`. |
| [Demo test data](assets/demo-test.csv.gz) | Unchanged public demo test set from the [pinned organizer data release](https://github.com/dmayudin/coordination-referee/tree/690fbc90f611f79b0d21222b087f33a24705965f/data); contains no targets. |
| [Launcher](reproduce.py) · [dependencies](requirements.txt) | Integrity check, extraction, execution and output validation. |
| [Data preparation](tools/prepare_demo.py) · [smoke split](examples/make_smoke.py) · [local scoring](tools/score_local.py) | Offline tools; no dependency on the parent workspace. |

For sharing on GitHub, keep the original ZIP intact and deliver it via a Release asset or Git LFS. A Release download belongs at `assets/submission_v84b_v81_lm2.zip`. If a checkout contains an LFS pointer instead of the archive, `verify` will reject it.
