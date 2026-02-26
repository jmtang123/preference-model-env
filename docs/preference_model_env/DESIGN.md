# RL Environment Design: MNIST Classifier Train & Export

This document specifies an RL environment for LLM training: Prompt, Judge, Tools, Data, anti-cheating (no false positives), and misjudgment control (no false negatives).

---

## 1. Environment Overview

| Component | Description |
|-----------|-------------|
| **Name** | MNIST Classifier Train & Export |
| **Goal** | The LLM must train an MNIST classifier inside the VM and export it via a fixed interface; the Judge loads and evaluates it independently. |
| **Difficulty** | Requires data loading, model definition, training loop, checkpoint saving, and a compliant `load_model()`; non-trivial for typical LLMs. |

---

## 2. Prompt (Task Description)

Below is the **unambiguous** task text given to the LLM. It can be used as-is in the environment.

```text
Your task is to train a machine learning model that classifies MNIST digits with high accuracy, then export it so the judge can evaluate it.

## Constraints

1. **Data**: The MNIST dataset is available in the environment. You may use any standard source (e.g. torchvision.datasets.MNIST with root="/data/mnist" or the path announced at runtime). Use the official train/test split; do not use the test set for training.

2. **Output requirements**: When done, you must create exactly the following:
   - A file at path: `/workspace/model.py`
   - Inside `model.py`, a function with signature:
     ```python
     def load_model() -> torch.nn.Module
     ```
   - `load_model()` must take no arguments and return a single `torch.nn.Module` that:
     - Accepts input of shape `(N, 1, 28, 28)` and dtype float32 in range [0, 1].
     - Outputs logits or class scores of shape `(N, 10)` for classes 0–9.
   - Optional: a weights file (e.g. `/workspace/model.pt` or `/workspace/checkpoint.pt`) that `load_model()` may read from. The judge will only run `load_model()` and evaluate the returned module on the **judge's own test data**.

3. **No external servers**: All work must be done locally in the VM. Do not call external APIs or read from the internet for the model or labels.

4. **Reproducibility**: The judge may run `load_model()` multiple times; it must return a valid model each time (e.g. by loading from a saved checkpoint).

## Success criterion

The judge will load your model with `load_model()`, run it on a held-out MNIST test set, and compute classification accuracy. Your submission **passes** if accuracy is at least 92% (0.92). The score you receive is the accuracy value in [0, 1] (continuous), or a binary pass/fail derived from this threshold.
```

Design notes:
- **Unambiguous**: File path, function signature, input/output shapes and dtypes, and pass threshold (92%) are all specified.
- **Verifiable**: The Judge does not trust any self-reported metrics; it computes accuracy only by loading and running the returned module.

---

## 3. Judge (Scorer)

### 3.1 Responsibilities

- Load the submitted `model.py` in an isolated context and call `load_model()`.
- Run inference on the **Judge’s own MNIST test set** (or public data with fixed seed) and compute accuracy.
- Output a continuous score (accuracy ∈ [0, 1]) and/or a binary result (pass/fail, e.g. pass if accuracy ≥ 0.92).

### 3.2 Scoring Flow (Pseudocode)

```text
1. Check that /workspace/model.py exists.
2. Import the model in a clean interpreter (or subprocess).
3. Call model.load_model() to get raw_module.
4. Verify raw_module is torch.nn.Module and that forward((1,1,28,28)) yields (1,10).
5. Load the Judge’s MNIST test set (fixed seed; independent of the submission).
6. Run inference batch-wise, compare argmax to labels, compute accuracy.
7. Output score = accuracy; optionally binary_pass = (accuracy >= 0.92).
```

### 3.3 Output Format

- **Continuous**: `{"score": 0.95, "accuracy": 0.95}` or a single `score`.
- **Binary**: `{"pass": true, "score": 1}` / `{"pass": false, "score": 0}`, or both alongside the continuous score.

The Judge does not read any self-reported accuracy from the submission; the only source of truth is its own inference result.

---

## 4. Tools (Available Actions)

The following capabilities are available to the LLM inside the VM (provided by the environment/sandbox):

| Tool | Description | Example |
|------|-------------|---------|
| **Shell / CLI** | Run commands (training scripts, pip install, etc.) | `python train.py` |
| **File read** | Read files in the VM (code, config, data path docs) | Read `/data/README.txt` |
| **File write** | Create or overwrite code and weight files | Write `/workspace/model.py`, `/workspace/model.pt` |
| **Network (whitelist)** | Only pip/index for dependencies; no access to model hosting or answer sources | Enforced by sandbox policy |

Tools such as “call external inference API” or “upload to external service” are not required; all training and export happen inside the VM.

---

## 5. Data

| Purpose | Content | Location / Method |
|---------|---------|-------------------|
| **Training** | MNIST train set (or subset) | Pre-installed or downloaded at first run to `/data/mnist`; readable by the LLM |
| **Judge** | MNIST test set (fixed split) | Held only by the Judge: same test data, fixed seed, reproducible |
| **Docs** | Data path and format | Optional: `/data/README.txt` describing root and train/test usage |

Important: The Judge **must** use its own test set and its own loading logic; it must not trust test data or preprocessing provided by the submission, to avoid leakage or tampering that could inflate scores.

---

## 6. Anti-Cheating (No False Positives / Reward Hacking)

Goal: **Never mark a submission as pass when it has not actually met the bar.**

| Risk | Mitigation |
|------|------------|
| Submission hardcodes “accuracy=0.99” or similar | Judge ignores all self-reported metrics; it computes accuracy only via `load_model()` and its own inference. |
| Submission reads or depends on Judge’s test path/files | Test set exists only inside the Judge process or a Judge-only directory; submission code cannot access it; use subprocess to run `model.py` if needed. |
| Submission uses random or constant outputs | Judge evaluates on a fixed test set; random output gives ~10% accuracy, well below 92%. |
| Submission pulls pretrained weights or answers from the network | Environment restricts network: only pip/index allowed; no access to model hosting or answer sources; Judge may optionally check weights against known hashes. |
| Submission tampers with or hijacks Judge script | Judge and submission directory are separate (submission directory read-only for Judge); Judge code is not under `/workspace` and runs with elevated or separate user. |
| Interface cheating (e.g. returned “model” hardcodes labels) | Judge uses **its own** test set and labels for inference; hardcoding train labels does not generalize to an unseen test set. |

In short: **The only source of truth is the accuracy obtained by the Judge running forward pass on the module returned by `load_model()` using the Judge’s own test data.**

---

## 7. Misjudgment Control (No False Negatives / Reward Denial)

Goal: **Any submission that truly reaches 92% accuracy should receive pass or the corresponding high score.**

| Risk | Mitigation |
|------|------------|
| Interface ambiguity causes valid implementations to fail | Prompt defines `load_model()` signature and I/O shapes/dtype; Judge does light validation (type, shape); document an example implementation. |
| Device mismatch (CPU/GPU) causes runtime failure | Judge runs on a fixed device (e.g. CPU), or Prompt states “model must run on CPU”; Judge moves model and data to CPU before evaluation. |
| Path or env differences cause `load_model()` to miss weights | Prompt fixes weight path (e.g. `/workspace/model.pt`); Judge sets working directory to `/workspace` (or as in Prompt) to avoid relative-path ambiguity. |
| Test split or preprocessing differs from submission’s expectation | Judge uses standard MNIST test set (e.g. torchvision default), aligned with common tutorials; Prompt states “Judge uses its own test set” so the submission need not guess. |
| Threshold too strict (e.g. 99%) rejects reasonable implementations | Set pass threshold to 92% to allow for data variance and seed differences. |
| Dependency or version mismatch causes import failure | Environment provides fixed Python and deps (e.g. requirements.txt); Judge runs in the same environment or in an isolated env with the same deps. |

Optional: Provide a reference implementation and expected accuracy range in the docs for Judge regression tests, so that compliant submissions consistently pass.

---

## 8. Difficulty and Challenge

- The LLM must: load data, choose a model, implement a training loop, save a checkpoint, and implement a compliant `load_model()`.
- Submissions that only produce pseudocode or an untrained model get low scores; those that correctly train and export can reliably reach 92%+ and pass.
- Difficulty can be increased by raising the threshold (e.g. 95%), limiting training time or compute, or requiring a specific architecture.

---

## 9. Mapping to Assessment Criteria

| Criterion | This design |
|-----------|-------------|
| Prompt is something an AI/ML engineer might do | Training and exporting an MNIST classifier is a standard ML engineering task. |
| Judge gives high score when successful, low when not | Score is the actual inference accuracy; pass if ≥92%. |
| No false positives | Judge evaluates independently, does not trust self-reports, uses fixed test set and network/privilege controls. |
| No false negatives | Clear interface and paths, fixed environment and device, reasonable threshold and documentation. |
| Binary or continuous score | Both: continuous (accuracy) and binary (pass/fail). |
| Environment is challenging | Full training and interface implementation required; non-trivial. |
| Prompt is unambiguous | Paths, signature, shapes, threshold, and success criterion are all written down. |

---

## 10. Document and Changes

- Document version: 1.0.
- If threshold, paths, or interface change, update the Prompt and Judge implementation and record the reason here.
