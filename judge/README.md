# Judge Skeleton

## Purpose

Runs the scoring pipeline inside a VM: load `load_model()` from the submitted `model.py`, evaluate on the **Judge’s own MNIST test set**, and output JSON (`score` / `pass`). It does not read any self-reported metrics from the submission, to avoid reward hacking.

## Environment

- Python 3.8+
- Dependencies: `pip install -r requirements.txt` (or use project `.venv`)

## Directory Layout

- **Judge code**: `judge/` (this directory)
- **Submission directory**: `judge/workspace/` (corresponds to `/workspace` in DESIGN.md; in a real VM this may be `/workspace`)
- **MNIST data**: Downloaded on first run via torchvision to `judge/data/mnist/` (or `/data/mnist`)

## Minimal Run (Verify Pipeline Only)

`workspace/model.py` is already present and compliant. If `workspace/model.pt` does not exist, `load_model()` returns an untrained model (~10% accuracy), but the **scoring pipeline runs to completion**.

```bash
# From project root with .venv activated
cd judge
python run_judge.py
```

Example output (untrained):

```json
{
  "score": 0.1023,
  "pass": false,
  "error": null
}
```

## Full Run (Train Then Pass)

1. Install dependencies: `pip install -r requirements.txt`
2. Run example training (downloads MNIST and saves weights to `workspace/model.pt`):
   ```bash
   python train_example.py
   ```
3. Run the Judge again:
   ```bash
   python run_judge.py
   ```
   Expected: `"score": 0.92+`, `"pass": true`.

## Output Fields

| Field   | Meaning |
|--------|---------|
| `score` | Continuous score: MNIST test accuracy in [0, 1] |
| `pass`  | Pass if `score >= 0.92` |
| `error` | Error message if load/validate/inference failed; otherwise `null` |

Exit code: `0` = pass, `1` = fail or error.

## Relation to Design Doc

- Anti-cheating: Score is determined only by inference inside `run_judge()`; no submission self-reports are used.
- Misjudgment control: Interface matches DESIGN.md; device is CPU or current GPU; paths follow the `workspace` convention.
