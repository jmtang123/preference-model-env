# Task: RL Environment Design Doc and Judge Skeleton

## Goals

1. Produce a full design document (Prompt / Judge / Tools / Data, with anti-cheating and misjudgment control).
2. Implement a minimal runnable Judge skeleton using public data (MNIST) so the scoring pipeline runs inside a VM.

## Subtasks

- [x] Write DESIGN.md: environment overview, full Prompt, Judge flow and output, Tools, Data, anti-cheating, misjudgment control, difficulty, and criteria mapping.
- [x] Implement judge/run_judge.py: load workspace/model.py, validate interface, evaluate on Judge’s test set, output JSON.
- [x] Provide judge/workspace/model.py and judge/train_example.py for local/VM runs.
- [x] Write judge/README.md.

## Test Commands

Run from project root with `.venv` activated.

**Verify Judge pipeline only (no training):**

```bash
cd judge && python run_judge.py
```

Expected: exit code 1, JSON with `score` (~0.1), `pass`: false.

**Full flow (train then pass):**

```bash
cd judge && python train_example.py && python run_judge.py
```

Expected: exit code 0, JSON with `pass`: true, `score` >= 0.92.

## Doc Layout

```
docs/preference_model_env/
  DESIGN.md   # Design document
  task.md     # This task and test commands
judge/
  run_judge.py
  requirements.txt
  README.md
  train_example.py
  workspace/
    model.py
  data/       # Created on first run (MNIST)
```
