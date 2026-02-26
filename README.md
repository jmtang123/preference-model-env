# Preference Model RL Environment: MNIST Classifier

RL environment for LLM training (Preference Model assessment). The agent trains an MNIST digit classifier and exports it via a fixed interface; the judge evaluates on its own test set.

- **Design doc (Prompt, Judge, Tools, Data, anti-cheating):** [docs/preference_model_env/DESIGN.md](docs/preference_model_env/DESIGN.md)
- **Judge (runnable):** [judge/](judge/) — see [judge/README.md](judge/README.md) for how to run.

## Quick run

```bash
cd judge
pip install -r requirements.txt
python run_judge.py          # uses workspace/model.py (untrained → low score)
python train_example.py      # train and save weights
python run_judge.py          # re-run judge → pass if accuracy ≥ 92%
```
