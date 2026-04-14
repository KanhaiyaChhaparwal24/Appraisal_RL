# Appraisal_RL

This repository contains code for modeling cognitive–affective processes with
appraisal and reinforcement learning. It now supports both the original
rule-based appraisal mechanism and an extended **neural appraisal +
emotion-based reward shaping** variant.

For experiment-specific details, see the readme files in each
experiment folder, for example [Exp3/readme.md](Exp3/readme.md).

## Current Extensions (This Fork)

This fork adds a learnable appraisal module and connects emotion to reward in
the RL loop for **Experiment 3**:

- Neural appraisal network in [models/neural_appraisal.py](models/neural_appraisal.py) (PyTorch).
- Modular appraisal layer in [Exp3/02_mdp_model/appraisal_model.py](Exp3/02_mdp_model/appraisal_model.py) with:
  - `RuleBasedAppraisalModel` wrapping the original appraisal formulas.
  - `NeuralAppraisalModel` that takes learning-aware features
    (TD-error, reward, Q-values) as input.
- Extended Q-learning agent in [Exp3/02_mdp_model/agent.py](Exp3/02_mdp_model/agent.py) with:
  - Optional **neural appraisal** (vs. rule-based).
  - Optional **emotion-based reward shaping**:
    \( r' = r + \lambda f(\text{appraisal}) \).
  - Optional per-step logging of TD-errors, Q-values, and appraisal to
    [Exp3/logs](Exp3/logs).

Configuration is controlled via environment variables (no code edits needed
for experiments):

- `USE_NEURAL_APPRAISAL` – `0` for rule-based, `1` for neural.
- `USE_EMOTION_REWARD` – `0` for standard reward, `1` to enable shaping.
- `EMOTION_REWARD_LAMBDA` – scalar \(\lambda\) for reward shaping.
- `LOG_STEPS` – `0` to disable, `1` to log each step.

To run the full Experiment 3 pipeline (all four scenarios + SVM inference)
for both **baseline** and **emotion-shaped** modes and save results into
timestamped folders under [Exp3/results](Exp3/results), use:

```bash
cd Exp3
python run_experiments.py
```

You can still run individual scenarios directly (e.g. anxiety, despair,
irritation, rage) from [Exp3/02_mdp_model](Exp3/02_mdp_model) when you want
fine-grained control and detailed logs.
