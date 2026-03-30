# Appraisal_RL

This repository contains code for modeling cognitive–affective processes with
appraisal and reinforcement learning. It now supports both the original
rule-based appraisal mechanism and an extended **neural appraisal +
emotion-based reward** variant.

For experiment-specific details, see the `readme.md` files in each
experiment folder, e.g. `Exp3/readme.md`.

## Current Extensions (This Fork)

This fork adds a learnable appraisal module and connects emotion to reward in
the RL loop for **Experiment 3**:

- New neural appraisal model in `models/neural_appraisal.py` (PyTorch).
- Extended Q-learning agent in `Exp3/02_mdp_model/agent.py` with:
  - Optional **neural appraisal** (vs. rule-based).
  - Optional **emotion-based reward shaping**.
  - Episode-level logging of emotion and cumulative reward.

You can run the original pipeline (baseline) or the extended neural version
by toggling flags in `Exp3/02_mdp_model/agent.py`.
