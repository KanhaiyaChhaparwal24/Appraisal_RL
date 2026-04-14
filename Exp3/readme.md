# **Experiment 3 Overview**

This folder contains the code for Experiment 3, where an agent learns via
reinforcement learning in several emotion-eliciting scenarios (anxiety,
despair, irritation, rage). Originally, appraisal was computed by
hand-crafted, rule-based functions; this fork extends the setup with a
**modular appraisal layer**, a **neural appraisal model**, and
**emotion-based reward shaping**.

---

## 1. Original Pipeline (Replication)

The original sequence remains available and can be fully reproduced.

### 1.1 Determining the Classifier

The following steps analyze human data, generate training/testing data, and
determine SVM classifiers over appraisal features.

**(a) Analysis of Human Data**  
Run `01_classifier/01_analyze_human_data.py` to analyze human data and obtain
the mean and variance of human emotion precision.

Human data:

- `data/human_free_limit.csv`

**(b) Data Generation**  
Run `01_classifier/02_classifier.py` to generate training and testing data
using Scherer's table. Outputs:

- `data/classifier_test.csv`
- `data/classifier_train.csv`

**(c) Classifier Determination**  
Run `01_classifier/03_determine_classifier_c.py` to determine optimal C values
for free and limit classifiers (matching human behavior):

- Human Free (loose classifier): c = 0.0032, variance = 0.0002
- Human Limit (restricted classifier): c = 0.014, variance = 0.0056

### 1.2 MDP Model Execution

Run `02_mdp_model/01_get_model_data.py` to execute the MDP models and obtain
appraisal vectors for each emotion. Results:

- `data/model_result.csv`

### 1.3 Model Emotion Prediction

Run `03_model_infer/01_svm_infer.py` to predict emotions for model data using
the classifiers from 1.1 and model data from 1.2. Outputs:

- `data/svm_free_0.0032_var.csv`
- `data/svm_limit_0.014_var.csv`

### 1.4 Statistical Analysis

Generate plots for Figures 6 and 7 by running:

- `04_statistical_analysis/Exp3_analyse.R`

---

## 2. Extended Pipeline: Modular / Neural Appraisal + Emotion-Based Reward

This fork extends the original Experiment 3 with a modular appraisal layer, a
learnable appraisal model, and a reward function that depends on emotion via
reward shaping.

### 2.1 Appraisal Models

Appraisal logic is encapsulated in
[02_mdp_model/appraisal_model.py](02_mdp_model/appraisal_model.py):

- `RuleBasedAppraisalModel` wraps the original appraisal functions from the
  agent (suddenness, goal relevance, conduciveness, power) and exposes a
  single `compute(...)` method that returns a 4D vector
  `[suddenness, goal_relevance, conduciveness, power]`.
- `NeuralAppraisalModel` uses the PyTorch network defined in
  [../models/neural_appraisal.py](../models/neural_appraisal.py). Instead of
  raw state, it consumes **learning-aware features** such as TD-error,
  immediate reward, max \(Q(s', \cdot)\), and \(Q(s,a)\), then outputs the
  same 4D appraisal vector (sigmoid-bounded to [0, 1]).

Both models share the same interface so the agent can seamlessly switch
between rule-based and neural appraisal.

### 2.2 Agent and Appraisal Interface

- File: [02_mdp_model/agent.py](02_mdp_model/agent.py)

The `agent` class now uses the appraisal models above and exposes a unified
`compute_emotion(...)` method that:

- Calls `RuleBasedAppraisalModel` by default.
- Optionally calls `NeuralAppraisalModel` when neural appraisal is enabled.
- Updates internal fields
  `self.sud_app`, `self.goal_app`, `self.cdc_app`, `self.power_app`.

This isolates appraisal logic from the rest of the RL implementation.

### 2.3 Emotion-Based Reward Shaping

In the extended agent, reward can depend on emotion via **shaping**:

- Base reward \(r\) is computed by each MDP (e.g.
  [02_mdp_model/anxiety.py](02_mdp_model/anxiety.py),
  [02_mdp_model/despair.py](02_mdp_model/despair.py),
  [02_mdp_model/irritation.py](02_mdp_model/irritation.py),
  [02_mdp_model/rage.py](02_mdp_model/rage.py)) via `mdp.calculate_reward()`.
- Appraisal produces a 4D vector
  \( e = [suddenness, goal, conduciveness, power] \).
- A scalar emotion term \(f(e)\) is computed (linear combination of these
  components), and the shaped reward is:

$$
r' = r + \lambda f(e)
$$

where \(\lambda\) is a tunable scalar. This keeps the original reward
signal while allowing emotion to influence learning.

Shaping and appraisal mode are controlled by environment variables (read in
`agent.py`):

- `USE_NEURAL_APPRAISAL` – `0` for rule-based appraisal, `1` for neural.
- `USE_EMOTION_REWARD` – `0` for plain reward, `1` to enable shaping.
- `EMOTION_REWARD_LAMBDA` – scalar \(\lambda\) multiplying the emotion term.

This enables comparisons between:

1. Original state-based reward (baseline).
2. Emotion-shaped reward using rule-based appraisal.
3. Emotion-shaped reward using neural appraisal.

### 2.4 Logging and Debugging

When `LOG_STEPS=1`, the agent writes detailed per-step logs to
[logs](logs):

- States, actions, rewards, TD-errors.
- Q-values (per state-action) serialized to JSON.
- Appraisal components `[suddenness, goal_relevance, conduciveness, power]`.

This makes it easier to track how changes in appraisal and reward shaping
impact learning dynamics.

---

## 3. How to Run Baseline vs. Extended Variants

All commands below assume you start from the [Exp3](.) directory.

### 3.1 Recommended: Automated Experiments

To run all four scenarios (anxiety, despair, irritation, rage) plus SVM
inference for both **baseline** and **emotion-shaped** modes, and save
results into timestamped folders under [results](results):

```bash
cd Exp3
python run_experiments.py
```

This will:

- Run each scenario once in **baseline** mode
  (`USE_NEURAL_APPRAISAL=0`, `USE_EMOTION_REWARD=0`).
- Run each scenario once in **emotion-shaped** mode
  (`USE_NEURAL_APPRAISAL=0`, `USE_EMOTION_REWARD=1`, with a default
  `EMOTION_REWARD_LAMBDA`).
- Call [03_model_infer/01_svm_infer.py](03_model_infer/01_svm_infer.py) in
  each mode.
- Save snapshots of `model_result.csv` and all `svm_*.csv` files into
  [results](results) with mode- and time-stamped subfolders.

### 3.2 Manual Runs: Baseline vs Emotion-Shaped

You can also run individual scenarios while explicitly setting environment
variables. Conceptually:

- **Baseline (rule-based appraisal, no shaping):**
  - `USE_NEURAL_APPRAISAL=0`, `USE_EMOTION_REWARD=0`.
- **Emotion-shaped (rule-based appraisal):**
  - `USE_NEURAL_APPRAISAL=0`, `USE_EMOTION_REWARD=1`,
    `EMOTION_REWARD_LAMBDA` set to your chosen value (e.g. `0.5`).
- **Emotion-shaped (neural appraisal):**
  - `USE_NEURAL_APPRAISAL=1`, `USE_EMOTION_REWARD=1`,
    `EMOTION_REWARD_LAMBDA` set as above.

On Windows PowerShell, for example, you can do:

```powershell
cd Exp3
$env:USE_NEURAL_APPRAISAL = "0"
$env:USE_EMOTION_REWARD   = "0"
$env:LOG_STEPS            = "0"   # or "1" for detailed logs
python 02_mdp_model/anxiety.py
```

Adjust the environment variables and script name to explore different
configurations and scenarios.

---

## 4. Future Directions

Planned and possible extensions include:

-- **Richer inputs to neural appraisal (in progress)**  
 The current neural appraisal already uses learning-aware features
(including TD-error and Q-values). Further extensions could add more
temporal information or history.

-- **Temporal / sequence modeling**  
 Extend `NeuralAppraisal` with an LSTM or GRU to handle sequences of states
and appraisals: \(emotion*t = f(state_t, hidden_state*{t-1})\).

- **Joint RL + appraisal training**  
  Instead of pure imitation of the rule-based appraisals, combine RL
  objectives with a small regularization term encouraging consistency with the
  original model.

- **Systematic comparisons**  
  Quantitatively compare behavior under:
  - Rule-based vs neural appraisal.
  - State-based vs emotion-based reward.
  - Different emotion→reward mappings.

These directions aim to turn the neural appraisal module from a function
approximator of existing rules into a component that **learns useful emotional
signals that shape RL behavior**.
