# **Experiment 3 Overview**

This folder contains the code for Experiment 3, where an agent learns via
reinforcement learning in several emotion-eliciting scenarios (anxiety,
despair, irritation, rage). Originally, appraisal was computed by
hand-crafted, rule-based functions; this fork extends the setup with a
**neural appraisal model** and **emotion-based reward**.

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

## 2. Extended Pipeline: Neural Appraisal + Emotion-Based Reward


This fork extends the original Experiment 3 with a learnable appraisal model
and a reward function that depends on emotion.

### 2.1 Neural Appraisal Model

- File: `models/neural_appraisal.py`  
- Class: `NeuralAppraisal(nn.Module)` (PyTorch)
	- Input: state vector (currently a one-hot encoding over permitted MDP
		states).
	- Architecture: `Linear → ReLU → Linear → ReLU → Linear`.
	- Output: 4D emotion/appraisal vector:  
		`[suddenness, goal_relevance, conduciveness, power]`.

The model is integrated into the Q-learning agent but kept optional via
feature flags.

### 2.2 Agent and Appraisal Interface

- File: `02_mdp_model/agent.py`

The `agent` class now exposes a unified appraisal interface:

- Rule-based appraisal methods (original):
	- `appraise_suddenness()`
	- `appraise_goal_relevance()`
	- `appraise_conduciveness()`
	- `appraise_power()`
- Unified accessor: `compute_emotion(...)`  
	Returns a 4D emotion vector and updates:
	- `self.sud_app`
	- `self.goal_app`
	- `self.cdc_app`
	- `self.power_app`

When neural appraisal is enabled, `compute_emotion` can route through the
`NeuralAppraisal` network.

### 2.3 Emotion-Based Reward

In `agent.py`, reward can be driven by emotion as follows:

- Base reward is computed by each MDP (e.g. `anxiety.py`, `despair.py`,
	`irritation.py`, `rage.py`) via `mdp.calculate_reward()`.
- The agent can optionally:
	- Replace this reward with an emotion-based signal, or
	- Mix both into a combined reward.

Configuration flags (in `agent.py`):

```python
USE_NEURAL_APPRAISAL = False      # rule-based vs neural appraisal

USE_EMOTION_REWARD = False        # state-based vs emotion-based reward
EMOTION_REWARD_MODE = "replace"   # "replace" or "mix"
EMOTION_REWARD_BASE_WEIGHT = 0.5  # weight for base reward when mix is used
```

Reward mapping from emotion to scalar (simplified example):

```python
# emotion = [suddenness, goal_relevance, conduciveness, power]
reward = 1.0 * goal_relevance \
			 + 1.0 * conduciveness \
			 - 0.5 * suddenness \
			 + 0.5 * power
```

This connects the appraisal signal directly to the RL update, enabling
comparisons between:

1. Original state-based reward (baseline).
2. Emotion-based reward using rule-based appraisal.
3. Emotion-based reward using neural appraisal.

### 2.4 Logging and Debugging

The agent logs:

- Q-values and TD-errors at key points.
- Episode-level cumulative reward.
- Final or current emotion vector `[suddenness, goal_relevance,
	conduciveness, power]`.

This makes it easier to track how changes in appraisal impact learning.

---

## 3. How to Run Baseline vs. Neural Variants

All commands below are from the `Exp3` directory.

### 3.1 Baseline: Rule-Based Appraisal + State-Based Reward

In `02_mdp_model/agent.py`:

```python
USE_NEURAL_APPRAISAL = False
USE_EMOTION_REWARD = False
```

Then run, for example, the anxiety scenario:

```bash
python 02_mdp_model/anxiety.py
```

### 3.2 Emotion-Based Reward (Rule-Based Appraisal)

```python
USE_NEURAL_APPRAISAL = False
USE_EMOTION_REWARD = True
EMOTION_REWARD_MODE = "replace"  # or "mix"
```

Run a scenario as before and observe differences in cumulative reward and
policy.

### 3.3 Emotion-Based Reward (Neural Appraisal)

After installing PyTorch (see `Exp3/requirements.txt`):

```python
USE_NEURAL_APPRAISAL = True
USE_EMOTION_REWARD = True
```

Now the reward signal depends on the neural appraisal output. Changing the
neural network (architecture, training regime) should lead to different
learning dynamics.

---

## 4. Future Directions

Planned and possible extensions include:

- **Richer inputs to neural appraisal**  
	Move from purely state-based input to `[state, TD-error, Q-values]` so that
	emotion becomes explicitly learning-aware.

- **Temporal / sequence modeling**  
	Extend `NeuralAppraisal` with an LSTM or GRU to handle sequences of states
	and appraisals: `emotion_t = f(state_t, hidden_state_{t-1})`.

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