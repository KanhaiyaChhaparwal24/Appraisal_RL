import random
from operator import itemgetter
import csv
import json
import os
import sys
from datetime import datetime

from appraisal_model import RuleBasedAppraisalModel, NeuralAppraisalModel

# Flags are controlled primarily via environment variables so that different
# experimental conditions can be launched without editing this file.

# Flag to switch between rule-based and neural appraisal
USE_NEURAL_APPRAISAL = os.getenv("USE_NEURAL_APPRAISAL", "0") == "1"

# Flag to switch between original state-based reward and emotion-aware reward shaping
USE_EMOTION_REWARD = os.getenv("USE_EMOTION_REWARD", "0") == "1"

# Reward shaping strength: r' = r + LAMBDA * f(appraisal)
try:
    EMOTION_REWARD_LAMBDA = float(os.getenv("EMOTION_REWARD_LAMBDA", "0.5"))
except ValueError:
    EMOTION_REWARD_LAMBDA = 0.5

# Per-step logging of TD error, appraisal, Q-values, and action
LOG_STEPS = os.getenv("LOG_STEPS", "0") == "1"

_this_dir = os.path.dirname(__file__)
_exp_dir = os.path.dirname(_this_dir)
LOG_DIR = os.path.join(_exp_dir, "logs")

class agent():
    def __init__(self,mdp):
        self.epsilon = 0.3
        self.gamma = 0.9
        # discount factor not so important
        self.alpha = 0.3
        # make different plots for different alpha 0.3, 0.5
        self.mdp = mdp
        self.q={}
        self.td_error = 0
        self.old_q = 0
        self.t_hat = {}
        self.max_q_table = 0
        # placeholders for appraisal values and neural model state
        self.sud_app = 0.0
        self.goal_app = 0.0
        self.cdc_app = 0.0
        self.power_app = 0.0
        self.last_neural_loss = None
        self.cumulative_reward = 0.0
        self.step_index = 0
        self._step_log_path = None
        #Q table is for every State Action pair.

        # build state index for neural model input if permitted_states available
        if hasattr(self.mdp, "permitted_states"):
            self._state_list = list(self.mdp.permitted_states)
        else:
            self._state_list = list(self.mdp.t.keys())
        self._state_to_idx = {s: i for i, s in enumerate(self._state_list)}

        for s in self.mdp.t.keys():
            self.q[s]={}
            self.t_hat[s]={}
            for a in self.mdp.t[s]:
                self.q[s][a]=0
                self.t_hat[s][a]={}
                for s2 in self.mdp.t.keys():
                    self.t_hat[s][a][s2]=0

        # initialise appraisal models
        self.rule_appraisal_model = RuleBasedAppraisalModel(self)
        self.neural_appraisal_model = None
        if USE_NEURAL_APPRAISAL:
            self.neural_appraisal_model = NeuralAppraisalModel(self)

        # set up step-level logging
        if LOG_STEPS:
            os.makedirs(LOG_DIR, exist_ok=True)
            scenario = self.mdp.__class__.__name__
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._step_log_path = os.path.join(
                LOG_DIR,
                f"{scenario}_steps_{timestamp}.csv",
            )
            with open(self._step_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step",
                    "state",
                    "next_state",
                    "action",
                    "reward",
                    "td_error",
                    "suddenness",
                    "goal_relevance",
                    "conduciveness",
                    "power",
                    "q_values_json",
                ])
    

    def update_q_learning(self):
        if  self.mdp.action != None and self.mdp.action not in self.q[self.mdp.previous_state]:
            self.q[self.mdp.previous_state][self.mdp.action] = 0
            self.t_hat[self.mdp.previous_state][self.mdp.action]={}
            self.t_hat[self.mdp.previous_state][self.mdp.action][self.mdp.state]=0

        if self.mdp.previous_state != None:
            previous_q = self.q[self.mdp.previous_state][self.mdp.action]
            self.old_q = previous_q
            next_q = max(self.q[self.mdp.state].items(), key = itemgetter(1))[1]
            self.td_error = self.alpha * (self.mdp.reward + self.gamma * next_q - previous_q)
            new_q = previous_q + self.td_error
            self.q[self.mdp.previous_state][self.mdp.action] = new_q
            self.t_hat[self.mdp.previous_state][self.mdp.action][self.mdp.state] += 1
            # Here, next_q is calculated in terms of the max_q, so it doesn't know about 
            # the action it is going to take. It is still expecting the best thing to happen
            # Until it actually takes the action.
        

    def get_td_error(self):
        if self.mdp.previous_state != None:
            tde = self.td_error
            # self.mdp.tde_sum[self.mdp.state] += tde
            self.mdp.tde.append(tde)

    def update_q_td(self):
        previous_q = self.q[self.mdp.previous_state][self.mdp.action]

        self.q[self.mdp.previous_state][self.mdp.action] = \
            previous_q + self.alpha * (self.mdp.reward - previous_q)
        
        self.t_hat[self.mdp.previous_state][self.mdp.action][self.mdp.state] += 1

    def get_max_q_table(self):
        max_q_table = max({key: max(val.values()) for key, val in self.q.items()}.values())
        if max_q_table == 0:
            max_q_table = max_q_table + 1
        self.max_q_table = max_q_table
        return max_q_table   

    def _emotion_to_reward(self, emotion):
        """Map an emotion vector to a scalar reward signal.

        emotion: [suddenness, goal_relevance, conduciveness, power]

        All components are softly clamped to [0, 1] to avoid instability.
        """
        if emotion is None or len(emotion) != 4:
            return self.mdp.reward

        sud, goal, cdc, power = emotion

        def _clip01(x):
            try:
                return max(0.0, min(1.0, float(x)))
            except (TypeError, ValueError):
                return 0.0

        sud = _clip01(sud)
        goal = _clip01(goal)
        cdc = _clip01(cdc)
        power = _clip01(power)

        # Simple linear combination; can be tuned or replaced later.
        reward = (
            1.0 * goal +
            1.0 * cdc -
            0.5 * sud +
            0.5 * power
        )
        return reward

    def compute_emotion(self, train_neural: bool = False, log_shapes: bool = False):
        """Return the current emotion/appraisal vector.

        - Always computes the rule-based appraisals via RuleBasedAppraisalModel.
        - If USE_NEURAL_APPRAISAL is enabled and the neural model is
          available, it produces a neural appraisal vector using learning-
          aware features (TD error, reward, Q-values).
        - The chosen vector is stored in self.sud_app, self.goal_app,
          self.cdc_app, self.power_app and also returned.
        """
        state = self.mdp.previous_state
        action = self.mdp.previous_action
        reward = getattr(self.mdp, "reward", 0.0)
        next_state = self.mdp.state

        rule_emotion = self.rule_appraisal_model.compute(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
        )

        # Default: use rule-based emotion
        chosen = list(rule_emotion)

        # Optional neural appraisal
        if USE_NEURAL_APPRAISAL and self.neural_appraisal_model is not None:
            chosen = self.neural_appraisal_model.compute(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                train=train_neural,
                target=rule_emotion,
            )

        if len(chosen) == 4:
            self.sud_app, self.goal_app, self.cdc_app, self.power_app = chosen

        if log_shapes:
            print("[Appraisal] vec:", [round(float(v), 4) for v in chosen])

        return chosen

    def choose_action_epsilon_greedy(self): 
        self.mdp.previous_action = self.mdp.action
        # if random.random() < self.epsilon and self.mdp.state == "P":


        if(self.mdp.story_m or self.mdp.model_changed) and self.mdp.state == self.mdp.chosen_state:
            self.mdp.action = self.mdp.chosen_action
        elif random.random() < self.epsilon:
            actions = []
            for key, value in self.mdp.t[self.mdp.state].items():
                actions.append(key)
            self.mdp.action = random.choice(actions)
        else:
            self.mdp.action = max(self.q[self.mdp.state].items(), key = itemgetter(1))[0]

    def do_step(self):
        # Q-learning update for the previous transition uses the reward
        # computed in the *last* call to do_step().
        self.update_q_learning()
        self.get_td_error()
        self.choose_action_epsilon_greedy()

        # Environment transition for the current action
        self.mdp.transition()

        # Base reward from the original MDP definition
        self.mdp.calculate_reward()
        base_reward = self.mdp.reward

        # Optional emotion-aware reward shaping: r' = r + lambda * f(appraisal)
        if USE_EMOTION_REWARD:
            emotion = self.compute_emotion(train_neural=False, log_shapes=False)
            emotion_reward = self._emotion_to_reward(emotion)
            self.mdp.reward = base_reward + EMOTION_REWARD_LAMBDA * emotion_reward

        # Track cumulative reward for logging at the episode level
        self.cumulative_reward += self.mdp.reward

        # Per-step logging (optional)
        if LOG_STEPS and self._step_log_path is not None:
            self.step_index += 1
            # Ensure appraisal attributes are up to date for logging
            emo_vec = self.compute_emotion(train_neural=False, log_shapes=False)
            state = self.mdp.previous_state
            next_state = self.mdp.state
            action = self.mdp.previous_action
            q_vals = self.q.get(next_state, {})
            with open(self._step_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.step_index,
                    state,
                    next_state,
                    action,
                    self.mdp.reward,
                    self.td_error,
                    emo_vec[0] if len(emo_vec) > 0 else None,
                    emo_vec[1] if len(emo_vec) > 1 else None,
                    emo_vec[2] if len(emo_vec) > 2 else None,
                    emo_vec[3] if len(emo_vec) > 3 else None,
                    json.dumps(q_vals),
                ])

        if self.mdp.terminal:
            self.update_q_td()

    def train(self,i_max,i_change=0):
        i=0
        while i < i_max:
            if i == i_max-i_change and not self.mdp.model_changed:
                self.mdp.make_transition(story_mode = False, model_changed = True)
                # self.mdp.model_changed = True
            self.do_step()

            if self.mdp.terminal:
                i += 1
                self.mdp.reset()

    def simulate_episode(self, terminate = None):
        # if not self.internal_active and app_acc:
        #     # when calculating instd, the behavior doesn't need to apprised 
        #     # only the sccores are trying to be calculated
        #     # instd needs to be active inside generate_instandard()
        #     # self.instandard=self.generate_instandard()
        #     sm=True

        # if self.mdp.repr_is_state:
        # self.terminate_else = terminate_else
        self.mdp.make_transition(story_mode = True) 
        self.mdp.reset()
        self.cumulative_reward = 0.0

        while True:
            self.do_step()
            if self.mdp.terminal:
                # Episode finished; log cumulative reward and final emotion
                final_emotion = self.compute_emotion(train_neural=False, log_shapes=False)
                print("Episode finished. Cumulative reward:", round(self.cumulative_reward, 3))
                print("Final emotion [sud, goal, cdc, power]:",
                      [round(float(v), 4) for v in final_emotion])
                return

            if terminate == self.mdp.state:

                self.update_q_learning()
                self.get_td_error()
                # self.choose_action_epsilon_greedy()
                self.get_max_q_table()
                rounded_tde = [round(num,3)for num in self.mdp.tde]
                print("Q value:\t",self.q)
                print("TDE list:\t", rounded_tde)
                print("Manual terminate")
                # Compute emotion; keep neural immutable here (no imitation).
                emotion = self.compute_emotion(train_neural=False, log_shapes=True)
                print("Emotion vector [sud, goal, cdc, power]:",
                      [round(float(v), 4) for v in emotion])
                print("Cumulative reward so far:", round(self.cumulative_reward, 3))
                # print("In standard:\t", round(self.appraise_instandard(),4))
                return

    def appraise_power(self):
        # If two q are very high, the power is very low
        # reward having too much influence
        # state = self.mdp.state
        state = self.mdp.chosen_state
        if state is None or state not in self.q:
            self.power_app = 0.0
            return self.power_app
        avg_q = sum(self.q[state].values())/len(self.q[state].values())
        min_q = min(self.q[state].values())
        max_q = max(self.q[state].values())
        if abs(min_q)<max_q:
            self.power_app = abs((max_q-avg_q)/max_q)
        else:
            self.power_app = abs((min_q-avg_q)/min_q)
        return self.power_app

    def appraise_goal_relevance(self):
        self.goal_app = min(1,abs(self.td_error))
        return self.goal_app

    def appraise_suddenness(self):
        # It calculates p(s'|at-1)
        prev_state = self.mdp.previous_state
        prev_action = self.mdp.previous_action
        if prev_state not in self.t_hat or prev_action not in self.t_hat[prev_state]:
            self.sud_app = 0
            return self.sud_app
        s = sum(self.t_hat[prev_state][prev_action].values())
        if s > 0:
            self.sud_app = 1 - self.t_hat[prev_state][prev_action][self.mdp.state] / s
            # suddennes = 1- (frequency)
        else:
            self.sud_app = 0
        return self.sud_app
        
    def appraise_conduciveness(self):
        self.cdc_app = max(-1,min(1, self.td_error))/2+0.5
        return self.cdc_app