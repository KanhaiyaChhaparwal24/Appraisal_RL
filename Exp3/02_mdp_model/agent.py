import random
from operator import itemgetter
import csv
import os
import sys

# Flag to switch between rule-based and neural appraisal
USE_NEURAL_APPRAISAL = False

# Flag to switch between original state-based reward and emotion-based reward
USE_EMOTION_REWARD = False

# How to combine base (MDP) reward and emotion reward: "replace" or "mix"
EMOTION_REWARD_MODE = "replace"

# When EMOTION_REWARD_MODE == "mix":
#   final_reward = BASE_WEIGHT * base_reward + (1-BASE_WEIGHT) * emotion_reward
EMOTION_REWARD_BASE_WEIGHT = 0.5

# Make sure we can import from the repository root (for models.neural_appraisal)
_this_dir = os.path.dirname(__file__)
_repo_root = os.path.dirname(os.path.dirname(_this_dir))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

try:
    import torch
    import torch.nn as nn
    from models.neural_appraisal import NeuralAppraisal
except ImportError:
    torch = None
    nn = None
    NeuralAppraisal = None


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

        # initialise neural appraisal model (optional)
        self.neural_appraisal = None
        self._init_neural_appraisal()
    

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

    def _init_neural_appraisal(self):
        """Initialise the neural appraisal model and optimiser if enabled.

        Keeps the existing rule-based pipeline intact when disabled.
        """
        if not USE_NEURAL_APPRAISAL:
            return
        if NeuralAppraisal is None or torch is None:
            print("Neural appraisal requested but PyTorch is not available.")
            return

        input_dim = len(self._state_list)
        output_dim = 4  # [suddenness, goal_relevance, conduciveness, power]
        self.neural_appraisal = NeuralAppraisal(input_dim, output_dim)
        self._na_optimizer = torch.optim.Adam(self.neural_appraisal.parameters(), lr=1e-3)
        self._na_loss_fn = nn.MSELoss()

    def _encode_state_vector(self, state=None):
        """Encode the current MDP state as a one-hot vector for the neural model."""
        if torch is None:
            return None
        if state is None:
            state = getattr(self.mdp, "chosen_state", None) or self.mdp.state
        idx = self._state_to_idx.get(state, None)
        if idx is None:
            return None
        x = torch.zeros(len(self._state_list), dtype=torch.float32)
        x[idx] = 1.0
        return x.unsqueeze(0)

    def _compute_rule_based_emotion(self):
        """Compute emotion using the original rule-based appraisal functions."""
        sud = float(self.appraise_suddenness())
        goal = float(self.appraise_goal_relevance())
        cdc = float(self.appraise_conduciveness())
        power = float(self.appraise_power())
        # keep attributes for downstream code and logging
        self.sud_app, self.goal_app, self.cdc_app, self.power_app = sud, goal, cdc, power
        return [sud, goal, cdc, power]

    def compute_emotion(self, train_neural: bool = False, log_shapes: bool = False):
        """Return the current emotion vector and optionally train the neural model.

        When USE_NEURAL_APPRAISAL is False, this falls back to the original
        rule-based appraisal while still exposing a consistent interface.
        """
        rule_emotion = self._compute_rule_based_emotion()

        if not USE_NEURAL_APPRAISAL or self.neural_appraisal is None:
            # Only rule-based emotion is used.
            return rule_emotion

        x = self._encode_state_vector()
        if x is None:
            # Fallback if encoding fails
            return rule_emotion

        emo_pred = self.neural_appraisal(x)
        if log_shapes:
            print("[NeuralAppraisal] input shape:", tuple(x.shape), "output shape:", tuple(emo_pred.shape))

        # update attributes from neural output
        emo_list = emo_pred.detach().cpu().view(-1).tolist()
        if len(emo_list) == 4:
            self.sud_app, self.goal_app, self.cdc_app, self.power_app = emo_list

        if train_neural:
            target = torch.tensor(rule_emotion, dtype=torch.float32).unsqueeze(0)
            loss = self._na_loss_fn(emo_pred, target)
            self._na_optimizer.zero_grad()
            loss.backward()
            self._na_optimizer.step()
            self.last_neural_loss = float(loss.item())
            print("[NeuralAppraisal] training loss:", round(self.last_neural_loss, 6))
            print("[NeuralAppraisal] rule-based:", [round(v, 4) for v in rule_emotion])
            print("[NeuralAppraisal] neural   :", [round(v, 4) for v in emo_list])

        return emo_list

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

        # Optionally replace or mix the reward with emotion-based reward
        if USE_EMOTION_REWARD:
            # Use the current appraisal as input to the reward function.
            # We do not train the neural model here; this keeps the reward
            # purely a function of the current emotion signal.
            emotion = self.compute_emotion(train_neural=False, log_shapes=False)
            emotion_reward = self._emotion_to_reward(emotion)

            if EMOTION_REWARD_MODE == "replace":
                self.mdp.reward = emotion_reward
            elif EMOTION_REWARD_MODE == "mix":
                self.mdp.reward = (
                    EMOTION_REWARD_BASE_WEIGHT * base_reward +
                    (1.0 - EMOTION_REWARD_BASE_WEIGHT) * emotion_reward
                )
            # Lightweight debugging print; comment out if too verbose.
            # print("[Reward] base=", round(base_reward,3),
            #       "emotion=", round(emotion_reward,3),
            #       "final=", round(self.mdp.reward,3))

        # Track cumulative reward for logging at the episode level
        self.cumulative_reward += self.mdp.reward

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
        s = sum(self.t_hat[self.mdp.previous_state][self.mdp.previous_action].values())
        if s > 0:
            self.sud_app = 1-self.t_hat[self.mdp.previous_state][self.mdp.previous_action][self.mdp.state]/s
            # suddennes = 1- (frequency)
        else:
            self.sud_app = 0
        return self.sud_app
        
    def appraise_conduciveness(self):
        self.cdc_app = max(-1,min(1, self.td_error))/2+0.5
        return self.cdc_app