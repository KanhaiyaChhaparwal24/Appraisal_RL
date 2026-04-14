import math
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    from models.neural_appraisal import NeuralAppraisal
except ImportError:  # pragma: no cover - PyTorch is optional
    torch = None
    nn = None
    NeuralAppraisal = None


class RuleBasedAppraisalModel:
    """Thin wrapper around the original rule-based appraisal functions.

    This keeps the old formulas intact but exposes a common interface:
    compute(state, action, reward, next_state) -> 4D appraisal vector.
    """

    def __init__(self, agent: "agent") -> None:  # type: ignore[name-defined]
        self.agent = agent

    def compute(
        self,
        state: Optional[str],
        action: Optional[str],
        reward: float,
        next_state: Optional[str],
    ) -> List[float]:
        # When there is no previous transition yet (e.g., at the start of an
        # episode), some of the original formulas cannot be evaluated. In that
        # case we fall back to neutral appraisals.
        if self.agent.mdp.previous_state is None or self.agent.mdp.previous_action is None:
            return [0.0, 0.0, 0.0, 0.0]

        # The underlying formulas use the agent's internal TD-error,
        # transition counts, and Q-table, so we simply delegate.
        sud = float(self.agent.appraise_suddenness())
        goal = float(self.agent.appraise_goal_relevance())
        cdc = float(self.agent.appraise_conduciveness())
        power = float(self.agent.appraise_power())
        return [sud, goal, cdc, power]


class NeuralAppraisalModel:
    """Neural appraisal model using learning-aware features.

    Input features per step (concatenated into a 4D vector):
      - TD error (delta)
      - reward r_t
      - max_a' Q(s_{t+1}, a')
      - Q(s_t, a_t)

    Output:
      - 4 appraisal values in [0, 1] via a sigmoid layer:
        [suddenness, goal_relevance, conduciveness, power]

    The model can optionally be trained to imitate the rule-based
    appraisals by providing a target vector.
    """

    def __init__(self, agent: "agent") -> None:  # type: ignore[name-defined]
        self.agent = agent
        self.model = None
        self.optimizer = None
        self.loss_fn = None

        if NeuralAppraisal is not None and torch is not None:
            input_dim = 4
            output_dim = 4
            self.model = NeuralAppraisal(input_dim, output_dim)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            self.loss_fn = nn.MSELoss()

    def _feature_vector(self) -> Optional["torch.Tensor"]:  # type: ignore[name-defined]
        if torch is None or self.model is None:
            return None

        # Current transition information from the agent
        td_error = float(self.agent.td_error)
        reward = float(getattr(self.agent.mdp, "reward", 0.0))

        # s_t and a_t are the previous state/action for which TD was computed
        s_t = self.agent.mdp.previous_state
        a_t = self.agent.mdp.previous_action
        s_tp1 = self.agent.mdp.state

        # Q(s_t, a_t)
        if s_t is not None and a_t is not None and s_t in self.agent.q:
            q_sa = float(self.agent.q[s_t].get(a_t, 0.0))
        else:
            q_sa = 0.0

        # max_a' Q(s_{t+1}, a')
        if s_tp1 is not None and s_tp1 in self.agent.q and self.agent.q[s_tp1]:
            max_q_next = float(max(self.agent.q[s_tp1].values()))
        else:
            max_q_next = 0.0

        x = torch.tensor([td_error, reward, max_q_next, q_sa], dtype=torch.float32)
        return x.unsqueeze(0)

    def compute(
        self,
        state: Optional[str],
        action: Optional[str],
        reward: float,
        next_state: Optional[str],
        train: bool = False,
        target: Optional[List[float]] = None,
    ) -> List[float]:
        # If the neural backend is unavailable, fall back silently to zeros.
        if torch is None or self.model is None:
            return [0.0, 0.0, 0.0, 0.0]

        x = self._feature_vector()
        if x is None:
            return [0.0, 0.0, 0.0, 0.0]

        emo_logits = self.model(x)
        emo = torch.sigmoid(emo_logits)

        if train and self.loss_fn is not None and self.optimizer is not None and target is not None:
            y = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
            loss = self.loss_fn(emo, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        vec = emo.detach().cpu().view(-1).tolist()
        # Ensure finite values in [0, 1]
        cleaned = []
        for v in vec:
            if not math.isfinite(v):
                cleaned.append(0.0)
            else:
                cleaned.append(max(0.0, min(1.0, float(v))))
        return cleaned
