"""
IPPO Model with Decentralized Critic.

Companion to MAPPOModelCentralizedCritic. The actor is identical, but the
critic sees only the agent's own 70-dim local observation rather than the
concatenated 280-dim global state. No postprocess hook is required — RLlib's
default PPO postprocessing computes GAE on local observations.

This isolates "centralized vs decentralized critic" as the only varied factor
between the MAPPO and IPPO experiments.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


class IPPOModelDecentralizedCritic(TorchModelV2, nn.Module):
    """
    IPPO with Decentralized Critic for 70-dim observations.

    Architecture:
    - Actor:  70-dim local obs -> hidden layers -> 4 actions
    - Critic: 70-dim local obs -> hidden layers -> 1 value
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config: ModelConfigDict,
        name: str,
        **kwargs,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        custom_config = model_config.get("custom_model_config", {})

        self.local_obs_dim = 70
        self.num_agents = custom_config.get("num_agents", 4)
        self.agent_ids = custom_config.get("agent_ids", ["J1", "J2", "J3", "J4"])

        self.actor_hiddens = custom_config.get("actor_hiddens", [64, 64])
        self.actor_activation = custom_config.get("actor_activation", "tanh")
        self.critic_hiddens = custom_config.get("critic_hiddens", [256, 128])
        self.critic_activation = custom_config.get("critic_activation", "relu")
        self.use_orthogonal_init = custom_config.get("use_orthogonal_init", True)
        self.orthogonal_gain = custom_config.get("orthogonal_gain", 0.01)

        self.use_lstm = custom_config.get("use_lstm", False)
        self.lstm_cell_size = custom_config.get("lstm_cell_size", 64)

        self.use_value_normalization = custom_config.get("use_value_normalization", True)

        self._build_actor()
        self._build_critic()

        if self.use_value_normalization:
            self.value_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.value_std = nn.Parameter(torch.ones(1), requires_grad=False)
            self.value_momentum = 0.99

        self._last_obs = None

    def _build_actor(self):
        """Decentralized actor — identical to MAPPO."""
        layers = []
        prev_size = self.local_obs_dim

        if self.use_lstm:
            self.actor_pre_lstm = nn.Linear(prev_size, self.actor_hiddens[0])
            if self.use_orthogonal_init:
                nn.init.orthogonal_(self.actor_pre_lstm.weight, gain=np.sqrt(2))
                nn.init.constant_(self.actor_pre_lstm.bias, 0)

            self.actor_lstm = nn.LSTM(
                self.actor_hiddens[0],
                self.lstm_cell_size,
                batch_first=True,
            )
            prev_size = self.lstm_cell_size
            start_idx = 1
        else:
            start_idx = 0

        act_fn = nn.Tanh if self.actor_activation == "tanh" else nn.ReLU
        for i in range(start_idx, len(self.actor_hiddens)):
            hidden_size = self.actor_hiddens[i]
            layer = nn.Linear(prev_size, hidden_size)
            if self.use_orthogonal_init:
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(5 / 3))
                nn.init.constant_(layer.bias, 0)
            layers.extend([layer, act_fn()])
            prev_size = hidden_size

        output_layer = nn.Linear(prev_size, self.num_outputs)
        if self.use_orthogonal_init:
            nn.init.orthogonal_(output_layer.weight, gain=self.orthogonal_gain)
            nn.init.constant_(output_layer.bias, 0)
        layers.append(output_layer)

        if not self.use_lstm:
            self.actor = nn.Sequential(*layers)
        else:
            self.actor_post_lstm = nn.Sequential(*layers)

    def _build_critic(self):
        """Decentralized critic — input is 70-dim local obs (vs 280 for MAPPO)."""
        layers = []
        prev_size = self.local_obs_dim

        act_fn = nn.Tanh if self.critic_activation == "tanh" else nn.ReLU
        for hidden_size in self.critic_hiddens:
            layer = nn.Linear(prev_size, hidden_size)
            if self.use_orthogonal_init:
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
            layers.extend([layer, act_fn()])
            prev_size = hidden_size

        value_layer = nn.Linear(prev_size, 1)
        if self.use_orthogonal_init:
            nn.init.orthogonal_(value_layer.weight, gain=1.0)
            nn.init.constant_(value_layer.bias, 0)
        layers.append(value_layer)

        self.critic = nn.Sequential(*layers)

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        obs = input_dict["obs"].float()
        self._last_obs = obs

        if self.use_lstm:
            x = torch.tanh(self.actor_pre_lstm(obs))
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            if state:
                x, (h, c) = self.actor_lstm(x, (state[0], state[1]))
                state_out = [h, c]
            else:
                x, (h, c) = self.actor_lstm(x)
                state_out = [h, c]

            x = x.squeeze(1)
            action_logits = self.actor_post_lstm(x)
        else:
            action_logits = self.actor(obs)
            state_out = []

        return action_logits, state_out

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """Value from local 70-dim observation only — no global state."""
        if self._last_obs is None:
            return torch.zeros(1, dtype=torch.float32)

        values = self.critic(self._last_obs)

        if self.use_value_normalization and self.training:
            values = values * self.value_std + self.value_mean

        return values.squeeze(-1)

    def update_value_normalization(self, values: torch.Tensor):
        if self.use_value_normalization:
            with torch.no_grad():
                batch_mean = values.mean()
                batch_std = values.std()

                self.value_mean.data = (
                    self.value_momentum * self.value_mean.data
                    + (1 - self.value_momentum) * batch_mean
                )
                self.value_std.data = (
                    self.value_momentum * self.value_std.data
                    + (1 - self.value_momentum) * batch_std
                )
                self.value_std.data = torch.clamp(self.value_std.data, min=1e-6)

    @override(TorchModelV2)
    def get_initial_state(self) -> List[TensorType]:
        if self.use_lstm:
            h = torch.zeros(1, 1, self.lstm_cell_size)
            c = torch.zeros(1, 1, self.lstm_cell_size)
            return [h, c]
        return []


from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("ippo_decentralized", IPPOModelDecentralizedCritic)
