"""
IPPO model with CNN encoder + decentralized critic for HarvestEnv.

Architecture:
  Actor:  (15,15,3) uint8 -> Conv(3->16, k=3) -> Conv(16->32, k=3) -> Flatten
                          -> Dense [128, 64] Tanh -> 6 action logits
  Critic: (15,15,3) uint8 (= same local obs) -> same Conv stack (separate weights)
                                              -> Dense [512, 256, 128] ReLU -> 1 value

The critic shares the actor's Conv topology but NOT its weights — this matches
the Phase-1 v2 discipline (vf_share_layers: false). No postprocessing hook is
required because the critic only sees the agent's own local observation, which
RLlib's default PPO postprocessing already provides via SampleBatch.OBS.

Mirrors RP-5/models/ippo_model.py's class structure (TorchModelV2 + nn.Module,
value normalization, orthogonal init, output gain) so the training pipeline
behaves identically across phases.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


# ── CNN encoder helpers ───────────────────────────────────────────────────────


def _conv_output_hw(in_h: int, in_w: int, kernel: int, stride: int = 1, pad: int = 0):
    out_h = (in_h + 2 * pad - kernel) // stride + 1
    out_w = (in_w + 2 * pad - kernel) // stride + 1
    return out_h, out_w


def _build_conv_stack(
    in_channels: int,
    in_h: int,
    in_w: int,
    conv_specs: List[Tuple[int, int]],   # list of (out_channels, kernel)
) -> Tuple[nn.Sequential, int]:
    """Stack of Conv2d -> ReLU layers (stride 1, pad 0). Returns (module, flat_dim)."""
    layers = []
    c, h, w = in_channels, in_h, in_w
    for out_c, k in conv_specs:
        layers.append(nn.Conv2d(c, out_c, kernel_size=k, stride=1, padding=0))
        layers.append(nn.ReLU(inplace=True))
        h, w = _conv_output_hw(h, w, kernel=k)
        c = out_c
    flat_dim = c * h * w
    return nn.Sequential(*layers), flat_dim


def _orthogonal_init_dense(layer: nn.Linear, gain: float):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


# ── Model ─────────────────────────────────────────────────────────────────────


class IPPOCNNModelDecentralizedCritic(TorchModelV2, nn.Module):
    """IPPO with decentralized critic, both branches over the local 15x15x3 patch."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config: ModelConfigDict,
        name: str,
        **kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom = model_config.get("custom_model_config", {}) or {}

        # Observation shape from the Box space (H, W, C). RLlib passes obs in
        # NHWC layout from gymnasium Box; we transpose to NCHW for Conv2d.
        obs_shape = tuple(obs_space.shape)
        if len(obs_shape) != 3:
            raise ValueError(
                f"IPPOCNNModelDecentralizedCritic expects (H, W, C) obs space, got {obs_shape}"
            )
        self.obs_h, self.obs_w, self.obs_c = obs_shape

        # Architecture knobs (match plan defaults; YAML-overridable).
        self.actor_conv_specs   = custom.get("actor_conv_specs",   [(16, 3), (32, 3)])
        self.actor_hiddens      = custom.get("actor_hiddens",      [128, 64])
        self.actor_activation   = custom.get("actor_activation",   "tanh")
        self.critic_conv_specs  = custom.get("critic_conv_specs",  [(16, 3), (32, 3)])
        self.critic_hiddens     = custom.get("critic_hiddens",     [512, 256, 128])
        self.critic_activation  = custom.get("critic_activation",  "relu")
        self.use_orthogonal_init = custom.get("use_orthogonal_init", True)
        self.orthogonal_gain    = custom.get("orthogonal_gain", 0.01)
        self.use_value_normalization = custom.get("use_value_normalization", True)

        self._build_actor()
        self._build_critic()

        if self.use_value_normalization:
            self.value_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.value_std  = nn.Parameter(torch.ones(1),  requires_grad=False)
            self.value_momentum = 0.99

        self._last_obs: TensorType = None  # cached for value_function()

    # ── Network construction ──────────────────────────────────────────────────

    def _build_actor(self):
        self.actor_conv, flat = _build_conv_stack(
            self.obs_c, self.obs_h, self.obs_w, self.actor_conv_specs
        )

        act_fn = nn.Tanh if self.actor_activation == "tanh" else nn.ReLU
        layers, prev = [], flat
        for h in self.actor_hiddens:
            lin = nn.Linear(prev, h)
            if self.use_orthogonal_init:
                _orthogonal_init_dense(lin, gain=np.sqrt(5.0 / 3.0))
            layers += [lin, act_fn()]
            prev = h

        head = nn.Linear(prev, self.num_outputs)
        if self.use_orthogonal_init:
            _orthogonal_init_dense(head, gain=self.orthogonal_gain)
        layers.append(head)
        self.actor_head = nn.Sequential(*layers)

    def _build_critic(self):
        self.critic_conv, flat = _build_conv_stack(
            self.obs_c, self.obs_h, self.obs_w, self.critic_conv_specs
        )

        act_fn = nn.Tanh if self.critic_activation == "tanh" else nn.ReLU
        layers, prev = [], flat
        for h in self.critic_hiddens:
            lin = nn.Linear(prev, h)
            if self.use_orthogonal_init:
                _orthogonal_init_dense(lin, gain=np.sqrt(2.0))
            layers += [lin, act_fn()]
            prev = h

        head = nn.Linear(prev, 1)
        if self.use_orthogonal_init:
            _orthogonal_init_dense(head, gain=1.0)
        layers.append(head)
        self.critic_head = nn.Sequential(*layers)

    # ── Forward ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_image_obs(obs: TensorType) -> TensorType:
        """uint8 (N, H, W, C) -> float32 [0,1] in NCHW. If already float, scale heuristically."""
        x = obs
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
            # If the rollout connector pre-normalised to [0, 1], skip; else scale.
            if x.max() > 1.5:
                x = x / 255.0
        # NHWC -> NCHW
        return x.permute(0, 3, 1, 2).contiguous()

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        obs_nhwc = input_dict["obs"]
        x = self._normalize_image_obs(obs_nhwc)
        self._last_obs = x

        feat = self.actor_conv(x)
        feat = feat.flatten(start_dim=1)
        logits = self.actor_head(feat)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        if self._last_obs is None:
            return torch.zeros(1, dtype=torch.float32)
        feat = self.critic_conv(self._last_obs)
        feat = feat.flatten(start_dim=1)
        values = self.critic_head(feat)
        if self.use_value_normalization and self.training:
            values = values * self.value_std + self.value_mean
        return values.squeeze(-1)

    def update_value_normalization(self, values: torch.Tensor):
        if not self.use_value_normalization:
            return
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
        return []


ModelCatalog.register_custom_model("ippo_cnn_decentralized", IPPOCNNModelDecentralizedCritic)
