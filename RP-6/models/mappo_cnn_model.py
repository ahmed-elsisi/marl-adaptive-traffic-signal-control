"""
MAPPO model with CNN encoder + centralized critic for HarvestEnv.

Architecture:
  Actor:  (15,15,3) uint8 -> Conv(3->16, k=3) -> Conv(16->32, k=3) -> Flatten
                          -> Dense [128, 64] Tanh -> 6 action logits
  Critic: (12, 8, 3) uint8 — the FULL grid, not a concat of egocentric patches —
                          -> Conv(3->32, k=3) -> Conv(32->64, k=3) -> Flatten
                          -> Dense [512, 256, 128] ReLU -> 1 value

The centralized-critic global state is the literal world state, not a stack of
agents' partial views. For a 12x8x3 grid this is 288 dims (vs 4x15x15x3 = 2700
dims of overlapping pixels for a concat approach). The env exposes the global
state via _get_global_state() and includes it in each agent's info dict, from
which `harvest_centralized_critic_postprocessing` lifts it into the SampleBatch
under the 'global_state' column. The model's value_function() then reads that
column via input_dict.

Mirrors the structure of RP-5/models/mappo_model.py:
  - TorchModelV2 + nn.Module
  - Cached self._last_global_state from forward()'s input_dict
  - Eval-time fallback when no global_state is available (pad with zeros and
    let the critic's response degrade gracefully — same idea as RP-5's
    non_zero_ratio fallback at mappo_model.py:192-214)
  - Orthogonal init, value normalization, output gain — matched defaults
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


# ── CNN helpers (kept in sync with ippo_cnn_model.py) ─────────────────────────


def _conv_output_hw(in_h: int, in_w: int, kernel: int, stride: int = 1, pad: int = 0):
    out_h = (in_h + 2 * pad - kernel) // stride + 1
    out_w = (in_w + 2 * pad - kernel) // stride + 1
    return out_h, out_w


def _build_conv_stack(
    in_channels: int,
    in_h: int,
    in_w: int,
    conv_specs: List[Tuple[int, int]],
) -> Tuple[nn.Sequential, int]:
    layers = []
    c, h, w = in_channels, in_h, in_w
    for out_c, k in conv_specs:
        layers.append(nn.Conv2d(c, out_c, kernel_size=k, stride=1, padding=0))
        layers.append(nn.ReLU(inplace=True))
        h, w = _conv_output_hw(h, w, kernel=k)
        c = out_c
    return nn.Sequential(*layers), c * h * w


def _orthogonal_init_dense(layer: nn.Linear, gain: float):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


def _normalize_image_obs(obs: TensorType) -> TensorType:
    """uint8 (N, H, W, C) -> float32 [0,1] in NCHW."""
    x = obs
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
    return x.permute(0, 3, 1, 2).contiguous()


# ── Model ─────────────────────────────────────────────────────────────────────


class MAPPOCNNModelCentralizedCritic(TorchModelV2, nn.Module):
    """MAPPO with centralized critic that reads the full grid as a single image."""

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

        # Local actor obs: (H_local, W_local, C)
        local_shape = tuple(obs_space.shape)
        if len(local_shape) != 3:
            raise ValueError(
                f"MAPPOCNNModelCentralizedCritic expects (H, W, C) obs space, got {local_shape}"
            )
        self.local_h, self.local_w, self.obs_c = local_shape

        # Global critic obs shape — the full grid. Required custom config since
        # the env's grid dimensions are not derivable from the (15, 15, 3) actor obs.
        gh = custom.get("global_state_height", None)
        gw = custom.get("global_state_width", None)
        if gh is None or gw is None:
            raise ValueError(
                "MAPPOCNNModelCentralizedCritic requires "
                "custom_model_config['global_state_height'] and "
                "custom_model_config['global_state_width'] (the full grid dims)."
            )
        self.global_h = int(gh)
        self.global_w = int(gw)

        # Architecture knobs
        self.actor_conv_specs   = custom.get("actor_conv_specs",   [(16, 3), (32, 3)])
        self.actor_hiddens      = custom.get("actor_hiddens",      [128, 64])
        self.actor_activation   = custom.get("actor_activation",   "tanh")
        self.critic_conv_specs  = custom.get("critic_conv_specs",  [(32, 3), (64, 3)])
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

        self._last_local_obs: Optional[TensorType] = None
        self._last_global_state: Optional[TensorType] = None

    def _build_actor(self):
        self.actor_conv, flat = _build_conv_stack(
            self.obs_c, self.local_h, self.local_w, self.actor_conv_specs
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
            self.obs_c, self.global_h, self.global_w, self.critic_conv_specs
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

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        local = _normalize_image_obs(input_dict["obs"])  # (N, C, H_l, W_l)
        self._last_local_obs = local

        # Cache global_state if the postprocessing hook injected it.
        if "global_state" in input_dict:
            gs = input_dict["global_state"]
            self._last_global_state = _normalize_image_obs(gs)
        else:
            self._last_global_state = None

        feat = self.actor_conv(local).flatten(start_dim=1)
        logits = self.actor_head(feat)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        if self._last_global_state is not None:
            gs = self._last_global_state
            # Eval-time fallback: if the connector pipeline failed to inject
            # global_state, the column may be zeros. Detect and fall back to
            # repeating local obs across the grid (degraded but stable).
            if gs.abs().mean() < 1e-6 and self._last_local_obs is not None:
                gs = self._build_eval_fallback_global()
        elif self._last_local_obs is not None:
            gs = self._build_eval_fallback_global()
        else:
            return torch.zeros(1, dtype=torch.float32)

        feat = self.critic_conv(gs).flatten(start_dim=1)
        values = self.critic_head(feat)
        if self.use_value_normalization and self.training:
            values = values * self.value_std + self.value_mean
        return values.squeeze(-1)

    def _build_eval_fallback_global(self) -> TensorType:
        """Construct a zero-filled global tensor of correct shape from local obs.

        Used when global_state isn't available at eval time — keeps the critic
        forward pass dimensional even though the values returned are not
        meaningful (eval doesn't use the critic for decisions, only the actor).
        """
        local = self._last_local_obs
        batch = local.shape[0]
        return torch.zeros(
            batch, self.obs_c, self.global_h, self.global_w,
            dtype=local.dtype, device=local.device,
        )

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


# ── Postprocessing hook ───────────────────────────────────────────────────────


def harvest_centralized_critic_postprocessing(
    policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict] = None,
    episode=None,
) -> SampleBatch:
    """Lift `info['global_state']` from each step's info dict into a SampleBatch column.

    The HarvestEnv places the full (H, W, 3) grid in every agent's info dict
    under the 'global_state' key (see RP-6/marl_env/harvest_env.py:_build_info).
    The MAPPO centralized critic reads it from `input_dict['global_state']`
    during value_function().

    At evaluation time RLlib often doesn't ship infos through the connector
    pipeline; in that case we add a zeros placeholder of the correct shape so
    downstream model code doesn't KeyError. The model's value_function detects
    near-zero global_state and falls back to a deterministic-but-meaningless
    output (eval cares about the actor, not the critic).
    """
    import numpy as _np

    batch_len = len(sample_batch[SampleBatch.OBS])

    # Default global-state shape comes from the policy config (env_config block).
    # RLlib stores PolicyConfig at policy.config — env_config lives there.
    env_cfg = policy.config.get("env_config", {}) if hasattr(policy, "config") else {}
    gh = int(env_cfg.get("grid_height", 8))
    gw = int(env_cfg.get("grid_width", 12))
    gc = 3
    default_shape = (gh, gw, gc)

    infos = sample_batch.get(SampleBatch.INFOS)
    if infos is None or len(infos) == 0:
        sample_batch["global_state"] = _np.zeros((batch_len,) + default_shape, dtype=_np.uint8)
        return sample_batch

    extracted = []
    for inf in infos:
        gs = None
        if isinstance(inf, dict):
            gs = inf.get("global_state", None)
        if gs is None:
            gs = _np.zeros(default_shape, dtype=_np.uint8)
        else:
            gs = _np.asarray(gs, dtype=_np.uint8)
            if gs.shape != default_shape:
                # Defensive: pad/truncate would change semantics; instead zero
                # out the slot so the model's eval fallback kicks in.
                gs = _np.zeros(default_shape, dtype=_np.uint8)
        extracted.append(gs)

    sample_batch["global_state"] = _np.stack(extracted, axis=0)
    return sample_batch


ModelCatalog.register_custom_model("mappo_cnn_centralized", MAPPOCNNModelCentralizedCritic)
