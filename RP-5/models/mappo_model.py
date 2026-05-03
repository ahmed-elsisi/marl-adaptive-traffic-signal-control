"""
EVALUATION FIX: MAPPO Model with Safe Centralized Critic

Handles missing other_agent_batches during evaluation gracefully.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


class MAPPOModelCentralizedCritic(TorchModelV2, nn.Module):
    """
    MAPPO with Centralized Critic for 70-dim observations.
    
    Architecture:
    - Actor (decentralized): 70-dim local obs → 64 → 64 → 4 actions
    - Critic (centralized): 280-dim global state → 256 → 128 → 1 value
    """
    
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config: ModelConfigDict,
        name: str,
        **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        
        # Get custom config
        custom_config = model_config.get('custom_model_config', {})
        
        # Dimensions
        self.local_obs_dim = 70  # Enhanced obs from obs_builder_v2
        self.num_agents = custom_config.get('num_agents', 4)
        self.global_state_dim = self.local_obs_dim * self.num_agents  # 280
        
        # Network configs
        self.actor_hiddens = custom_config.get('actor_hiddens', [64, 64])
        self.critic_hiddens = custom_config.get('critic_hiddens', [256, 128])
        self.actor_activation = custom_config.get('actor_activation', 'tanh')
        self.critic_activation = custom_config.get('critic_activation', 'relu')
        self.use_orthogonal_init = custom_config.get('use_orthogonal_init', True)
        self.orthogonal_gain = custom_config.get('orthogonal_gain', 0.01)
        
        # LSTM (optional)
        self.use_lstm = custom_config.get('use_lstm', False)
        self.lstm_cell_size = custom_config.get('lstm_cell_size', 64)
        
        # Value normalization
        self.use_value_normalization = custom_config.get('use_value_normalization', True)
        
        # Build networks
        self._build_actor()
        self._build_critic()
        
        # Value normalization parameters
        if self.use_value_normalization:
            self.value_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.value_std = nn.Parameter(torch.ones(1), requires_grad=False)
            self.value_momentum = 0.99
        
        # Cache for global state
        self._last_global_state = None
        self._last_obs = None
    
    def _build_actor(self):
        """Build decentralized actor (uses LOCAL observation - 70-dim)."""
        layers = []
        prev_size = self.local_obs_dim  # 70
        
        # Pre-LSTM or direct processing
        if self.use_lstm:
            self.actor_pre_lstm = nn.Linear(prev_size, self.actor_hiddens[0])
            if self.use_orthogonal_init:
                nn.init.orthogonal_(self.actor_pre_lstm.weight, gain=np.sqrt(2))
                nn.init.constant_(self.actor_pre_lstm.bias, 0)
            
            self.actor_lstm = nn.LSTM(
                self.actor_hiddens[0],
                self.lstm_cell_size,
                batch_first=True
            )
            prev_size = self.lstm_cell_size
            start_idx = 1
        else:
            start_idx = 0
        
        # Hidden layers
        act_fn = nn.Tanh if self.actor_activation == 'tanh' else nn.ReLU
        hidden_init_gain = np.sqrt(5/3) if self.actor_activation == 'tanh' else np.sqrt(2)
        for i in range(start_idx, len(self.actor_hiddens)):
            hidden_size = self.actor_hiddens[i]
            layer = nn.Linear(prev_size, hidden_size)
            if self.use_orthogonal_init:
                nn.init.orthogonal_(layer.weight, gain=hidden_init_gain)
                nn.init.constant_(layer.bias, 0)
            layers.extend([layer, act_fn()])
            prev_size = hidden_size
        
        # Output layer
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
        """Build centralized critic (uses GLOBAL STATE - 280-dim)."""
        layers = []
        prev_size = self.global_state_dim  # 280 (CRITICAL!)
        
        # Larger network for centralized critic
        act_fn = nn.Tanh if self.critic_activation == 'tanh' else nn.ReLU
        for hidden_size in self.critic_hiddens:
            layer = nn.Linear(prev_size, hidden_size)
            if self.use_orthogonal_init:
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
            layers.extend([layer, act_fn()])
            prev_size = hidden_size
        
        # Value output
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
        seq_lens: TensorType
    ) -> Tuple[TensorType, List[TensorType]]:
        """
        Forward pass for actor (uses LOCAL observation - 70-dim).
        """
        obs = input_dict['obs'].float()  # (batch, 70)
        self._last_obs = obs
        
        # Cache global state if provided
        if 'global_state' in input_dict:
            self._last_global_state = input_dict['global_state'].float()
        
        # Actor forward
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
        """
        Value function using GLOBAL STATE (centralized critic - 280-dim).
        
        EVALUATION FIX: Handles cases where global state is incomplete.
        """
        if self._last_global_state is not None:
            # Check if global state looks valid (not mostly zeros from eval)
            non_zero_ratio = (self._last_global_state.abs() > 1e-6).float().mean()
            
            if non_zero_ratio >= 0.4:  # At least 40% non-zero (reasonable threshold)
                # Normal training: use full global state
                global_state = self._last_global_state
            else:
                # Evaluation mode: global state is incomplete
                # Replicate local obs to fill critic input
                batch_size = self._last_obs.shape[0]
                global_state = self._last_obs.repeat(1, self.num_agents)  # 70 → 280
        else:
            # Initialization case
            if self._last_obs is not None:
                batch_size = self._last_obs.shape[0]
                global_state = torch.zeros(
                    batch_size,
                    self.global_state_dim,
                    dtype=self._last_obs.dtype,
                    device=self._last_obs.device
                )
                global_state[:, :self.local_obs_dim] = self._last_obs
            else:
                return torch.zeros(1, dtype=torch.float32)
        
        # Critic forward pass
        values = self.critic(global_state)
        
        # Apply value normalization if enabled
        if self.use_value_normalization and self.training:
            values = values * self.value_std + self.value_mean
        
        return values.squeeze(-1)
    
    def update_value_normalization(self, values: torch.Tensor):
        """Update running statistics for value normalization."""
        if self.use_value_normalization:
            with torch.no_grad():
                batch_mean = values.mean()
                batch_std = values.std()
                
                # Update running statistics with momentum
                self.value_mean.data = (
                    self.value_momentum * self.value_mean.data +
                    (1 - self.value_momentum) * batch_mean
                )
                self.value_std.data = (
                    self.value_momentum * self.value_std.data +
                    (1 - self.value_momentum) * batch_std
                )
                
                # Prevent division by zero
                self.value_std.data = torch.clamp(self.value_std.data, min=1e-6)
    
    @override(TorchModelV2)
    def get_initial_state(self) -> List[TensorType]:
        """Get initial LSTM state."""
        if self.use_lstm:
            h = torch.zeros(1, 1, self.lstm_cell_size)
            c = torch.zeros(1, 1, self.lstm_cell_size)
            return [h, c]
        return []


def centralized_critic_postprocessing(
    policy,
    sample_batch,
    other_agent_batches=None,
    episode=None
):
    """
    EVALUATION FIX: Safe centralized critic postprocessing.
    
    Handles missing other_agent_batches gracefully for evaluation.
    """
    import torch
    
    # CRITICAL: Define fixed global agent order
    AGENT_ORDER = ["J1", "J2", "J3", "J4"]
    
    # Get this agent's observations (batch_size, 70)
    local_obs = sample_batch['obs']
    if not isinstance(local_obs, torch.Tensor):
        local_obs = torch.from_numpy(local_obs).float()
    
    # Check if we have other agents (training mode)
    if not other_agent_batches or len(other_agent_batches) == 0:
        # EVALUATION MODE: No other agents available
        # Use local obs only (will be detected by model's non_zero_ratio check)
        # Pad to 280-dim but mark as incomplete by keeping it sparse
        batch_size = local_obs.shape[0]
        global_state = torch.zeros(
            batch_size, 280,
            dtype=local_obs.dtype,
            device=local_obs.device
        )
        # Only fill first 70 dims (agent will detect this pattern)
        global_state[:, :70] = local_obs
        sample_batch['global_state'] = global_state
        return sample_batch
    
    # TRAINING MODE: Build proper global state with all agents
    obs_by_agent = {}
    
    # Add other agents' observations
    for agent_id, (policy_id, batch) in other_agent_batches.items():
        other_obs = batch['obs']
        if not isinstance(other_obs, torch.Tensor):
            other_obs = torch.from_numpy(other_obs).float()
        obs_by_agent[agent_id] = other_obs
    
    # Determine current agent (the one NOT in other_agent_batches)
    for agent_id in AGENT_ORDER:
        if agent_id not in obs_by_agent:
            obs_by_agent[agent_id] = local_obs
            break
    
    # CRITICAL: Concatenate in FIXED global order
    all_agent_obs = []
    for agent_id in AGENT_ORDER:
        if agent_id in obs_by_agent:
            all_agent_obs.append(obs_by_agent[agent_id])
        else:
            # Should not happen in training
            batch_size = local_obs.shape[0]
            zeros = torch.zeros(
                batch_size, 70,
                dtype=local_obs.dtype,
                device=local_obs.device
            )
            all_agent_obs.append(zeros)
    
    # Concatenate to form global state (batch_size, 280)
    # ALWAYS: [J1, J2, J3, J4] regardless of which agent is calling
    global_state = torch.cat(all_agent_obs, dim=-1)
    
    # Verify dimensions
    assert global_state.shape[-1] == 280, \
        f"Global state should be 280-dim, got {global_state.shape[-1]}"
    
    # Add global state to batch
    sample_batch['global_state'] = global_state
    
    return sample_batch


# Register model
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("mappo_centralized", MAPPOModelCentralizedCritic)