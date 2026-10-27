import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.net(state)
        probs = torch.softmax(logits, dim=-1)
        return probs


class Critic(nn.Module):
    def __init__(self, full_state_dim: int, full_action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(full_state_dim + full_action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.storage: List[Tuple] = []

    def add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray, dones: np.ndarray):
        self.storage.append((obs, actions, rewards, next_obs, dones))
        if len(self.storage) > self.max_size:
            self.storage.pop(0)

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.storage, batch_size)
        obs, actions, rewards, next_obs, dones = map(np.array, zip(*batch))

        # Convert to tensors
        obs_t = torch.FloatTensor(obs).to(device)              # (B, N, S)
        actions_t = torch.LongTensor(actions).to(device)       # (B, N)
        rewards_t = torch.FloatTensor(rewards).to(device)      # (B, N)
        next_obs_t = torch.FloatTensor(next_obs).to(device)    # (B, N, S)
        dones_t = torch.FloatTensor(dones).to(device)          # (B, N)

        return {
            'obs': obs_t,
            'actions': actions_t,
            'rewards': rewards_t,
            'next_obs': next_obs_t,
            'dones': dones_t,
        }


class MADDPG:
    """A compact, runnable MADDPG implementation for discrete actions.

    - Each agent has an Actor network (policy producing action probabilities).
    - Each agent has its own centralized Critic that takes global state and all agents' one-hot actions.
    - Actions are discrete and converted to one-hot for critics.
    """

    def __init__(self, num_agents: int, state_dim: int, action_dim: int, lr: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.01, device: str = None):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.full_state_dim = num_agents * state_dim
        self.full_action_dim = num_agents * action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Actors and targets -- allow different designs per team (first half red, second half blue)
        red_hidden = (128, 64)
        blue_hidden = (256, 128)
        self.actors = []
        self.actor_targets = []
        self.actor_opts = []
        for i in range(num_agents):
            hidden = red_hidden if i < (num_agents // 2) else blue_hidden
            a = Actor(state_dim, action_dim, hidden).to(self.device)
            at = Actor(state_dim, action_dim, hidden).to(self.device)
            self.actors.append(a)
            self.actor_targets.append(at)
            self.actor_opts.append(optim.Adam(a.parameters(), lr=lr))

        # Critics and targets (one per agent, centralized)
        self.critics = [Critic(self.full_state_dim, self.full_action_dim).to(self.device) for _ in range(num_agents)]
        self.critic_targets = [Critic(self.full_state_dim, self.full_action_dim).to(self.device) for _ in range(num_agents)]
        self.critic_opts = [optim.Adam(c.parameters(), lr=lr) for c in self.critics]

        # sync targets
        for i in range(num_agents):
            self._hard_update(self.actor_targets[i], self.actors[i])
            self._hard_update(self.critic_targets[i], self.critics[i])

        self.buffer = ReplayBuffer()

    def _hard_update(self, target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.tau) + sp.data * self.tau)

    def action_probs(self, obs: np.ndarray) -> List[np.ndarray]:
        # obs: (N, S) or (S,) for one agent states list
        out = []
        for i, actor in enumerate(self.actors):
            s = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
            probs = actor(s).detach().cpu().numpy()[0]
            out.append(probs)
        return out

    def select_action(self, agent_id: int, obs: np.ndarray, explore: bool = True) -> int:
        s = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        probs = self.actors[agent_id](s).detach().cpu().numpy()[0]
        if explore:
            return np.random.choice(self.action_dim, p=probs)
        else:
            return int(np.argmax(probs))

    def _actions_to_onehot(self, actions: torch.LongTensor) -> torch.FloatTensor:
        # actions: (B, N) -> returns (B, N * A)
        B, N = actions.shape
        A = self.action_dim
        onehots = torch.zeros((B, N, A), device=self.device)
        onehots.scatter_(2, actions.unsqueeze(-1), 1.0)
        return onehots.view(B, N * A)

    def store(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray, dones: np.ndarray):
        # shapes expected: obs (N, S), actions (N,), rewards (N,), next_obs (N, S), dones (N,)
        self.buffer.add(obs, actions, rewards, next_obs, dones)

    def update(self, batch_size: int = 256):
        if len(self.buffer.storage) < batch_size:
            return

        batch = self.buffer.sample(batch_size, self.device)
        obs = batch['obs']            # (B, N, S)
        actions = batch['actions']    # (B, N)
        rewards = batch['rewards']    # (B, N)
        next_obs = batch['next_obs']  # (B, N, S)
        dones = batch['dones']        # (B, N)

        B = obs.shape[0]

        # Flatten global states for critics
        obs_flat = obs.view(B, -1)
        next_obs_flat = next_obs.view(B, -1)

        # Prepare one-hot actions
        actions_onehot = self._actions_to_onehot(actions)            # (B, N*A)

        # compute target actions (from actor_targets) for next states
        next_actions_list = []
        for i in range(self.num_agents):
            s_i = next_obs[:, i, :]
            probs = self.actor_targets[i](s_i)
            # sample deterministically with argmax for target
            a_idx = torch.argmax(probs, dim=-1, keepdim=True)
            next_actions_list.append(a_idx)
        next_actions = torch.cat(next_actions_list, dim=1)  # (B, N)
        next_actions_onehot = self._actions_to_onehot(next_actions)

        for agent_idx in range(self.num_agents):
            # Critic update for agent_idx
            critic = self.critics[agent_idx]
            critic_target = self.critic_targets[agent_idx]

            with torch.no_grad():
                q_next = critic_target(next_obs_flat, next_actions_onehot).squeeze(-1)  # (B,)
                r = rewards[:, agent_idx]
                d = dones[:, agent_idx]
                q_target = r + self.gamma * (1. - d) * q_next

            q_val = critic(obs_flat, actions_onehot).squeeze(-1)
            critic_loss = nn.MSELoss()(q_val, q_target)

            self.critic_opts[agent_idx].zero_grad()
            critic_loss.backward()
            self.critic_opts[agent_idx].step()

            # Actor update for agent_idx
            # construct actions where this agent's action is from current actor, others use sampled actions (detach)
            curr_actions_onehot = actions_onehot.clone()

            # compute actor action probabilities and pick argmax (deterministic policy gradient style)
            a_probs = self.actors[agent_idx](obs[:, agent_idx, :])  # (B, A)
            a_pred = torch.argmax(a_probs, dim=-1, keepdim=True)    # (B,1)
            # replace the agent's block in curr_actions_onehot
            onehot_pred = torch.zeros((B, self.num_agents, self.action_dim), device=self.device)
            onehot_pred.scatter_(2, a_pred.unsqueeze(-1), 1.0)
            onehot_pred_flat = onehot_pred.view(B, -1)

            # For actor loss we need combined actions where only this agent's actions are changed
            # Keep other agents' actions from the sampled batch
            other_mask = torch.ones_like(actions_onehot, device=self.device)
            start = agent_idx * self.action_dim
            end = start + self.action_dim
            # build mixed actions: take sampled actions_onehot, but replace slice [start:end] with onehot_pred_flat slice
            mixed_actions = actions_onehot.clone()
            mixed_actions[:, start:end] = onehot_pred_flat[:, start:end]

            # actor loss: maximize Q -> minimize negative Q
            actor_loss = -self.critics[agent_idx](obs_flat, mixed_actions).mean()

            self.actor_opts[agent_idx].zero_grad()
            actor_loss.backward()
            self.actor_opts[agent_idx].step()

            # soft update targets
            self._soft_update(self.actor_targets[agent_idx], self.actors[agent_idx])
            self._soft_update(self.critic_targets[agent_idx], self.critics[agent_idx])

    def save(self, path: str):
        data = {
            'actors': [a.state_dict() for a in self.actors],
            'critics': [c.state_dict() for c in self.critics],
        }
        torch.save(data, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        for a, st in zip(self.actors, data.get('actors', [])):
            a.load_state_dict(st)
        for c, st in zip(self.critics, data.get('critics', [])):
            c.load_state_dict(st)

