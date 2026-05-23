import random
from typing import Any, Tuple

import numpy as np


class IQLAgent:
    """Tabular IQL agent that supports both positional and vector observations."""

    def __init__(
        self,
        grid_size: int = 15,
        state_size: int = None,
        action_dim: int = 4,
        action_size: int = None,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
    ):
        self.grid_size = grid_size
        self.state_size = state_size
        self.action_size = action_size if action_size is not None else action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def _coarse_flag_state(self, agent_pos: Tuple[int, int], enemy_flag_pos: Tuple[int, int]):
        ax, ay = agent_pos
        fx, fy = enemy_flag_pos
        bucket = max(1, self.grid_size // 3)
        return (ax // bucket, ay // bucket, fx // bucket, fy // bucket)

    def _vector_state_key(self, state: Any):
        return tuple(np.asarray(state, dtype=np.float32).flatten())

    def _ensure_state(self, key):
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[key]

    def get_action(self, agent_pos, enemy_flag_pos):
        state_key = self._coarse_flag_state(agent_pos, enemy_flag_pos)
        q_values = self._ensure_state(state_key)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return int(np.argmax(q_values))

    def choose_action(self, state):
        state_key = self._vector_state_key(state)
        q_values = self._ensure_state(state_key)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return int(np.argmax(q_values))

    def update(self, *args):
        # Positional API: (agent_pos, enemy_flag_pos, action, reward, next_agent_pos, next_enemy_flag_pos, done)
        if len(args) == 7:
            agent_pos, enemy_flag_pos, action, reward, next_agent_pos, next_enemy_flag_pos, done = args
            state_key = self._coarse_flag_state(agent_pos, enemy_flag_pos)
            next_key = self._coarse_flag_state(next_agent_pos, next_enemy_flag_pos)
        # Vector API: (state, action, reward, next_state, done)
        elif len(args) == 5:
            state, action, reward, next_state, done = args
            state_key = self._vector_state_key(state)
            next_key = self._vector_state_key(next_state)
        else:
            raise TypeError("IQLAgent.update expects either 5 or 7 positional arguments.")

        q_values = self._ensure_state(state_key)
        next_q_values = self._ensure_state(next_key)
        target = reward
        if not done:
            target += self.gamma * float(np.max(next_q_values))
        q_values[action] += self.lr * (target - q_values[action])

    def decay_epsilon(self):
        self.epsilon = max(0.05, self.epsilon * 0.995)

    def reset(self):
        pass
