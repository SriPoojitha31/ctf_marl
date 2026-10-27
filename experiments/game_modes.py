"""
Examples of different CTF game modes and how to create them.

Shows how to customize the CTF environment for different competitive/cooperative scenarios.
"""

import numpy as np
from envs.ctf_env import CTFEnv
from envs.moba_ctf_env import MobaCTFEnv


# ===== MODE 1: ASYMMETRIC TEAMS =====

class AsymmetricCTFEnv(CTFEnv):
    """
    Red team: 2 agents (elite squad)
    Blue team: 4 agents (militia)
    
    Red must overcome numbers with better performance.
    """
    def __init__(self):
        super().__init__(grid_size=20, team_size=2, max_steps=300)
        self.red_team = [0, 1]      # 2 red agents
        self.blue_team = [2, 3, 4, 5]  # 4 blue agents
        self.num_agents = 6


# ===== MODE 2: KING OF THE HILL =====

class KingOfTheHillMode(CTFEnv):
    """
    Both teams must hold a central control point.
    Controlling team scores 1 point per step.
    First to 100 points wins.
    """
    def __init__(self):
        super().__init__(grid_size=15, team_size=3, max_steps=1000)
        self.control_point = (self.grid_size // 2, self.grid_size // 2)
        self.control_radius = 2
        self.red_control_score = 0
        self.blue_control_score = 0
    
    def step(self, actions):
        obs, rewards, done, info = super().step(actions)
        
        # Check who controls center
        red_near = sum(1 for a in self.red_team 
                      if self._distance(self.agent_positions[a], self.control_point) < self.control_radius)
        blue_near = sum(1 for a in self.blue_team 
                       if self._distance(self.agent_positions[a], self.control_point) < self.control_radius)
        
        # Award control points
        if red_near > blue_near:
            self.red_control_score += 1
            for a in self.red_team:
                rewards[a] += 0.1  # Control bonus
        elif blue_near > red_near:
            self.blue_control_score += 1
            for a in self.blue_team:
                rewards[a] += 0.1
        
        # Win condition: first to 100 points
        if self.red_control_score >= 100:
            self.red_score = 1
            done = True
        elif self.blue_control_score >= 100:
            self.blue_score = 1
            done = True
        
        return obs, rewards, done, info


# ===== MODE 3: TEAM DEATHMATCH =====

class TeamDeathmatchMode(CTFEnv):
    """
    Instead of capturing flags:
    - Teams score by getting near opponent agents
    - Agents learn to cluster vs scatter
    - First team with 20 points wins
    """
    def __init__(self):
        super().__init__(grid_size=15, team_size=3, max_steps=300)
        self.red_deathmatch_score = 0
        self.blue_deathmatch_score = 0
    
    def step(self, actions):
        # Update positions
        for agent_id, action in actions.items():
            if agent_id not in self.agent_positions:
                continue
            x, y = self.agent_positions[agent_id]
            if action == 'up':
                y = max(0, y - 1)
            elif action == 'down':
                y = min(self.grid_size - 1, y + 1)
            elif action == 'left':
                x = max(0, x - 1)
            elif action == 'right':
                x = min(self.grid_size - 1, x + 1)
            self.agent_positions[agent_id] = (x, y)
        
        rewards = {i: 0.0 for i in range(self.num_agents)}
        
        # Scoring: get near opponents
        for red_id in self.red_team:
            for blue_id in self.blue_team:
                dist = self._distance(
                    self.agent_positions[red_id],
                    self.agent_positions[blue_id]
                )
                if dist < 1.5:
                    # Red scored on blue
                    self.red_deathmatch_score += 1
                    rewards[red_id] += 1.0
        
        self.current_step += 1
        done = (self.red_deathmatch_score >= 20 or 
                self.blue_deathmatch_score >= 20 or 
                self.current_step >= self.max_steps)
        
        if self.red_deathmatch_score >= 20:
            self.red_score = 1
        elif self.blue_deathmatch_score >= 20:
            self.blue_score = 1
        
        return self.get_observation(), rewards, done, {}


# ===== MODE 4: RELAY RACE =====

class RelayRaceMode(CTFEnv):
    """
    Agents must pass a baton in relay fashion:
    - Agent A carries flag to point 1
    - Agent B picks it up and carries to point 2
    - Agent C brings it home
    
    Encourages coordination and relay strategies.
    """
    def __init__(self):
        super().__init__(grid_size=15, team_size=3, max_steps=400)
        
        # Checkpoint locations
        self.checkpoints = [
            (self.grid_size // 4, self.grid_size // 4),      # checkpoint 1
            (3 * self.grid_size // 4, self.grid_size // 4),  # checkpoint 2
            (3 * self.grid_size // 4, 3 * self.grid_size // 4),  # checkpoint 3
        ]
        self.current_checkpoint = 0
        self.flag_location = self.red_flag_base
    
    def step(self, actions):
        # Basic movement
        for agent_id, action in actions.items():
            x, y = self.agent_positions[agent_id]
            if action == 'up':
                y = max(0, y - 1)
            elif action == 'down':
                y = min(self.grid_size - 1, y + 1)
            elif action == 'left':
                x = max(0, x - 1)
            elif action == 'right':
                x = min(self.grid_size - 1, x + 1)
            self.agent_positions[agent_id] = (x, y)
        
        rewards = {i: 0.0 for i in range(self.num_agents)}
        
        # Check if any agent reaches current checkpoint
        for agent_id in self.red_team:
            if self._distance(self.agent_positions[agent_id], 
                            self.checkpoints[self.current_checkpoint]) < 1.5:
                rewards[agent_id] += 5.0
                self.current_checkpoint += 1
                
                if self.current_checkpoint >= len(self.checkpoints):
                    # Reached end!
                    self.red_score = 1
                    rewards[agent_id] += 10.0
        
        self.current_step += 1
        done = self.red_score > 0 or self.current_step >= self.max_steps
        
        return self.get_observation(), rewards, done, {}


# ===== MODE 5: TOWER DEFENSE =====

class TowerDefenseMode(CTFEnv):
    """
    Red team (defenders) protects central tower.
    Blue team (attackers) tries to reach and capture tower.
    Roles are asymmetric and fixed.
    """
    def __init__(self):
        super().__init__(grid_size=15, team_size=3, max_steps=200)
        self.tower = (self.grid_size // 2, self.grid_size // 2)
        self.tower_health = 10
    
    def step(self, actions):
        # Movement
        for agent_id, action in actions.items():
            x, y = self.agent_positions[agent_id]
            if action == 'up':
                y = max(0, y - 1)
            elif action == 'down':
                y = min(self.grid_size - 1, y + 1)
            elif action == 'left':
                x = max(0, x - 1)
            elif action == 'right':
                x = min(self.grid_size - 1, x + 1)
            self.agent_positions[agent_id] = (x, y)
        
        rewards = {i: 0.0 for i in range(self.num_agents)}
        
        # Blue team attacks tower (if near it)
        for blue_id in self.blue_team:
            if self._distance(self.agent_positions[blue_id], self.tower) < 2:
                self.tower_health -= 0.5
                rewards[blue_id] += 0.5
        
        # Red team defends (bonus for being near tower)
        for red_id in self.red_team:
            if self._distance(self.agent_positions[red_id], self.tower) < 3:
                rewards[red_id] += 0.2
                # Also defend tower slightly
                self.tower_health = min(10, self.tower_health + 0.1)
        
        self.current_step += 1
        
        done = False
        if self.tower_health <= 0:
            self.blue_score = 1
            done = True
        elif self.tower_health >= 10 and self.current_step > 50:
            self.red_score = 1
            done = True
        elif self.current_step >= self.max_steps:
            # Defender wins if tower survives
            self.red_score = 1
            done = True

        return self.get_observation(), rewards, done, {}


# ===== MODE 6: STANDARD 5v5 MOBA =====

class StandardMobaMode(MobaCTFEnv):
    """
    Simplified 5v5 lane-based MOBA mode.

    Features:
    - 3 lanes (top, mid, bot)
    - 2 towers per lane per team (outer + inner)
    - Team roles: Tank, Fighter, Assassin, Marksman, Support
    - Health, combat, tower aggro, core destruction, respawn
    """

    def __init__(self):
        super().__init__(grid_size=15, team_size=5, max_steps=300)


# ===== UTILITY FUNCTION =====

def create_game_mode(mode='standard'):
    """
    Factory function to create different game modes.
    
    Args:
        mode: 'standard', 'asymmetric', 'king_of_hill', 'deathmatch', 'relay', 'tower'
    
    Returns:
        Environment instance
    """
    modes = {
        'standard': StandardMobaMode,
        'asymmetric': AsymmetricCTFEnv,
        'king_of_hill': KingOfTheHillMode,
        'deathmatch': TeamDeathmatchMode,
        'relay': RelayRaceMode,
        'tower': TowerDefenseMode,
    }
    
    if mode not in modes:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(modes.keys())}")
    
    return modes[mode]()


if __name__ == "__main__":
    print("CTF Game Mode Examples")
    print("=" * 60)
    
    # Demo each mode
    for mode_name in ['standard', 'asymmetric', 'king_of_hill', 'deathmatch', 'relay', 'tower']:
        print(f"\n{mode_name.upper()} MODE:")
        env = create_game_mode(mode_name)
        obs = env.reset()
        
        # Run 1 step
        actions = {i: 'up' for i in range(env.num_agents)}
        obs, rewards, done, _ = env.step(actions)
        
        print(f"  - Grid size: {env.grid_size}")
        print(f"  - Agents: {env.num_agents}")
        print(f"  - Max steps: {env.max_steps}")
        print(f"  - Sample reward: {rewards[0]:.2f}")
        print(f"  - Done: {done}")
