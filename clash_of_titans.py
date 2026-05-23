"""
Clash of Titans: Advanced multi-agent game mode.

Combines:
- Tower Defense: Defend home tower, attack enemy tower
- Relay Race: Escort powerful units through checkpoints
- Team Deathmatch: Score by eliminating/tagging opponents
- Role Specialization: Defenders vs Attackers

Game Flow:
1. Towers spawn at team bases with health
2. Attackers escort "titans" (powerful units) toward enemy tower
3. Defenders protect own tower and intercept attackers
4. Defeating enemies scores points (proximity-based)
5. Tower destruction scores massive points
6. First team to objective wins
"""

import random
import numpy as np


class Titan:
    """Powerful escort unit that teams use to attack towers."""
    
    def __init__(self, team_color, pos, health=20):
        self.team_color = team_color  # 'red' or 'blue'
        self.pos = pos
        self.health = health
        self.max_health = health
        self.escort_agent = None  # Which agent is escorting
        self.progress = 0  # Progress toward enemy tower (0-100)


class ClashOfTitansEnv:
    """
    6-agent game: Red team (3) vs Blue team (3)
    
    Features:
    - Two towers with health
    - Titans that must be escorted to towers
    - Proximity-based damage (deathmatch mechanics)
    - Checkpoints for relay progression
    - Multiple victory conditions
    """
    
    def __init__(self, grid_size=20, team_size=3, max_steps=500):
        self.grid_size = grid_size
        self.team_size = team_size
        self.num_agents = team_size * 2
        self.max_steps = max_steps
        
        # Teams
        self.red_team = list(range(team_size))
        self.blue_team = list(range(team_size, team_size * 2))
        
        # Tower bases and health
        self.red_tower_base = (2, 2)
        self.blue_tower_base = (grid_size - 3, grid_size - 3)
        self.red_tower_health = 100
        self.blue_tower_health = 100
        self.tower_max_health = 100
        
        # Checkpoints for titan progression (relay-style)
        self.red_checkpoints = [
            (grid_size // 4, grid_size // 4),
            (grid_size // 2, grid_size // 2),
            (3 * grid_size // 4, 3 * grid_size // 4),
        ]
        self.blue_checkpoints = [
            (3 * grid_size // 4, 3 * grid_size // 4),
            (grid_size // 2, grid_size // 2),
            (grid_size // 4, grid_size // 4),
        ]
        
        # Titans (escorts units)
        self.red_titan = None
        self.blue_titan = None
        self.titan_speed = 1  # Moves 1 unit per successful escort
        self.titan_proximity_radius = 3  # Must be within this distance to count as escort
        
        # Game state
        self.agent_positions = {}
        self.red_score = 0
        self.blue_score = 0
        self.current_step = 0
        self.red_titan_checkpoint = 0
        self.blue_titan_checkpoint = 0
        
        # Combat mechanics
        self.attack_radius = 1.5
        self.attack_cooldown = {}  # Per agent cooldown
        
        self.reset()
    
    def reset(self):
        """Reset game to initial state."""
        # Spawn agents near bases
        self.agent_positions = {}
        for agent_id in self.red_team:
            x = self.red_tower_base[0] + random.randint(-2, 2)
            y = self.red_tower_base[1] + random.randint(-2, 2)
            self.agent_positions[agent_id] = (
                max(0, min(self.grid_size - 1, x)),
                max(0, min(self.grid_size - 1, y))
            )
        
        for agent_id in self.blue_team:
            x = self.blue_tower_base[0] + random.randint(-2, 2)
            y = self.blue_tower_base[1] + random.randint(-2, 2)
            self.agent_positions[agent_id] = (
                max(0, min(self.grid_size - 1, x)),
                max(0, min(self.grid_size - 1, y))
            )
        
        # Initialize towers
        self.red_tower_health = self.tower_max_health
        self.blue_tower_health = self.tower_max_health
        
        # Spawn titans at red and blue bases
        self.red_titan = Titan('red', self.red_tower_base, health=30)
        self.blue_titan = Titan('blue', self.blue_tower_base, health=30)
        
        # Reset game state
        self.red_score = 0
        self.blue_score = 0
        self.current_step = 0
        self.red_titan_checkpoint = 0
        self.blue_titan_checkpoint = 0
        self.attack_cooldown = {i: 0 for i in range(self.num_agents)}
        
        return self.get_observation()
    
    def get_observation(self):
        """Get current game state."""
        return {
            'agent_positions': self.agent_positions.copy(),
            'red_tower_pos': self.red_tower_base,
            'blue_tower_pos': self.blue_tower_base,
            'red_tower_health': self.red_tower_health,
            'blue_tower_health': self.blue_tower_health,
            'red_titan_pos': self.red_titan.pos if self.red_titan else None,
            'blue_titan_pos': self.blue_titan.pos if self.blue_titan else None,
            'red_titan_health': self.red_titan.health if self.red_titan else 0,
            'blue_titan_health': self.blue_titan.health if self.blue_titan else 0,
            'red_checkpoint': self.red_titan_checkpoint,
            'blue_checkpoint': self.blue_titan_checkpoint,
            'red_score': self.red_score,
            'blue_score': self.blue_score,
            'step': self.current_step,
        }
    
    def _distance(self, pos1, pos2):
        """Euclidean distance."""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
    
    def _get_team(self, agent_id):
        """Get team color."""
        return 'red' if agent_id in self.red_team else 'blue'
    
    def step(self, actions):
        """
        Execute one game step.
        
        Args:
            actions: {agent_id: 'up'/'down'/'left'/'right'}
        
        Returns:
            obs, rewards, done, info
        """
        # Update agent positions
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
        
        # Decrease attack cooldowns
        for agent_id in self.attack_cooldown:
            self.attack_cooldown[agent_id] = max(0, self.attack_cooldown[agent_id] - 1)
        
        # Handle titan mechanics (escort, movement, health)
        self._update_titans()
        
        # Handle combat (proximity-based damage)
        rewards = self._handle_combat()
        
        # Check victory conditions
        done = False
        if self.red_tower_health <= 0:
            self.blue_score += 50  # Bonus for tower destruction
            done = True
        elif self.blue_tower_health <= 0:
            self.red_score += 50
            done = True
        elif self.red_score >= 100 or self.blue_score >= 100:
            done = True
        elif self.current_step >= self.max_steps:
            done = True
        
        self.current_step += 1
        
        return self.get_observation(), rewards, done, {}
    
    def _update_titans(self):
        """Handle titan escort, movement, and health."""
        # Red titan escort and movement
        if self.red_titan:
            red_escorts = [
                a for a in self.red_team
                if self._distance(self.agent_positions[a], self.red_titan.pos) < self.titan_proximity_radius
            ]
            
            if red_escorts:
                # Advance toward checkpoint
                target = self.blue_checkpoints[self.red_titan_checkpoint]
                self._move_toward(self.red_titan, target)
                
                # Check if reached checkpoint
                if self._distance(self.red_titan.pos, target) < 1.5:
                    self.red_titan_checkpoint += 1
                    if self.red_titan_checkpoint >= len(self.blue_checkpoints):
                        # Reached tower!
                        self.blue_tower_health -= 20
                        self.red_score += 30
                        self.red_titan_checkpoint = 0
                        self.red_titan.pos = self.red_tower_base
                        self.red_titan.health = 30
        
        # Blue titan escort and movement
        if self.blue_titan:
            blue_escorts = [
                a for a in self.blue_team
                if self._distance(self.agent_positions[a], self.blue_titan.pos) < self.titan_proximity_radius
            ]
            
            if blue_escorts:
                # Advance toward checkpoint
                target = self.red_checkpoints[self.blue_titan_checkpoint]
                self._move_toward(self.blue_titan, target)
                
                # Check if reached checkpoint
                if self._distance(self.blue_titan.pos, target) < 1.5:
                    self.blue_titan_checkpoint += 1
                    if self.blue_titan_checkpoint >= len(self.red_checkpoints):
                        # Reached tower!
                        self.red_tower_health -= 20
                        self.blue_score += 30
                        self.blue_titan_checkpoint = 0
                        self.blue_titan.pos = self.blue_tower_base
                        self.blue_titan.health = 30
        
        # Titans take damage from nearby enemies
        if self.red_titan:
            for enemy_id in self.blue_team:
                if self._distance(self.agent_positions[enemy_id], self.red_titan.pos) < 2:
                    self.red_titan.health -= 0.5
        
        if self.blue_titan:
            for enemy_id in self.red_team:
                if self._distance(self.agent_positions[enemy_id], self.blue_titan.pos) < 2:
                    self.blue_titan.health -= 0.5
    
    def _move_toward(self, obj, target_pos, speed=1):
        """Move object toward target."""
        if obj.pos == target_pos:
            return
        
        dx = target_pos[0] - obj.pos[0]
        dy = target_pos[1] - obj.pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 0:
            dx, dy = dx / dist, dy / dist
            new_x = obj.pos[0] + dx * speed
            new_y = obj.pos[1] + dy * speed
            obj.pos = (new_x, new_y)
    
    def _handle_combat(self):
        """Handle proximity-based combat between teams."""
        rewards = {i: 0.0 for i in range(self.num_agents)}
        
        # Check for proximity combat (reward for staying near opponents)
        for red_id in self.red_team:
            for blue_id in self.blue_team:
                red_pos = self.agent_positions[red_id]
                blue_pos = self.agent_positions[blue_id]
                dist = self._distance(red_pos, blue_pos)
                
                # Proximity-based reward (being near enemy team)
                if dist < 2:
                    rewards[red_id] += 0.5  # Red gets reward for harassment
                    rewards[blue_id] += 0.5  # Blue gets reward for standing ground
        
        return rewards

