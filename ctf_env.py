import random
import math
from typing import Dict, Tuple, List


class CTFEnv:
    """Enhanced Capture-The-Flag environment.

    Features:
    - Configurable grid size and team sizes
    - Roles per agent (Scout, Soldier, Tank, Defender, Attacker)
    - Towers near each base that automatically attack enemies in range
    - Static obstacles (walls) and dynamic obstacles (patrol bots)
    - Health, respawn, tagging, and reward shaping
    """

    ROLE_ORDER = ['Scout', 'Soldier', 'Tank', 'Defender', 'Attacker']

    def __init__(self, grid_size: int = 15, team_size: int = 5, max_steps: int = 400):
        self.grid_size = grid_size
        self.team_size = team_size
        self.num_agents = team_size * 2
        self.max_steps = max_steps

        # Teams
        self.red_team = list(range(team_size))
        self.blue_team = list(range(team_size, 2 * team_size))

        # Bases and flags
        quarter = max(2, self.grid_size // 4)
        self.red_flag_base = (quarter, quarter)
        self.blue_flag_base = (self.grid_size - 1 - quarter, self.grid_size - 1 - quarter)

        # Towers (positioned just outside base)
        self.red_tower_pos = (self.red_flag_base[0] + 1, self.red_flag_base[1])
        self.blue_tower_pos = (self.blue_flag_base[0] - 1, self.blue_flag_base[1])
        self.tower_range = 3
        self.tower_damage = 25  # damage points per hit
        self.tower_max_hp = 100

        # Obstacles
        self.static_obstacles = set()
        self.dynamic_obstacles = []  # list of dicts {pos, dir, interval, last_moved}

        # State containers
        self.agent_positions: Dict[int, Tuple[int, int]] = {}
        self.agent_health: Dict[int, float] = {}
        self.agent_roles: Dict[int, str] = {}
        self.flag_carriers: Dict[str, int] = {}
        self.red_score = 0
        self.blue_score = 0
        self.current_step = 0

        # Tower health
        self.red_tower_hp = self.tower_max_hp
        self.blue_tower_hp = self.tower_max_hp

        # Respawn configs
        self.respawn_delay = 3
        self.respawn_timers: Dict[int, int] = {}

        # Initialize map
        self._init_map()

        # Experience/history for reward shaping
        self._last_dist_to_enemy_flag: Dict[int, float] = {}

    def _init_map(self):
        # Static obstacles: place several blocks around the map (not purely central)
        cx = self.grid_size // 2
        # horizontal barrier with gaps
        for x in range(2, self.grid_size - 2):
            if x % 5 == 0:
                continue
            y = cx - 3
            if (x, y) != self.red_flag_base and (x, y) != self.blue_flag_base:
                self.static_obstacles.add((x, y))
        # vertical barrier on the right side
        for y in range(3, self.grid_size - 3):
            x = cx + 4
            if (x, y) != self.red_flag_base and (x, y) != self.blue_flag_base:
                if y % 4 != 0:
                    self.static_obstacles.add((x, y))

        # Random small rock clusters
        rnd = random.Random(42)
        for _ in range(max(4, self.grid_size // 3)):
            rx = rnd.randint(1, self.grid_size - 2)
            ry = rnd.randint(1, self.grid_size - 2)
            self.static_obstacles.add((rx, ry))

        # Bushes: cover tiles that reduce tower damage (stealth areas)
        self.bushes = set()
        for _ in range(max(3, self.grid_size // 4)):
            bx = rnd.randint(1, self.grid_size - 2)
            by = rnd.randint(1, self.grid_size - 2)
            if (bx, by) not in self.static_obstacles and (bx, by) != self.red_flag_base and (bx, by) != self.blue_flag_base:
                self.bushes.add((bx, by))

        # Dynamic obstacles: patrol bots placed at varied rows
        for i in range(4):
            pos = (max(1, cx - 2 + i), max(1, self.grid_size // 2 - 4 + i))
            self.dynamic_obstacles.append({'pos': pos, 'dir': (1, 0), 'interval': 2 + (i % 3), 'last_moved': 0})

        # Assign roles per team in fixed order
        for idx, aid in enumerate(self.red_team):
            role = self.ROLE_ORDER[idx % len(self.ROLE_ORDER)]
            self.agent_roles[aid] = role
        for idx, aid in enumerate(self.blue_team):
            role = self.ROLE_ORDER[idx % len(self.ROLE_ORDER)]
            self.agent_roles[aid] = role

    def reset(self):
        # Spawn agents near their bases depending on role
        self.agent_positions = {}
        self.agent_health = {}
        self.respawn_timers = {}
        self.flag_carriers = {}
        self.red_score = 0
        self.blue_score = 0
        self.current_step = 0
        self.red_tower_hp = self.tower_max_hp
        self.blue_tower_hp = self.tower_max_hp

        # spawn red
        for i, aid in enumerate(self.red_team):
            rx, ry = self.red_flag_base
            # spread spawn by role
            x = rx + (i % 3)
            y = ry + (i // 3)
            self.agent_positions[aid] = (int(x), int(y))
            self.agent_health[aid] = self._max_health_for_role(self.agent_roles[aid])
            self._last_dist_to_enemy_flag[aid] = self._distance(self.agent_positions[aid], self.blue_flag_base)

        # spawn blue
        for i, aid in enumerate(self.blue_team):
            bx, by = self.blue_flag_base
            x = bx - (i % 3)
            y = by - (i // 3)
            self.agent_positions[aid] = (int(x), int(y))
            self.agent_health[aid] = self._max_health_for_role(self.agent_roles[aid])
            self._last_dist_to_enemy_flag[aid] = self._distance(self.agent_positions[aid], self.red_flag_base)

        return self.get_observation()

    def _max_health_for_role(self, role: str) -> float:
        if role == 'Scout':
            return 50.0
        if role == 'Tank':
            return 200.0
        return 100.0

    def get_observation(self):
        # Return a rich observation dict used by trainers and visualizers
        return {
            'agent_positions': self.agent_positions.copy(),
            'agent_health': self.agent_health.copy(),
            'agent_roles': self.agent_roles.copy(),
            'red_flag_position': self._get_red_flag_pos(),
            'blue_flag_position': self._get_blue_flag_pos(),
            'flag_carriers': self.flag_carriers.copy(),
            'red_score': self.red_score,
            'blue_score': self.blue_score,
            'step': self.current_step,
            'red_tower_hp': self.red_tower_hp,
            'blue_tower_hp': self.blue_tower_hp,
            'obstacles': list(self.static_obstacles),
            'dynamic_obstacles': [d['pos'] for d in self.dynamic_obstacles],
            'bushes': list(getattr(self, 'bushes', [])),
        }

    def _get_red_flag_pos(self):
        if 'red' in self.flag_carriers:
            return self.agent_positions[self.flag_carriers['red']]
        return self.red_flag_base

    def _get_blue_flag_pos(self):
        if 'blue' in self.flag_carriers:
            return self.agent_positions[self.flag_carriers['blue']]
        return self.blue_flag_base

    def step(self, actions: Dict[int, str]):
        """Step the environment.

        Args:
            actions: dict agent_id -> action in {'up','down','left','right','stay'}
        """
        rewards = {i: 0.0 for i in range(self.num_agents)}

        # Move dynamic obstacles if needed
        self._move_dynamic_obstacles()

        # Process agent actions (movement and possible collisions)
        proposed_positions = {}
        for aid, action in actions.items():
            if aid in self.respawn_timers and self.respawn_timers[aid] > 0:
                # dead, decrement timer
                self.respawn_timers[aid] -= 1
                if self.respawn_timers[aid] == 0:
                    self._respawn_agent(aid)
                continue

            if aid not in self.agent_positions:
                continue

            x, y = self.agent_positions[aid]
            nx, ny = x, y
            if action == 'up':
                ny = max(0, y - 1)
            elif action == 'down':
                ny = min(self.grid_size - 1, y + 1)
            elif action == 'left':
                nx = max(0, x - 1)
            elif action == 'right':
                nx = min(self.grid_size - 1, x + 1)
            elif action == 'stay':
                nx, ny = x, y

            # check static obstacles
            if (nx, ny) in self.static_obstacles:
                nx, ny = x, y

            # check dynamic obstacles
            if any((nx, ny) == tuple(d['pos']) for d in self.dynamic_obstacles):
                nx, ny = x, y

            proposed_positions[aid] = (nx, ny)

        # Resolve collisions: if multiple agents propose same cell, keep original positions for all
        counts = {}
        for pos in proposed_positions.values():
            counts[pos] = counts.get(pos, 0) + 1
        for aid, pos in proposed_positions.items():
            if counts[pos] > 1:
                # collision — agents stay in place
                continue
            self.agent_positions[aid] = pos

        # Tower attacks
        self._handle_tower_attacks(rewards)

        # Handle flag pickups/drops
        self._handle_flag_pickups()

        # Handle tagging (adjacent enemy collision)
        self._handle_tagging(rewards)

        # Calculate step rewards (shaping)
        step_rewards = self._calculate_rewards()
        for k, v in step_rewards.items():
            rewards[k] += v

        self.current_step += 1

        done = False
        if self.red_score > 0 or self.blue_score > 0 or self.current_step >= self.max_steps:
            done = True

        return self.get_observation(), rewards, done, {}

    def _handle_tower_attacks(self, rewards: Dict[int, float]):
        # Towers target enemies in range and deal damage
        if self.red_tower_hp > 0:
            tx, ty = self.red_tower_pos
            for aid in self.blue_team:
                if aid in self.respawn_timers and self.respawn_timers[aid] > 0:
                    continue
                pos = self.agent_positions.get(aid)
                if pos is None:
                    continue
                if self._distance(pos, (tx, ty)) <= self.tower_range:
                    # tower hits; reduce damage if agent is hidden in a bush
                    in_bush = tuple(pos) in getattr(self, 'bushes', set())
                    dmg = self.tower_damage // 2 if in_bush else self.tower_damage
                    self.agent_health[aid] -= dmg
                    rewards[aid] -= 5.0 if not in_bush else 2.5
                    if self.agent_health[aid] <= 0:
                        # eliminate agent
                        self.respawn_timers[aid] = self.respawn_delay
                        rewards[aid] -= 10.0

        if self.blue_tower_hp > 0:
            tx, ty = self.blue_tower_pos
            for aid in self.red_team:
                if aid in self.respawn_timers and self.respawn_timers[aid] > 0:
                    continue
                pos = self.agent_positions.get(aid)
                if pos is None:
                    continue
                if self._distance(pos, (tx, ty)) <= self.tower_range:
                    in_bush = tuple(pos) in getattr(self, 'bushes', set())
                    dmg = self.tower_damage // 2 if in_bush else self.tower_damage
                    self.agent_health[aid] -= dmg
                    rewards[aid] -= 5.0 if not in_bush else 2.5
                    if self.agent_health[aid] <= 0:
                        self.respawn_timers[aid] = self.respawn_delay
                        rewards[aid] -= 10.0

    def _handle_flag_pickups(self):
        # Agents can pick up enemy flag when on its base
        for aid, pos in list(self.agent_positions.items()):
            team = 'red' if aid in self.red_team else 'blue'
            opponent_base = self.blue_flag_base if team == 'red' else self.red_flag_base
            opponent_color = 'blue' if team == 'red' else 'red'
            if opponent_color not in self.flag_carriers and pos == opponent_base:
                self.flag_carriers[opponent_color] = aid

        # If carrier reaches own base -> capture
        for color in list(self.flag_carriers.keys()):
            carrier = self.flag_carriers[color]
            carrier_pos = self.agent_positions.get(carrier)
            if carrier_pos is None:
                continue
            own_base = self.red_flag_base if color == 'blue' else self.blue_flag_base
            if carrier_pos == own_base:
                # Capture
                if carrier in self.red_team:
                    self.red_score += 1
                else:
                    self.blue_score += 1
                del self.flag_carriers[color]

    def _handle_tagging(self, rewards: Dict[int, float]):
        # If opposing agents occupy same cell or adjacent, tag/respawn
        for r in self.red_team:
            for b in self.blue_team:
                if r in self.respawn_timers or b in self.respawn_timers:
                    continue
                pr = self.agent_positions.get(r)
                pb = self.agent_positions.get(b)
                if pr is None or pb is None:
                    continue
                if pr == pb or self._distance(pr, pb) < 1.5:
                    # both get penalized; attacker gets reward
                    self.respawn_timers[r] = self.respawn_delay
                    self.respawn_timers[b] = self.respawn_delay
                    rewards[r] -= 10.0
                    rewards[b] -= 10.0

    def _move_dynamic_obstacles(self):
        for d in self.dynamic_obstacles:
            if (self.current_step - d['last_moved']) >= d['interval']:
                x, y = d['pos']
                dx, dy = d['dir']
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and (nx, ny) not in self.static_obstacles:
                    d['pos'] = (nx, ny)
                else:
                    # reverse direction
                    d['dir'] = (-dx, -dy)
                d['last_moved'] = self.current_step

    def _respawn_agent(self, aid: int):
        # place agent back at its base spawn and restore health
        if aid in self.red_team:
            base = self.red_flag_base
        else:
            base = self.blue_flag_base
        x, y = base
        # small random offset
        x = min(self.grid_size - 1, max(0, x + random.randint(-1, 1)))
        y = min(self.grid_size - 1, max(0, y + random.randint(-1, 1)))
        self.agent_positions[aid] = (x, y)
        self.agent_health[aid] = self._max_health_for_role(self.agent_roles[aid])

    def _calculate_rewards(self) -> Dict[int, float]:
        rewards = {i: -1.0 for i in range(self.num_agents)}  # step penalty

        # Shaping: distance to enemy flag
        for aid in range(self.num_agents):
            if aid in self.respawn_timers and self.respawn_timers[aid] > 0:
                continue
            pos = self.agent_positions[aid]
            team = 'red' if aid in self.red_team else 'blue'
            enemy_flag = self.blue_flag_base if team == 'red' else self.red_flag_base
            last = self._last_dist_to_enemy_flag.get(aid, None)
            dist = self._distance(pos, enemy_flag)
            if last is not None and dist < last - 0.5:
                # moved significantly closer
                rewards[aid] += 10.0
            self._last_dist_to_enemy_flag[aid] = dist

        # Team coordination: if >=2 teammates near enemy flag, give small bonus
        for team_ids in (self.red_team, self.blue_team):
            for aid in team_ids:
                pos = self.agent_positions.get(aid)
                if pos is None:
                    continue
                enemy_flag = self.blue_flag_base if aid in self.red_team else self.red_flag_base
                near_count = sum(1 for t in team_ids if self._distance(self.agent_positions.get(t, (-999, -999)), enemy_flag) < 3)
                if near_count >= 2:
                    rewards[aid] += 5.0

        # Defending own flag: small bonus if near own base and no enemy flag carrier
        for aid in range(self.num_agents):
            team = 'red' if aid in self.red_team else 'blue'
            own_base = self.red_flag_base if team == 'red' else self.blue_flag_base
            if self._distance(self.agent_positions[aid], own_base) < 3:
                rewards[aid] += 15.0

        return rewards

    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
