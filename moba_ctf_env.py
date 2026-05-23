import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Tower:
    team: str
    lane: str
    tier: str
    pos: Tuple[int, int]
    hp: float
    max_hp: float
    attack_range: float
    attack_damage: float

    @property
    def alive(self) -> bool:
        return self.hp > 0


class MobaCTFEnv:
    """Simplified 5v5 MOBA-style environment for MARL experiments."""

    ROLE_ORDER = ["Tank", "Fighter", "Assassin", "Marksman", "Support"]
    ROLE_STATS = {
        "Tank": {"max_hp": 180.0, "attack_damage": 18.0, "attack_range": 1.5, "cooldown": 1},
        "Fighter": {"max_hp": 130.0, "attack_damage": 22.0, "attack_range": 1.5, "cooldown": 1},
        "Assassin": {"max_hp": 95.0, "attack_damage": 34.0, "attack_range": 1.5, "cooldown": 2},
        "Marksman": {"max_hp": 90.0, "attack_damage": 24.0, "attack_range": 3.0, "cooldown": 1},
        "Support": {"max_hp": 110.0, "attack_damage": 14.0, "attack_range": 2.0, "cooldown": 1},
    }

    def __init__(self, grid_size: int = 15, team_size: int = 5, max_steps: int = 300):
        if team_size != 5:
            raise ValueError("MobaCTFEnv is designed for 5v5 play, so team_size must be 5.")

        self.grid_size = grid_size
        self.team_size = team_size
        self.num_agents = team_size * 2
        self.max_steps = max_steps

        self.red_team = list(range(team_size))
        self.blue_team = list(range(team_size, self.num_agents))
        self.agent_roles = {aid: self.ROLE_ORDER[idx % len(self.ROLE_ORDER)] for idx, aid in enumerate(range(self.num_agents))}

        self.lane_rows = {
            "top": max(1, self.grid_size // 4),
            "mid": self.grid_size // 2,
            "bot": min(self.grid_size - 2, self.grid_size - 1 - self.grid_size // 4),
        }
        self.red_base = (1, self.grid_size - 2)
        self.blue_base = (self.grid_size - 2, 1)
        self.red_core_pos = self.red_base
        self.blue_core_pos = self.blue_base
        self.core_max_hp = 250.0

        self.tower_max_hp = 120.0
        self.tower_attack_range = 3.5
        self.tower_attack_damage = 26.0
        self.objective_damage = 20.0
        self.respawn_delay = 3
        self.heal_amount = 10.0
        self.heal_range = 2.0

        self.static_obstacles = set()
        self.dynamic_obstacles: List[Dict[str, object]] = []

        self.agent_positions: Dict[int, Tuple[int, int]] = {}
        self.agent_health: Dict[int, float] = {}
        self.respawn_timers: Dict[int, int] = {}
        self.attack_cooldowns: Dict[int, int] = {}
        self.last_positions: Dict[int, Tuple[int, int]] = {}
        self.last_dist_to_goal: Dict[int, float] = {}
        self.red_score = 0
        self.blue_score = 0
        self.current_step = 0
        self.red_core_hp = self.core_max_hp
        self.blue_core_hp = self.core_max_hp
        self.towers: List[Tower] = []

        self._build_map()
        self.reset()

    def _build_map(self):
        self.static_obstacles = set()
        self.dynamic_obstacles = []

        lane_rows = set(self.lane_rows.values())
        jungle_columns = [self.grid_size // 3, (2 * self.grid_size) // 3]
        for x in jungle_columns:
            for y in range(1, self.grid_size - 1):
                if y in lane_rows:
                    continue
                if (x + y) % 3 != 0:
                    self.static_obstacles.add((x, y))

        jungle_rows = [self.grid_size // 3, (2 * self.grid_size) // 3]
        for y in jungle_rows:
            for x in range(1, self.grid_size - 1):
                if y in lane_rows:
                    continue
                if x in jungle_columns:
                    continue
                if (x * y) % 5 == 0:
                    self.static_obstacles.add((x, y))

        protected_tiles = {self.red_base, self.blue_base, self.red_core_pos, self.blue_core_pos}
        for lane_y in self.lane_rows.values():
            for x in range(self.grid_size):
                protected_tiles.add((x, lane_y))
        self.static_obstacles.difference_update(protected_tiles)

        self.dynamic_obstacles = [
            {"pos": (self.grid_size // 2 - 1, self.lane_rows["top"] + 1), "dir": (0, 1), "interval": 2, "last_moved": 0},
            {"pos": (self.grid_size // 2, self.lane_rows["bot"] - 1), "dir": (0, -1), "interval": 3, "last_moved": 0},
        ]

    def reset(self):
        self.agent_positions = {}
        self.agent_health = {}
        self.respawn_timers = {}
        self.attack_cooldowns = {aid: 0 for aid in range(self.num_agents)}
        self.red_score = 0
        self.blue_score = 0
        self.current_step = 0
        self.red_core_hp = self.core_max_hp
        self.blue_core_hp = self.core_max_hp
        self._init_towers()

        red_spawn_points = [
            (self.red_base[0], self.lane_rows["top"]),
            (self.red_base[0], self.lane_rows["mid"]),
            (self.red_base[0], self.lane_rows["bot"]),
            (self.red_base[0] + 1, min(self.grid_size - 1, self.lane_rows["top"] + 1)),
            (self.red_base[0] + 1, max(0, self.lane_rows["bot"] - 1)),
        ]
        blue_spawn_points = [
            (self.blue_base[0], self.lane_rows["top"]),
            (self.blue_base[0], self.lane_rows["mid"]),
            (self.blue_base[0], self.lane_rows["bot"]),
            (self.blue_base[0] - 1, min(self.grid_size - 1, self.lane_rows["top"] + 1)),
            (self.blue_base[0] - 1, max(0, self.lane_rows["bot"] - 1)),
        ]

        for idx, aid in enumerate(self.red_team):
            pos = red_spawn_points[idx]
            self.agent_positions[aid] = pos
            self.agent_health[aid] = self._max_health_for_agent(aid)

        for idx, aid in enumerate(self.blue_team):
            pos = blue_spawn_points[idx]
            self.agent_positions[aid] = pos
            self.agent_health[aid] = self._max_health_for_agent(aid)

        self.last_positions = self.agent_positions.copy()
        self.last_dist_to_goal = {aid: self._distance(self.agent_positions[aid], self._enemy_core_for_agent(aid)) for aid in range(self.num_agents)}
        return self.get_observation()

    def _init_towers(self):
        self.towers = []
        for lane, lane_y in self.lane_rows.items():
            self.towers.append(Tower("red", lane, "outer", (3, lane_y), self.tower_max_hp, self.tower_max_hp, self.tower_attack_range, self.tower_attack_damage))
            self.towers.append(Tower("red", lane, "inner", (1, lane_y), self.tower_max_hp, self.tower_max_hp, self.tower_attack_range, self.tower_attack_damage))
            self.towers.append(Tower("blue", lane, "outer", (self.grid_size - 4, lane_y), self.tower_max_hp, self.tower_max_hp, self.tower_attack_range, self.tower_attack_damage))
            self.towers.append(Tower("blue", lane, "inner", (self.grid_size - 2, lane_y), self.tower_max_hp, self.tower_max_hp, self.tower_attack_range, self.tower_attack_damage))

    def get_observation(self):
        return {
            "agent_positions": self.agent_positions.copy(),
            "agent_health": self.agent_health.copy(),
            "agent_roles": self.agent_roles.copy(),
            "respawn_timers": self.respawn_timers.copy(),
            "red_team": list(self.red_team),
            "blue_team": list(self.blue_team),
            "red_core_pos": self.red_core_pos,
            "blue_core_pos": self.blue_core_pos,
            "red_core_hp": self.red_core_hp,
            "blue_core_hp": self.blue_core_hp,
            "red_score": self.red_score,
            "blue_score": self.blue_score,
            "step": self.current_step,
            "lanes": self.lane_rows.copy(),
            "towers": [
                {
                    "team": tower.team,
                    "lane": tower.lane,
                    "tier": tower.tier,
                    "pos": tower.pos,
                    "hp": tower.hp,
                    "max_hp": tower.max_hp,
                }
                for tower in self.towers
            ],
            "obstacles": list(self.static_obstacles),
            "dynamic_obstacles": [tuple(d["pos"]) for d in self.dynamic_obstacles],
            "red_flag_position": self.red_core_pos,
            "blue_flag_position": self.blue_core_pos,
        }

    def step(self, actions: Dict[int, str]):
        rewards = {aid: -1.0 for aid in range(self.num_agents)}
        self.last_positions = self.agent_positions.copy()

        self._tick_respawns()
        self._move_dynamic_obstacles()
        self._move_agents(actions)
        self._tick_cooldowns()
        self._handle_agent_combat(rewards)
        self._handle_support_actions(rewards)
        self._handle_tower_attacks(rewards)
        self._handle_objective_damage(rewards)
        self._apply_shaping_rewards(rewards, actions)

        self.current_step += 1
        done = self.red_core_hp <= 0 or self.blue_core_hp <= 0 or self.current_step >= self.max_steps
        if self.red_core_hp <= 0:
            self.blue_score += 1
        elif self.blue_core_hp <= 0:
            self.red_score += 1
        return self.get_observation(), rewards, done, {}

    def _tick_respawns(self):
        for aid in list(self.respawn_timers.keys()):
            if self.respawn_timers[aid] > 0:
                self.respawn_timers[aid] -= 1
            if self.respawn_timers[aid] <= 0:
                self._respawn_agent(aid)
                del self.respawn_timers[aid]

    def _move_dynamic_obstacles(self):
        for obstacle in self.dynamic_obstacles:
            if (self.current_step - obstacle["last_moved"]) < obstacle["interval"]:
                continue
            x, y = obstacle["pos"]
            dx, dy = obstacle["dir"]
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and (nx, ny) not in self.static_obstacles:
                obstacle["pos"] = (nx, ny)
            else:
                obstacle["dir"] = (-dx, -dy)
            obstacle["last_moved"] = self.current_step

    def _move_agents(self, actions: Dict[int, str]):
        proposed_positions = {}
        for aid in range(self.num_agents):
            if aid in self.respawn_timers:
                continue
            x, y = self.agent_positions[aid]
            action = actions.get(aid, "stay")
            nx, ny = x, y
            if action == "up":
                ny = max(0, y - 1)
            elif action == "down":
                ny = min(self.grid_size - 1, y + 1)
            elif action == "left":
                nx = max(0, x - 1)
            elif action == "right":
                nx = min(self.grid_size - 1, x + 1)

            if (nx, ny) in self.static_obstacles:
                nx, ny = x, y
            if any(tuple(ob["pos"]) == (nx, ny) for ob in self.dynamic_obstacles):
                nx, ny = x, y
            proposed_positions[aid] = (nx, ny)

        occupancy = {}
        for pos in proposed_positions.values():
            occupancy[pos] = occupancy.get(pos, 0) + 1

        for aid, pos in proposed_positions.items():
            if occupancy[pos] == 1:
                self.agent_positions[aid] = pos

    def _tick_cooldowns(self):
        for aid in self.attack_cooldowns:
            self.attack_cooldowns[aid] = max(0, self.attack_cooldowns[aid] - 1)

    def _handle_agent_combat(self, rewards: Dict[int, float]):
        for aid in range(self.num_agents):
            if aid in self.respawn_timers or self.attack_cooldowns[aid] > 0:
                continue
            target = self._nearest_enemy_in_range(aid)
            if target is None:
                continue

            role = self.agent_roles[aid]
            damage = self.ROLE_STATS[role]["attack_damage"]
            self.agent_health[target] -= damage
            rewards[aid] += 4.0
            self.attack_cooldowns[aid] = self.ROLE_STATS[role]["cooldown"]

            if self.agent_health[target] <= 0:
                rewards[aid] += 10.0
                rewards[target] -= 20.0
                self.respawn_timers[target] = self.respawn_delay
                self._drop_to_spawn(target)
                for teammate in self._allied_agents(aid):
                    if teammate == aid or teammate in self.respawn_timers:
                        continue
                    if self._distance(self.agent_positions[teammate], self.agent_positions[aid]) <= 2.5:
                        rewards[teammate] += 5.0

    def _handle_support_actions(self, rewards: Dict[int, float]):
        for aid in range(self.num_agents):
            if aid in self.respawn_timers or self.agent_roles[aid] != "Support":
                continue
            for teammate in self._allied_agents(aid):
                if teammate == aid or teammate in self.respawn_timers:
                    continue
                if self._distance(self.agent_positions[aid], self.agent_positions[teammate]) <= self.heal_range:
                    max_hp = self._max_health_for_agent(teammate)
                    healed = min(self.heal_amount, max_hp - self.agent_health[teammate])
                    if healed > 0:
                        self.agent_health[teammate] += healed
                        rewards[aid] += 2.0

    def _handle_tower_attacks(self, rewards: Dict[int, float]):
        for tower in self.towers:
            if not tower.alive:
                continue
            enemy_team = self.blue_team if tower.team == "red" else self.red_team
            candidates = []
            for aid in enemy_team:
                if aid in self.respawn_timers:
                    continue
                dist = self._distance(self.agent_positions[aid], tower.pos)
                if dist <= tower.attack_range:
                    candidates.append((dist, aid))
            if not candidates:
                continue
            _, target = min(candidates, key=lambda item: item[0])
            self.agent_health[target] -= tower.attack_damage
            rewards[target] -= 10.0
            if self.agent_health[target] <= 0:
                rewards[target] -= 20.0
                self.respawn_timers[target] = self.respawn_delay
                self._drop_to_spawn(target)

    def _handle_objective_damage(self, rewards: Dict[int, float]):
        for aid in range(self.num_agents):
            if aid in self.respawn_timers:
                continue
            lane = self._closest_lane(self.agent_positions[aid][1])
            target_tower = self._next_enemy_tower(aid, lane)
            target_pos = target_tower.pos if target_tower else self._enemy_core_for_agent(aid)
            if self._distance(self.agent_positions[aid], target_pos) > 1.5:
                continue

            if target_tower is not None:
                target_tower.hp = max(0.0, target_tower.hp - self.objective_damage)
                rewards[aid] += 6.0
                if not target_tower.alive:
                    rewards[aid] += 20.0
                    for teammate in self._allied_agents(aid):
                        rewards[teammate] += 4.0
            else:
                if aid in self.red_team:
                    self.blue_core_hp = max(0.0, self.blue_core_hp - self.objective_damage)
                else:
                    self.red_core_hp = max(0.0, self.red_core_hp - self.objective_damage)
                rewards[aid] += 12.0
                if self.red_core_hp <= 0 or self.blue_core_hp <= 0:
                    for teammate in self._allied_agents(aid):
                        rewards[teammate] += 100.0

    def _apply_shaping_rewards(self, rewards: Dict[int, float], actions: Dict[int, str]):
        for aid in range(self.num_agents):
            if aid in self.respawn_timers:
                continue

            current_dist = self._distance(self.agent_positions[aid], self._enemy_core_for_agent(aid))
            if current_dist < self.last_dist_to_goal[aid]:
                rewards[aid] += 5.0
            self.last_dist_to_goal[aid] = current_dist

            if actions.get(aid, "stay") == "stay":
                rewards[aid] -= 1.0
            else:
                rewards[aid] += 1.0

            rewards[aid] += 1.0

            lane = self._closest_lane(self.agent_positions[aid][1])
            lane_mates = [
                teammate for teammate in self._allied_agents(aid)
                if teammate not in self.respawn_timers and self._closest_lane(self.agent_positions[teammate][1]) == lane
            ]
            if len(lane_mates) >= 2:
                rewards[aid] += 2.0

    def _nearest_enemy_in_range(self, aid: int):
        attack_range = self.ROLE_STATS[self.agent_roles[aid]]["attack_range"]
        candidates = []
        for enemy in self._enemy_agents(aid):
            if enemy in self.respawn_timers:
                continue
            dist = self._distance(self.agent_positions[aid], self.agent_positions[enemy])
            if dist <= attack_range:
                candidates.append((dist, enemy))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _respawn_agent(self, aid: int):
        base = self.red_base if aid in self.red_team else self.blue_base
        offset_y = random.choice([-1, 0, 1])
        x = min(self.grid_size - 1, max(0, base[0]))
        y = min(self.grid_size - 1, max(0, base[1] + offset_y))
        self.agent_positions[aid] = (x, y)
        self.agent_health[aid] = self._max_health_for_agent(aid)
        self.attack_cooldowns[aid] = 0
        self.last_dist_to_goal[aid] = self._distance(self.agent_positions[aid], self._enemy_core_for_agent(aid))

    def _drop_to_spawn(self, aid: int):
        base = self.red_base if aid in self.red_team else self.blue_base
        self.agent_positions[aid] = base

    def _closest_lane(self, y_coord: int) -> str:
        return min(self.lane_rows, key=lambda lane: abs(self.lane_rows[lane] - y_coord))

    def _next_enemy_tower(self, aid: int, lane: str):
        enemy_team = "blue" if aid in self.red_team else "red"
        towers = [tower for tower in self.towers if tower.team == enemy_team and tower.lane == lane and tower.alive]
        if not towers:
            return None
        return min(towers, key=lambda tower: 0 if tower.tier == "outer" else 1)

    def _max_health_for_agent(self, aid: int) -> float:
        return self.ROLE_STATS[self.agent_roles[aid]]["max_hp"]

    def _enemy_core_for_agent(self, aid: int) -> Tuple[int, int]:
        return self.blue_core_pos if aid in self.red_team else self.red_core_pos

    def _allied_agents(self, aid: int) -> List[int]:
        return self.red_team if aid in self.red_team else self.blue_team

    def _enemy_agents(self, aid: int) -> List[int]:
        return self.blue_team if aid in self.red_team else self.red_team

    @staticmethod
    def _distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
