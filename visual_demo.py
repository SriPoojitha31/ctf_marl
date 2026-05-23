import sys
import time
import pygame
from envs.ctf_env import CTFEnv
from algorithms.maddpg import MADDPG


ACTION_MAP = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}


def draw(env, screen, cell_size=30):
    screen.fill((200, 200, 200))
    gs = env.grid_size
    # grid
    for x in range(gs + 1):
        pygame.draw.line(screen, (180, 180, 180), (x * cell_size, 0), (x * cell_size, gs * cell_size))
    for y in range(gs + 1):
        pygame.draw.line(screen, (180, 180, 180), (0, y * cell_size), (gs * cell_size, y * cell_size))

    obs = env.get_observation()
    # draw obstacles
    for (ox, oy) in obs.get('obstacles', []):
        pygame.draw.rect(screen, (100, 100, 100), (ox * cell_size, oy * cell_size, cell_size, cell_size))
    # draw bushes
    for (bx, by) in obs.get('bushes', []):
        pygame.draw.rect(screen, (34, 139, 34), (bx * cell_size, by * cell_size, cell_size, cell_size))

    # flags
    rx, ry = obs['red_flag_position']
    bx, by = obs['blue_flag_position']
    pygame.draw.rect(screen, (200, 30, 30), (rx * cell_size + 4, ry * cell_size + 4, cell_size - 8, cell_size - 8))
    pygame.draw.rect(screen, (30, 30, 200), (bx * cell_size + 4, by * cell_size + 4, cell_size - 8, cell_size - 8))

    # towers
    rt = env.red_tower_pos
    bt = env.blue_tower_pos
    pygame.draw.circle(screen, (120, 60, 60), (rt[0] * cell_size + cell_size // 2, rt[1] * cell_size + cell_size // 2), cell_size // 2)
    pygame.draw.circle(screen, (60, 60, 120), (bt[0] * cell_size + cell_size // 2, bt[1] * cell_size + cell_size // 2), cell_size // 2)

    # agents
    for aid, pos in obs['agent_positions'].items():
        x, y = pos
        color = (200, 50, 50) if aid in env.red_team else (50, 50, 200)
        role = obs['agent_roles'].get(aid, 'Soldier')
        cx = x * cell_size + cell_size // 2
        cy = y * cell_size + cell_size // 2
        if role == 'Scout':
            pygame.draw.circle(screen, color, (cx, cy), cell_size // 4)
        elif role == 'Tank':
            pygame.draw.rect(screen, color, (x * cell_size + 2, y * cell_size + 2, cell_size - 4, cell_size - 4))
        elif role == 'Defender':
            pygame.draw.polygon(screen, color, [(cx, cy - 8), (cx - 8, cy + 8), (cx + 8, cy + 8)])
        elif role == 'Attacker':
            pygame.draw.circle(screen, color, (cx, cy), cell_size // 3, 2)
        else:
            pygame.draw.rect(screen, color, (x * cell_size + 6, y * cell_size + 6, cell_size - 12, cell_size - 12))

    pygame.display.flip()


def demo(model_path=None):
    env = CTFEnv(grid_size=20, team_size=5)
    # try to load MADDPG model if provided
    agent = None
    if model_path:
        try:
            agent = MADDPG(num_agents=env.num_agents, state_dim=9, action_dim=4)
            agent.load(model_path)
            print('Loaded model', model_path)
        except Exception as e:
            print('Could not load model:', e)

    pygame.init()
    size = env.grid_size * 30
    screen = pygame.display.set_mode((size, size))
    clock = pygame.time.Clock()

    obs = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # select random or model actions
        actions = {}
        for aid in range(env.num_agents):
            if agent is not None:
                # use actor to select action
                state = [0] * 9
                aidx = agent.select_action(aid, state, explore=False)
            else:
                import random as _r
                aidx = _r.randrange(4)
            actions[aid] = ACTION_MAP[aidx]

        env.step(actions)
        draw(env, screen)
        clock.tick(6)

    pygame.quit()


if __name__ == '__main__':
    demo() 
