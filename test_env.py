import random

from envs.ctf_env import CTFEnv


env = CTFEnv()

for episode in range(5):
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        actions = {
            agent_id: random.choice(["up", "down", "left", "right", "stay"])
            for agent_id in range(env.num_agents)
        }
        obs, rewards, done, info = env.step(actions)
        total_reward += sum(rewards.values())
    print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}")
