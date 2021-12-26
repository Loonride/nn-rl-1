import torch
from torch import nn
from pathlib import Path
import numpy as np
import time, datetime
import matplotlib.pyplot as plt
import gym
from gym_env import create_env
from agent import Agent
from logger import MetricLogger

env = create_env()
env.reset()

for i in range(10):
    next_state, reward, done, info = env.step(action=env.action_space.sample())
# print(f"{next_state.shape},\n {reward},\n {done},\n {info}")
frame1 = next_state[0]
plt.imshow(frame1.squeeze(), cmap="gray")
# plt.show()

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")

save_dir = Path("/local/knagaitsev/breakout_rl") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

agent = Agent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 40000
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:
        # Run agent on the state
        action = agent.act(state)
        # Agent performs action
        next_state, reward, done, info = env.step(action)
        # Remember
        agent.cache(state, next_state, action, reward, done)
        # Learn
        q, loss = agent.learn()
        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        if done:
            break

    logger.log_episode()

    if e % 100 == 0:
        logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)
