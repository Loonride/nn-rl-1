import time
import matplotlib.pyplot as plt
from agent import Agent
from gym_env import create_env

env = create_env()
agent = Agent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=None)
agent.exploration_rate = 1.0

for e in range(1):

    state = env.reset()
    while True:
        # env.render()
        # time.sleep(0.1)
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        state = next_state

        frame1 = next_state[0]
        # plt.imshow(frame1.squeeze(), cmap="gray")
        # plt.show()

        if done:
            break

env.close()
# env = create_env()
