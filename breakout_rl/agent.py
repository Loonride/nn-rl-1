import torch
import numpy as np
from collections import deque
import random

from model import Net

class Agent:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = Net(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_index = 0

        self.save_every = 1000000
        
        self.max_memory = 100000
        self.memory = deque(maxlen=self.max_memory)
        self.batch_size = 32

        self.gamma = 0.99

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 50000  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 10000  # no. of experiences between Q_target & Q_online sync

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = state.__array__()
        next_state = next_state.__array__()

        cpu = 'cpu'
        state = torch.tensor(state).to(cpu)
        next_state = torch.tensor(next_state).to(cpu)
        action = torch.tensor([action]).to(cpu)
        reward = torch.tensor([reward]).to(cpu)
        done = torch.tensor([done]).to(cpu)

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state.to(self.device), next_state.to(self.device), action.squeeze().to(self.device), reward.squeeze().to(self.device), done.squeeze().to(self.device)

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
    
    def save(self):
        save_path = self.save_dir / f"net_{self.save_index}.p"
        mem_save_path = self.save_dir / f"mem_{self.save_index}.p"
        torch.save(
            dict(model=self.net.state_dict(), optim=self.optimizer.state_dict(), exploration_rate=self.exploration_rate, curr_step=self.curr_step),
            save_path,
        )
        torch.save(
            dict(memory=list(self.memory)),
            mem_save_path,
        )
        # self.save_index += 1
        print(f"Net saved to {save_path} at step {self.curr_step}")
        exit()

    def load(self, d=None, save_index=None):
        d = self.save_dir
        save_index = self.save_index
        save_path = d / f"net_{save_index}.p"
        mem_save_path = d / f"mem_{save_index}.p"
        saved_data = torch.load(save_path)
        mem_data = torch.load(mem_save_path)
        self.net.load_state_dict(saved_data['model'])
        self.optimizer.load_state_dict(saved_data['optim'])
        self.exploration_rate = saved_data['exploration_rate']
        self.curr_step = saved_data['curr_step']

        self.memory = deque(mem_data['memory'], maxlen=self.max_memory)
