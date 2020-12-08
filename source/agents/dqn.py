from source.agents.agent import Agent
import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np


class DQN(Agent):

    def __init__(self,
                 action_space,
                 frames,
                 config,
                 device):
        
        super(DQN, self).__init__(action_space, frames)
            
        self.device = device
        self.config = config

        # Determine network type
        self.Q = Network(self.frames, self.action_space).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)

        # Determine optimizer
        if config.optimizer == "Adam":
            self.optimizer = optim.Adam(self.Q.parameters(), lr=config.lr, eps=config.eps)
        elif config.optimizer == "RMSProp":
            self.optimizer = optim.RMSProp(self.Q.parameters(), lr=config.lr, eps=config.eps)
        else:
            raise ValueError("Optimizer not implemented")

        # loss function
        self.huber = nn.SmoothL1Loss()

        # Number of training iterations
        self.iterations = 0

        # how to update target network
        self.maybe_update_target = self.soft_target_update if config.soft_target_update else self.copy_target_update

        # slope
        self.slope = (self.config.end_eps - self.config.initial_eps) / self.config.eps_decay_period

    def get_name(self):
        return "DQN"

    def policy(self, state, eval=False):
        if eval:
            eps = self.config.eval_eps
        else:
            eps = max(self.slope * self.iterations + self.config.initial_eps, self.config.end_eps)

        # epsilon greedy policy
        if np.random.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                return int(self.Q(state).argmax(dim=1))
        else:
            return np.random.randint(self.action_space)

    def train(self, replay_buffer):
        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample()

        # Compute the target Q value
        with torch.no_grad():
            target_Q = reward + done * self.config.discount * self.Q_target(next_state).max(1, keepdim=True)[0]

        # Get current Q estimate
        current_Q = self.Q(state).gather(1, action)

        # Compute Q loss (Huber loss)
        Q_loss = self.huber(current_Q, target_Q)

        # Optimize the Q
        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()

        # Update target network either continuously by soft update or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

    def soft_target_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)

    def copy_target_update(self):
        """
        Hard update model parameters, "snap" target policy to local policy
        :return:
        """
        if self.config.iterations % self.config.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save_state(self) -> None:
        torch.save(self.Q.state_dict(), os.path.join("data", self.config.experiment, self.get_name() + "_Q"))
        torch.save(self.Q_optimizer.state_dict(), os.path.join("data", self.config.experiment, self.get_name() + "_optimizer"))

    def load_state(self) -> None:
        self.Q.load_state_dict(torch.load(os.path.join("data", self.config.experiment, self.get_name() + "_Q")))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(os.path.join("data", self.config.experiment, self.get_name() + "_optimizer")))


class Network(nn.Module):

    def __init__(self, frames, num_actions):
        super(Network, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=frames, out_channels=32, kernel_size=8, stride=4),
            nn.SELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.SELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.SELU(),
            nn.Flatten(),
            nn.Linear(in_features=7 * 7 * 64, out_features=512),
            nn.SELU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )

    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(dim=0)

        return self.net(state)

