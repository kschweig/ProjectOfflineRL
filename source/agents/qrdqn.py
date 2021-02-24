from source.agents.agent import Agent
from source.utils.utils import entropy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np


class QRDQN(Agent):

    def __init__(self, params):

        super(QRDQN, self).__init__(params.action_space, params.frame_stack)

        self.device = params.device
        self.params = params

        # Determine network type
        self.Q = Network(self.frames, self.action_space, params.quantiles).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)

        # Determine optimizer
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(self.Q.parameters(), lr=params.lr, eps=params.eps)
        elif params.optimizer == "RMSProp":
            self.optimizer = optim.RMSProp(self.Q.parameters(), lr=params.lr, eps=params.eps)
        else:
            raise ValueError("Optimizer not implemented")

        # loss function
        self.huber = nn.SmoothL1Loss()

        # Number of training iterations
        self.iterations = 0

        # how to update target network
        self.maybe_update_target = self.soft_target_update if params.soft_target_update else self.copy_target_update

        # slope
        self.slope = (self.params.end_eps - self.params.initial_eps) / self.params.eps_decay_period

        # quantiles
        self.quantiles = params.quantiles
        self.quantile_tau = torch.FloatTensor([i / self.quantiles for i in range(1, self.quantiles + 1)]).to(self.device)

    def get_name(self):
        return "QRDQN"

    def policy(self, state, eval=False, eps=None):
        if eps == None:
            if eval:
                eps = self.params.eval_eps
            else:
                eps = max(self.slope * self.iterations + self.params.initial_eps, self.params.end_eps)

        # epsilon greedy policy
        if np.random.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_values = self.Q.get_action(state)
                if eval:
                    return int(q_values.argmax(dim=1)), entropy(q_values), float(q_values.max(dim=1)[0])
                return int(q_values.argmax(dim=1)), np.nan, np.nan
        else:
            return np.random.randint(self.action_space), np.nan, np.nan

    def train(self, replay_buffer):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample()

        # Compute the target Q value
        with torch.no_grad():
            target_Qs = self.Q_target.forward(next_state)
            action_indices = torch.argmax(target_Qs.mean(dim=2), dim=1, keepdim=True)
            target_Qs = target_Qs.gather(1, action_indices.unsqueeze(2).expand(self.params.batch_size, 1, self.quantiles))
            assert target_Qs.shape == (self.params.batch_size, 1, self.quantiles), f"was {target_Qs.shape} instead"
            target_Qs = reward.unsqueeze(1) + not_done.unsqueeze(1) * self.params.discount * target_Qs

        # Get current Q estimate
        current_Qs = self.Q(state).gather(1, action.unsqueeze(2).expand(self.params.batch_size, 1, self.quantiles)).transpose(1,2)

        # Compute Q loss
        td_error = target_Qs - current_Qs
        assert td_error.shape == (self.params.batch_size, self.quantiles, self.quantiles), f"was {td_error.shape} instead"
        #huber_l = self.huber(current_Qs, target_Qs)

        # huber loss
        k = 1.0
        huber_l = torch.where(td_error.abs() <= k, 0.5 * td_error.pow(2), k * (td_error.abs() - 0.5 * k))

        quantil_l = abs(self.quantile_tau - (td_error.detach() < 0).float()) * huber_l
        Q_loss = quantil_l.sum(dim=1).mean(dim=1).mean()

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
            target_param.data.copy_(self.params.tau * param.data + (1.0 - self.params.tau) * target_param.data)

    def copy_target_update(self):
        """
        Hard update model parameters, "snap" target policy to local policy
        :return:
        """
        if self.iterations % self.params.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save_state(self, online, run):
        mode = "online" if online else "offline"
        torch.save(self.Q.state_dict(), os.path.join("data", self.params.experiment, "models",
                                                     self.get_name() + "_" + mode + f"_{run}_Q.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("data", self.params.experiment, "models",
                                                             self.get_name() + "_" + mode + f"_{run}_optimizer.pt"))

    def load_state(self, online, run):
        mode = "online" if online else "offline"
        self.Q.load_state_dict(torch.load(
            os.path.join("data", self.params.experiment, "models", self.get_name() + "_" + mode + f"_{run}_Q.pt")))
        self.Q_target = copy.deepcopy(self.Q)
        self.optimizer.load_state_dict(torch.load(os.path.join("data", self.params.experiment, "models",
                                                               self.get_name() + "_" + mode + f"_{run}_optimizer.pt")))


class Network(nn.Module):

    def __init__(self, frames, num_actions, quantiles):
        super(Network, self).__init__()

        self.num_actions = num_actions
        self.quantiles = quantiles

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=frames, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 64, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions * quantiles)
        )

    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(dim=0)

        features = self.conv(state)

        return self.linear(features).reshape(len(state), self.num_actions, self.quantiles)

    def get_action(self, state):
        qval_quantiles = self.forward(state)
        return qval_quantiles.mean(dim=2)