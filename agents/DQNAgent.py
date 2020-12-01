from agents.Agent import Agent
import torch
import torch.nn as nn


class DQNAgent(Agent):

    def __init__(self,
                 actionspace,
                 obspace,
                 directory,
                 offline):

        super.__init__(actionspace, obspace, directory, offline)

        # use gpu if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # DQN hyperparameters
        self.buffer_size = 10**7 # size of the buffer to use: 1M
        self.minibatch_size = 32  # size of the minibatch sampled
        self.tau = 0.005 # hyperparameter for updating the target network
        self.gamma = 0.99  # discount factor
        self.min_replay_history = 20000 # update after num time steps
        self.epsilon_initial = 1
        self.epsilon_train = 0.01
        self.epsilon_eval = 0.001

        # after Mnih et al. 2015 -> start with 1 decay over 250000 steps to 0.01
        self.epsilon_decay_period = 250000

        #networks
        self.Q = Network(self.obspace, self.actionspace)
        self.Q_t = Network(self.obspace, self.actionspace)

        # Optimizer hyperparams -> RMSProp
        self.learning_rate = 0.00025
        self.eps = 1e-5
        self.weight_decay = 0.95
        self.momentum = 0.0
        self.centered = True
        self.optimizer =

    # Update Target network
    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def get_name(self):
        return "DQNAgent"

    def policy(self, obs):
        return self.actionspec.sample()

    def update(self, state, action, reward, next_step, done):
        return

    def train(self):
        return

    def check_integrity(self) -> bool:
        return True

    def save_state(self) -> None:
        return

    def load_state(self) -> None:
        return


class Network(nn.Module):

    def __init__(self, num_actions):

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(in_features=7*7, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=num_actions)
        self.selu = nn.SELU()

    def forward(self, state):
        q = self.selu(self.conv1(state))
        q = self.selu(self.conv2(q))
        q = self.selu(self.conv3(q))
        q = self.selu(self.linear1(q))
        return self.linear2(q)

