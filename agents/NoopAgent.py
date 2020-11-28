import numpy as np
from agents.Agent import Agent


class NoopAgent(Agent):

    def get_name(self):
        return "NoopAgent"

    def policy(self, obs):
        return np.zeros(self.actionspec.n)

    def update(self, state, action, reward, next_step, done):
        return

    def check_integrity(self) -> bool:
        return True

    def save_state(self) -> None:
        return

    def load_state(self) -> None:
        return

