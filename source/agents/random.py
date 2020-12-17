from source.agents.agent import Agent
import numpy as np


class Random(Agent):

    def __init__(self, params):
        super(Random, self).__init__(params.action_space, params.frame_stack)

    def get_name(self):
        return "Random"

    def policy(self, state, eval=False, eps = None):
        return np.random.randint(self.action_space), 0, 0

    def train(self, replay_buffer):
        return

    def save_state(self) -> None:
        return

    def load_state(self) -> None:
        return
