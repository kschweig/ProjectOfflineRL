from source.agents.agent import Agent


class QRDQN(Agent):

    def __init__(self,
                 action_space,
                 frames,
                 config,
                 device):
        super(QRDQN, self).__init__(action_space, frames)

    def get_name(self):
        return "QRDQN"

    def policy(self, state):
        return self.actionspec.sample()

    def train(self, replay_buffer):
        return

    def save_state(self) -> None:
        return

    def load_state(self) -> None:
        return
