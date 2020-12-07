from source.agents.Agent import Agent


class BCQ(Agent):

    def __init__(self,
                 action_space,
                 frames,
                 config,
                 device):
        super(BCQ, self).__init__(action_space, frames)

    def get_name(self):
        return "BCQ"

    def policy(self, state):
        return self.actionspec.sample()

    def train(self, replay_buffer):
        return

    def save_state(self) -> None:
        return

    def load_state(self) -> None:
        return
