from source.agents.agent import Agent


class REM(Agent):

    def __init__(self,
                 action_space,
                 frames,
                 config,
                 device):
        super(REM, self).__init__(action_space, frames)

    def get_name(self):
        return "REM"

    def policy(self, state):
        return self.actionspec.sample()

    def train(self, replay_buffer):
        return

    def save_state(self) -> None:
        return

    def load_state(self) -> None:
        return
