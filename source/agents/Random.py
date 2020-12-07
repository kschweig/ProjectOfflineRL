from source.agents.Agent import Agent


class Random(Agent):

    def get_name(self):
        return "Random"

    def policy(self, state):
        return self.actionspec.sample()

    def train(self, replay_buffer):
        return

    def save_state(self) -> None:
        return

    def load_state(self) -> None:
        return
