from agents.Agent import Agent


class RandomAgent(Agent):
    def get_name(self):
        return "RandomAgent"

    def policy(self, obs):
        return self.actionspec.sample()

    def update(self, state, action, reward, next_step, done):
        return

    def check_integrity(self) -> bool:
        return True

    def save_state(self) -> None:
        return

    def load_state(self) -> None:
        return
