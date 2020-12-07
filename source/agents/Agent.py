from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self,
                 action_space,
                 frames):
        self.action_space = action_space
        self.frames = frames


    @abstractmethod
    def get_name(self) -> str:
        """
        Return the name of the agent.
        """

    @abstractmethod
    def policy(self, state):
        """
        This function returns the action given the observation.
        :param obs: observation received
        """

    @abstractmethod
    def train(self, replay_buffer):
        """
        Train the agent, either called from update method in online case
        or directly for offline training
        """

    @abstractmethod
    def save_state(self) -> None:
        """
        Use this method to save the current state of your agent to the agent_directory.
        """

    @abstractmethod
    def load_state(self) -> None:
        """ 
        Use this method to load the agent state from the self.agent_directory
        """

