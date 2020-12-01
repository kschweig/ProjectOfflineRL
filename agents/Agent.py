from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self,
                 actionspace,
                 obspace,
                 directory,
                 offline):
        self.actionspace = actionspace
        self.obspace = obspace
        # all the hyperaparameters, model weights, lesson buffers, a Logger if 
        # needed, etc, should be stored in this
        # directory. Also wouldn't be bad to store some metadata like agent name
        # training session length, etc.
        self.agent_directory = directory
        self.inference = False
        self.offline = offline


    @abstractmethod
    def get_name(self) -> str:
        """
        Return the name of the agent.
        """

    @abstractmethod
    def policy(self, obs):
        """
        This function returns the action given the observation.
        :param obs: observation received
        """

    @abstractmethod
    def update(self, state, action, reward, next_step, done):
        """
        Update function
        :param state: old observation
        :param action: action taken
        :param reward: reward received
        :param next_step: the new observation
        :param done: from env.step
        """

    @abstractmethod
    def train(self):
        """
        Train the agent, either called from update method in online case
        or directly for offline training
        """

    @abstractmethod
    def check_integrity(self) -> bool:
        """
        Use to check state, if not needed return True
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

