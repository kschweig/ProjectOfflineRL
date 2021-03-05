import osthere is quite some overhead to safe datasets as efficient then
import numpy as np
import torch
from source.utils.atari_wrapper import make_env


class ReplayBuffer(object):

    def __init__(self, params):
        self.batch_size = params.batch_size
        self.max_size = int(params.buffer_size)
        self.device = params.device

        self.state_history = 4

        self.idx = 0
        self.current_size = 0

        self.state = np.zeros((self.max_size + 1, 84, 84), dtype=np.uint8)
        self.action = np.zeros((self.max_size, 1), dtype=np.int64)
        self.reward = np.zeros((self.max_size, 1))

        # not_done only consider "done" if episode terminates due to failure condition
        # if episode terminates due to timelimit, the transition is not added to the buffer
        self.not_done = np.zeros((self.max_size, 1))
        self.first_timestep = np.zeros(self.max_size, dtype=np.uint8)

    def full(self):
        """
        returns True if buffer is at full capacity
        """
        return self.idx == self.max_size - 1

    def add(self, state, action, next_state, reward, done, env_done, first_timestep):
        # If dones don't match, env has reset due to timelimit
        # and we don't add the transition to the buffer
        if done != env_done:
            return

        self.state[self.idx] = state[0]
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.not_done[self.idx] = 1. - done
        self.first_timestep[self.idx] = first_timestep

        self.idx = (self.idx + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def sample(self, batch_size=None):
        if batch_size != None:
            self.batch_size = batch_size
        ind = np.random.randint(0, self.current_size, size=self.batch_size)

        # + is concatenate here
        state = np.zeros(((self.batch_size, self.state_history) + self.state.shape[1:]), dtype=np.uint8)
        next_state = np.array(state)

        state_not_done = 1.
        next_not_done = 1.
        for i in range(self.state_history):

            # Wrap around if the buffer is filled
            if self.current_size == self.max_size:
                j = (ind - i) % self.max_size
                k = (ind - i + 1) % self.max_size
            else:
                j = ind - i
                k = (ind - i + 1).clip(min=0)
                # If j == -1, then we set state_not_done to 0.
                state_not_done *= (j + 1).clip(min=0, max=1).reshape(-1, 1, 1)
                j = j.clip(min=0)

            # State should be all 0s if the episode terminated previously
            state[:, i] = self.state[j] * state_not_done
            next_state[:, i] = self.state[k] * next_not_done

            # If this was the first timestep, make everything previous = 0
            next_not_done *= state_not_done
            state_not_done *= (1. - self.first_timestep[j]).reshape(-1, 1, 1)

        return (
            torch.ByteTensor(state).to(self.device).float(),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.ByteTensor(next_state).to(self.device).float(),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def save(self, save_folder, number, chunk=int(1e5)):
        """
        Save Transition Buffer to data folder
        """
        np.save(f"{save_folder}_action_n{number}.npy", self.action[:self.current_size])
        np.save(f"{save_folder}_reward_n{number}.npy", self.reward[:self.current_size])
        np.save(f"{save_folder}_not_done_n{number}.npy", self.not_done[:self.current_size])
        np.save(f"{save_folder}_first_timestep_n{number}.npy", self.first_timestep[:self.current_size])
        np.save(f"{save_folder}_replay_info_n{number}.npy", [self.idx, chunk])

        current = 0
        end = min(chunk, self.current_size + 1)
        while current < self.current_size + 1:
            np.save(f"{save_folder}_state_n{number}_{end}.npy", self.state[current:end])
            current = end
            end = min(end + chunk, self.current_size + 1)

    def load(self, save_folder, number, size=-1):
        """
        load Transition Buffer from data folder
        """
        reward_buffer = np.load(f"{save_folder}_reward_n{number}.npy")

        # Adjust current_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.current_size = min(reward_buffer.shape[0], size)

        self.action[:self.current_size] = np.load(f"{save_folder}_action_n{number}.npy")[:self.current_size]
        self.reward[:self.current_size] = reward_buffer[:self.current_size]
        self.not_done[:self.current_size] = np.load(f"{save_folder}_not_done_n{number}.npy")[:self.current_size]
        self.first_timestep[:self.current_size] = np.load(f"{save_folder}_first_timestep_n{number}.npy")[:self.current_size]

        self.idx, chunk = np.load(f"{save_folder}_replay_info_n{number}.npy")

        current = 0
        end = min(chunk, self.current_size + 1)
        while current < self.current_size + 1:
            self.state[current:end] = np.load(f"{save_folder}_state_n{number}_{end}.npy")
            current = end
            end = min(end + chunk, self.current_size + 1)


class DatasetGenerator():
    """
    Utility class, that holds a replay buffer and handles dataset generation
    by the online agent. Not most efficient way, better would be to store as hdf5,
    but as I built upon an existing buffer solution, this will be done in the follow-up
    project as it works as is, just not that efficient.
    """

    def __init__(self, params):
        self.params = params
        self.env = make_env(params.env, params)
        self.replay_buffer = ReplayBuffer(params)
        self.number = 0

        self.done = False
        self.reset()

    def reset(self):
        self.low_p = False
        if np.random.uniform(0, 1) < self.params.low_noise_p:
            self.low_p = True
        self.state = self.env.reset()
        self.episode_timestep = 0
        self.episode_start = True

    def gen_data(self, steps, agent):
        for step in range(steps):
            self.episode_timestep += 1

            if self.done:
                self.reset()

            if self.episode_timestep >= self.env._max_episode_steps:
                self.done = False

            # if we are in low prob episode, generate
            if self.low_p:
                eps = self.params.eval_eps
            else:
                eps = self.params.gen_eps

            # take action
            action, _, _ = agent.policy(self.state, eval=True, eps=eps)

            next_state, reward, self.done, _ = self.env.step(action)

            self.replay_buffer.add(self.state, action, next_state, reward, float(self.done), self.done, self.episode_start)
            self.episode_start = False

            # if buffer is full, just save to disk and create a new one
            if self.replay_buffer.full():
                self.save_and_reset()

    def set_data(self, state, action, next_state, reward, done, episode_start):
        self.replay_buffer.add(state, action, next_state, reward, float(done), done, episode_start)
        # if buffer is full, just save to disk and create a new one
        if self.replay_buffer.full():
            self.save_and_reset()

    def save_and_reset(self):
        path = os.path.join("data", self.params.experiment, "dataset", "ds")
        self.replay_buffer.save(path, self.number)
        del self.replay_buffer
        self.replay_buffer = ReplayBuffer(self.params)
        self.number += 1
