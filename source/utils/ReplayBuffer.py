import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, batch_size, buffer_size, device):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device

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

    def sample(self):
        ind = np.random.randint(0, self.current_size, size=self.batch_size)

        # Note + is concatenate here
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
                state_not_done *= (j + 1).clip(min=0, max=1).reshape(-1, 1, 1)  # np.where(j < 0, state_not_done * 0, state_not_done)
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

    def save(self, save_folder, chunk=int(1e5)):
        """
        Save Transition Buffer to data folder
        :param save_folder: folder to store data in
        :param chunk: chunk buffer into pieces of this size for saving
        """
        np.save(f"{save_folder}_action.npy", self.action[:self.current_size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.current_size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.current_size])
        np.save(f"{save_folder}_first_timestep.npy", self.first_timestep[:self.current_size])
        np.save(f"{save_folder}_replay_info.npy", [self.idx, chunk])

        current = 0
        end = min(chunk, self.current_size + 1)
        while current < self.current_size + 1:
            np.save(f"{save_folder}_state_{end}.npy", self.state[current:end])
            current = end
            end = min(end + chunk, self.current_size + 1)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.current_size = min(reward_buffer.shape[0], size)

        # Adjust current_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.current_size = min(reward_buffer.shape[0], size)

        self.action[:self.current_size] = np.load(f"{save_folder}_action.npy")[:self.current_size]
        self.reward[:self.current_size] = reward_buffer[:self.current_size]
        self.not_done[:self.current_size] = np.load(f"{save_folder}_not_done.npy")[:self.current_size]
        self.first_timestep[:self.current_size] = np.load(f"{save_folder}_first_timestep.npy")[:self.current_size]

        self.idx, chunk = np.load(f"{save_folder}_replay_info.npy")

        current = 0
        end = min(chunk, self.current_size + 1)
        while current < self.current_size + 1:
            self.state[current:end] = np.load(f"{save_folder}_state_{end}.npy")
            current = end
            end = min(end + chunk, self.current_size + 1)