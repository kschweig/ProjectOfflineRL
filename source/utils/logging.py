from collections import deque
import numpy as np
import matplotlib.pyplot as plt


class TrainLogger():

    def __init__(self, config):
        self.config = config
        # sliding window of 5
        self.reward_window = deque(maxlen=config.window)
        self.rewards = []
        self.episode_length_window = deque(maxlen=config.window)
        self.episode_lengths = []
        self.entropy_window = deque(maxlen=config.window)
        self.entropies = []
        self.value_window = deque(maxlen=config.window)
        self.values = []

    def append(self, reward, episode_length, entropy, value):
        self.reward_window.append(reward)
        self.rewards.append(np.mean(self.reward_window))
        self.episode_length_window.append(episode_length)
        self.episode_lengths.append(np.mean(self.episode_length_window))
        self.entropy_window.append(entropy)
        self.entropies.append(np.mean(self.entropy_window))
        self.value_window.append(value)
        self.values.append(np.mean(self.value_window))

    def plot(self):
        plt.figure(figsize=(8,6))
        plt.ylabel("Reward")
        plt.xlabel("Time steps")
        plt.plot(np.linspace(0, self.config.max_timesteps, len(self.rewards)), self.rewards)
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.ylabel("Episode Length")
        plt.xlabel("Time steps")
        plt.plot(np.linspace(0, self.config.max_timesteps, len(self.episode_lengths)), self.episode_lengths)
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.ylabel("Entropy")
        plt.xlabel("Time steps")
        plt.plot(np.linspace(0, self.config.max_timesteps, len(self.entropies)), self.entropies)
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.ylabel("Estimated Value")
        plt.xlabel("Time steps")
        plt.plot(np.linspace(0, self.config.max_timesteps, len(self.values)), self.values)
        plt.show()