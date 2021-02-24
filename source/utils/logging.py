from collections import deque
import os
import numpy as np
import matplotlib.pyplot as plt


class TrainLogger():

    def __init__(self, agent, params, run, online):
        self.agent = agent
        self.params = params
        self.online = online
        self.run = run

        # timestep
        self.timesteps = []
        # sliding window of 5
        self.reward_window = deque(maxlen=params.window)
        self.rewards = []
        self.episode_length_window = deque(maxlen=params.window)
        self.episode_lengths = []
        self.entropy_window = deque(maxlen=params.window)
        self.entropies = []
        self.value_window = deque(maxlen=params.window)
        self.values = []

    def append(self, reward, episode_length, entropy, value):
        self.timesteps.append((len(self.timesteps) + 1) * self.params.eval_freq)
        self.reward_window.append(reward)
        self.rewards.append(np.mean(self.reward_window))
        self.episode_length_window.append(episode_length)
        self.episode_lengths.append(np.mean(self.episode_length_window))
        self.entropy_window.append(entropy)
        self.entropies.append(np.mean(self.entropy_window))
        self.value_window.append(value)
        self.values.append(np.mean(self.value_window))

    def plot(self):
        fig, axs = plt.subplots(2, 2, figsize=(16,12))

        axs[0,0].set_ylabel("Reward")
        axs[0,0].set_xlabel("Time steps")
        axs[0,0].plot(self.timesteps, self.rewards)

        axs[0,1].set_ylabel("Episode Length")
        axs[0,1].set_xlabel("Time steps")
        axs[0,1].plot(self.timesteps, self.episode_lengths)

        axs[1,0].set_ylabel("Entropy")
        axs[1,0].set_xlabel("Time steps")
        axs[1,0].plot(self.timesteps, self.entropies)

        axs[1,1].set_ylabel("Estimated Value")
        axs[1,1].set_xlabel("Time steps")
        axs[1,1].plot(self.timesteps, self.values)

        # save and show
        fig.patch.set_alpha(0)
        mode = "online" if self.online else "offline"
        plt.savefig(os.path.join("results", self.params.experiment, self.agent.get_name()+ "_" + mode + f"_{self.run}.pdf"),
                    facecolor=fig.get_facecolor(), bbox_inches='tight')

    def save(self, eval_reward, train_reward):
        mode = "online" if self.online else "offline"
        with open(os.path.join("data", self.params.experiment, "logs", self.agent.get_name()+ "_" + mode + f"_{self.run}.csv"), "w") as f:
            f.write("timestep;reward;episode length;entropy;estimated value\n")
            for i in range(len(self.timesteps)):
                f.write(f"{self.timesteps[i]};{self.rewards[i]};{self.episode_lengths[i]};{self.entropies[i]};{self.values[i]}\n")
        with open(os.path.join("data", self.params.experiment, "logs",
                               self.agent.get_name() + "_" + mode + f"_{self.run}_info.csv"), "w") as f:
            f.write(f"eval_reward;train_reward;{eval_reward};{train_reward}\n")
