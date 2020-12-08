from source.utils.atari_wrapper import make_env
from source.utils.replay_buffer import ReplayBuffer
from source.agents.dqn import DQN
from source.agents.bcq import BCQ
from source.agents.rem import REM
from source.agents.qrdqn import QRDQN
from source.agents.random import Random
from source.utils.utils import load_config, Configuration, bcolors
import os
import argparse
import torch
import copy
import numpy as np
from tqdm import tqdm


def online(env, action_space, frames, config, device):
    """
    Train agent online on environment.
    TODO: create dataset
    :param env: Environment
    :param action_space: dim of action space
    :param frames: framestack, therefore first dim of state-space
    :param config: configuration object, representing the experimentation details
    :param device: device to operate networks on, should be gpu
    :return:
    """

    # create agent and replay buffer
    agent = DQN(action_space, frames, config, device)
    replay_buffer = ReplayBuffer(config.batch_size, config.buffer_size, device)

    evaluations = []

    state = env.reset()
    done = False
    episode_timestep = 0
    episode_timesteps = []
    episode_reward = 0
    episode_rewards = []
    episode_num = 0
    episode_start = True

    # Interact with the environment for max_timesteps
    for t in range(config.max_timesteps):

        episode_timestep += 1

        # select action, random for start_timesteps.
        if t < config.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.policy(state, eval=False)

        # perform action and log results
        next_state, reward, done, info = env.step(action)

        # raise episode reward
        episode_reward += reward

        # only consider done if episode terminates due to failure condition, not because timesteps exceeded
        if episode_timestep >= env._max_episode_steps:
            done = False

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, float(done), done, episode_start)
        episode_start = False

        # state is next state
        state = copy.copy(next_state)

        # only update policy after initial filling of buffer and only every "train_freq"th interaction with the env
        if t >= config.start_timesteps and (t + 1) % config.train_freq == 0:
            agent.train(replay_buffer)

        # clean up upon episode end
        if done:
            # every xth episode report means of last episodes
            if episode_num % 100 == 0 and episode_num > 0:
                print(
                    f"Total timesteps: {t + 1} ({round(t/config.max_timesteps * 100, 2)}%) Episode Num: {episode_num + 1} "
                    f"Episode timesteps: {round(np.mean(episode_timesteps), 2)} Reward: {round(np.mean(episode_rewards), 2):.3f}")
            # append for mean over last episodes
            episode_rewards.append(episode_reward)
            episode_timesteps.append(episode_timestep)
            # Reset environment
            state, done = env.reset(), False
            episode_start = True
            episode_reward = 0
            episode_timestep = 0
            episode_num += 1

def offline(agents, env, action_space, frames, config, device):
    for agent in agents:
        for t in range(config.max_timesteps):
            pass


if __name__ == "__main__":

    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Breakout")  # OpenAI gym environment name
    parser.add_argument("--seed", default=42, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--config", default="experiment")  # experiment config to load
    parser.add_argument("--online", action="store_true") # Train online and generate buffer
    parser.add_argument("--offline", action="store_true")  # Train online and generate buffer
    parser.add_argument("--agent", default="dqn") # which agent should be trained? options: 'dqn', 'bcq', 'rem', 'qrdqn' or 'all'
    args = parser.parse_args()

    if args.config == "experiment":
        print(bcolors.WARNING + "Warning: executing default experiment!" + bcolors.ENDC)

    atari_pp = load_config("atari_preprocessing")
    env = make_env(args.env, atari_pp)

    # set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load experiment config
    config = load_config(args.config)
    # set experiment
    config.experiment = args.config

    # define action space and framestack
    action_space = env.action_space.n
    frames = atari_pp.frame_stack

    # create folders to store
    if not os.path.exists(os.path.join("results", config.experiment)):
        os.makedirs(os.path.join("results", config.experiment))
    #if not os.path.exists(os.path.join("results", config.experiment, "plots")):
    #    os.makedirs(os.path.join("results", config.experiment, "plots"))

    if not os.path.exists(os.path.join("data", config.experiment)):
        os.makedirs(os.path.join("data", config.experiment))
    if not os.path.exists(os.path.join("data", config.experiment, "logs")):
        os.makedirs(os.path.join("data", config.experiment, "logs"))
    if not os.path.exists(os.path.join("data", config.experiment, "models")):
        os.makedirs(os.path.join("data", config.experiment, "models"))
    if not os.path.exists(os.path.join("data", config.experiment, "dataset")):
        os.makedirs(os.path.join("data", config.experiment, "dataset"))


    agents = []
    if args.agent == "random" or args.agent == "all":
        agents.append(Random())
    if args.agent == "dqn" or args.agent == "all":
        agents.append(DQN(action_space, frames, config, device))
    if args.agent == "bcq" or args.agent == "all":
        pass
        #agents.append(BCQ())
    if args.agent == "rem" or args.agent == "all":
        pass
        #agents.append(REM())
    if args.agent == "qrdqn" or args.agent == "all":
        pass
        #agents.append(QRDQN())



    if args.online:
        online(env, action_space, frames, config, device)
    elif args.offline:
        pass
    else:
        # TODO: first do online with dqn agent, then offline with all agents + behavioral
        #online
        online(env, action_space, frames, config, device)

        #offline
        agents = []
        agents.append(DQN(action_space, frames, config, device))
        agents.append(BCQ())
        agents.append(REM())
        agents.append(QRDQN())
        agents.append(Random())

        pass