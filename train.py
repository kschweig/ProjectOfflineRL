from source.utils.atari_wrapper import make_env
from source.utils.replay_buffer import ReplayBuffer
from source.agents.dqn import DQN
from source.agents.bcq import BCQ
from source.agents.rem import REM
from source.agents.qrdqn import QRDQN
from source.agents.random import Random
from source.utils.utils import load_config, bcolors
from source.utils.logging import TrainLogger
import os
import argparse
import torch
import copy
import numpy as np
from tqdm import tqdm


def online(agent, env, config, device):
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
    replay_buffer = ReplayBuffer(config.batch_size, config.buffer_size, device)

    state = env.reset()
    logger = TrainLogger(agent, config, True)
    episode_timestep = 0
    episode_reward = 0
    episode_num = 0
    episode_start = True

    # Interact with the environment for max_timesteps
    for t in tqdm(range(config.max_timesteps)):

        episode_timestep += 1

        # every k episodes, policy gets evaluated
        if (t+1) % config.eval_freq == 0:
            eval_policy(t, agent, logger, 42, eval_episodes=10)

        # select action, random for start_timesteps.
        # first action of every episode must be fire
        if episode_start:
            action = 1
        else:
            if t < config.start_timesteps:
                action = env.action_space.sample()
            else:
                action, _, _ = agent.policy(state, eval=False)

        # perform action and log results
        next_state, reward, done, _ = env.step(action)

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
            # Reset environment
            if episode_num % 100 == 0:
                pass
                #print(episode_num, episode_timestep, episode_reward)
            state, done = env.reset(), False
            episode_start = True
            episode_reward = 0
            episode_timestep = 0
            episode_num += 1

    # once finished, safe behavioral policy
    agent.save_state(online=True)
    # save buffer, just for simplicity here
    # replay_buffer.save()

    logger.plot()
    logger.save()


def offline(agents, env, config, device):

    replay_buffer = ReplayBuffer(config.batch_size, config.buffer_size, device)
    # load dataset
    # TODO: must be done in loop, always switching between datasets
    replay_buffer.load()

    for agent in agents:

        logger = TrainLogger(agent, config, online=False)

        for t in range(config.max_timesteps):
            state = env.reset()

            # TODO: here switch between replay buffers

            agent.train(replay_buffer)

            # every k episodes, policy gets evaluated
            if (t + 1) % config.eval_freq == 0:
                eval_policy(t, agent, logger, 42, eval_episodes=10)


# Runs policy for 10 episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(timestep, agent, logger, seed, eval_episodes=10):

    eval_env = make_env(args.env, atari_pp)
    eval_env.seed(seed + 100)

    avg_reward = []
    avg_length = []
    avg_entropy = []
    avg_values = []
    for ep in range(eval_episodes):
        state = eval_env.reset()
        done = False
        episode_reward = 0
        episode_length = 1
        while not done:
            action, entropy, value = agent.policy(np.array(state), eval=True)
            state, reward, done, lives = eval_env.step(action)
            episode_reward += reward
            avg_entropy.append(entropy)
            avg_values.append(value)
            episode_length += 1
            #print(ep, episode_length, episode_reward, done, lives)

        avg_reward.append(episode_reward)
        avg_length.append(episode_length)

    avg_reward = np.nanmean(avg_reward)
    avg_length = np.nanmean(avg_length)
    avg_entropy = np.nanmean(avg_entropy)
    avg_values = np.nanmean(avg_values)

    logger.append(avg_reward, avg_length, avg_entropy, avg_values)

    print(
        f"Total timesteps: {timestep + 1} ({round(timestep / config.max_timesteps * 100, 2)}%) "
        f"Episode timesteps: {round(avg_length, 2)} Reward: {round(avg_reward, 1):.1f} "
        f"Entropy: {round(avg_entropy, 4)} Value: {round(avg_values, 2)}"
    )


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
        agent = DQN(action_space, frames, config, device)
        online(agent, env, config, device)
    elif args.offline:
        offline(agents, env, config, device)
    else:
        # online is done by default by dqn, otherwise train with online extra
        agent = DQN(action_space, frames, config, device)
        online(agent, env, action_space, frames, config, device)

        # offline is done with all given agents
        offline(agents, env, config, device)