from source.utils.atari_wrapper import make_env
from source.utils.replay_buffer import ReplayBuffer, DatasetGenerator
from source.agents.dqn import DQN
from source.agents.bcq import BCQ
from source.agents.rem import REM
from source.agents.qrdqn import QRDQN
from source.agents.random import Random
from source.utils.utils import load_config, bcolors, ParameterManager
from source.utils.logging import TrainLogger
import os
import argparse
import torch
import copy
import math
import numpy as np
from tqdm import tqdm


def online(agent, params):
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

    # create replay buffer
    replay_buffer = ReplayBuffer(params)

    # create Environment
    env = make_env(params.env, params)

    # helper to generate datasets
    dataset = DatasetGenerator(params)

    state = env.reset()
    logger = TrainLogger(agent, params, run=1, online=True)
    episode_timestep = 0
    episode_reward = 0
    episode_num = 0
    episode_start = True

    # Interact with the environment for max_timesteps
    for t in tqdm(range(params.max_timesteps)):

        episode_timestep += 1

        # every k episodes, policy gets evaluated
        if (t+1) % params.eval_freq == 0:
            eval_policy(t, agent, logger, params, eval_episodes=10)

        # select action, random for start_timesteps.
        # first action of every episode must be fire
        if t < params.start_timesteps:
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
        if t >= params.start_timesteps and (t + 1) % params.train_freq == 0:
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

        # generate dataset for offline agents
        steps = params.max_timesteps // params.policies
        if (t+1) % steps == 0:
            dataset.gen_data(steps, agent)

    # once finished, safe behavioral policy
    agent.save_state(online=True, run=1)

    logger.plot()
    logger.save()


def offline(agent, params):

    replay_buffer = ReplayBuffer(params)
    # number of datasets is how often the buffer size can fit into the max_timesteps
    num_ds = math.ceil(params.max_timesteps / params.buffer_size)
    # path to datasets
    path = os.path.join("data", params.experiment, "dataset", "ds")

    for run in range(params.runs):
        # load dataset
        replay_buffer.load(path, np.random.randint(num_ds))

        logger = TrainLogger(agent, params, run=run, online=False)

        for t in tqdm(range(params.max_timesteps)):

            # switch between replay buffers
            if (t+1) % (params.buffer_size // 8) == 0:
                replay_buffer.load(path, np.random.randint(num_ds))

            agent.train(replay_buffer)

            # every k episodes, policy gets evaluated
            if (t + 1) % config.eval_freq == 0:
                eval_policy(t, agent, logger, params, eval_episodes=10)

        # once finished, safe obtained policy
        agent.save_state(online=True, run=run)

        logger.plot()
        logger.save()



# Runs policy for 10 episodes and returns average reward
def eval_policy(timestep, agent, logger, params, eval_episodes=10):

    eval_env = make_env(params.env, params)
    # different seed for evaluation
    eval_env.seed(params.seed + 100)

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


def seed_all(environment, seed):
    # set seeds
    environment.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--config", default="experiment")  # experiment config to load
    parser.add_argument("--online", action="store_true") # Train online and generate buffer
    parser.add_argument("--offline", action="store_true")  # Train online and generate buffer
    parser.add_argument("--agent", default="dqn") # which agent should be trained? options: 'dqn', 'bcq', 'rem', 'qrdqn' or 'all'
    parser.add_argument("--runs", default=3, type=int) # how many runs of offline agents (for creating std afterwards)
    args = parser.parse_args()

    if args.config == "experiment":
        print(bcolors.WARNING + "Warning: executing default experiment!" + bcolors.ENDC)

    # load environment config
    atari_pp = load_config("atari_preprocessing")
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load experiment config
    config = load_config(args.config)
    # set experiment and action_space
    config.set_value("experiment", args.config)
    config.set_value("action_space", make_env(config.env, atari_pp).action_space.n)

    # unified access to all parameters
    # this allows for streamlined function calls and class creations.
    params = ParameterManager(config, atari_pp, args, device)

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


    # depending on arguments of call, train online, offline or both with the provided agents.
    # Always use DQN as online baseline.
    if params.online:
        online(DQN(params), params)
    elif params.offline:
        offline(DQN(params), params)
    else:
        # online is done by default by dqn, otherwise train with online extra
        online(DQN(params), params)
        # offline is done with all given agents
        offline(DQN(params), params)