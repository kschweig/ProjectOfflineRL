from source.utils.atari_wrapper import make_env
from source.utils.replay_buffer import ReplayBuffer, DatasetGenerator
from source.agents.dqn import DQN
from source.agents.bcq import BCQ
from source.agents.rem import REM
from source.agents.qrdqn import QRDQN
from source.agents.random import Random
from collections import deque
from source.utils.utils import load_config, bcolors, ParameterManager
from source.utils.logging import TrainLogger
import os
import argparse
import torch
import copy
import math
import numpy as np
from tqdm import tqdm


def online(params):
    """
    Train agent online on environment.
    :param env: Environment
    :param action_space: dim of action space
    :param frames: framestack, therefore first dim of state-space
    :param config: configuration object, representing the experimentation details
    :param device: device to operate networks on, should be gpu
    :return:
    """

    # create agent
    agent = get_agent(params)
    # create replay buffer
    replay_buffer = ReplayBuffer(params)
    # create Environment
    env = make_env(params.env, params)
    # helper to generate datasets
    dataset = DatasetGenerator(params)
    # create logger
    logger = TrainLogger(agent, params, run=1, online=True)

    # initialize
    state = env.reset()
    episode_timestep = 0
    episode_reward = 0
    episode_num = 0
    episode_start = True
    # keep highest rewards (eval and behavioral)
    highest_reward_eval = 0
    episode_rewards = deque(maxlen=params.eval_iters)
    highest_reward_behavioral = 0

    # Interact with the environment for max_timesteps
    for t in tqdm(range(params.max_timesteps)):

        episode_timestep += 1
        episode_start = False

        # every k episodes, policy gets evaluated
        if (t+1) % params.eval_freq == 0:
            eval_reward = eval_policy(t, agent, logger, params, eval_episodes=params.eval_iters)
            # save highest performing agent
            if eval_reward > highest_reward_eval:
                highest_reward_eval = eval_reward
                agent.save_state(online=True, run=1)

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
        # if we use setup from Agarwal et at. 2020 (Optimistic perspective) use training buffer as dataset
        if params.use_train_buffer:
            dataset.set_data(state, action, next_state, reward, done, episode_start)

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
            # store in window of rewards
            episode_rewards.append(episode_reward)
            if np.mean(episode_rewards) > highest_reward_behavioral:
                highest_reward_behavioral = np.mean(episode_rewards)
            episode_reward = 0
            episode_timestep = 0
            episode_num += 1

        # generate dataset for offline agents, setup from Fujimoto et al. 2019 (Benchmarking Batch DRL)
        steps = params.max_timesteps // params.policies
        if (t+1) % steps == 0 and not params.use_train_buffer:
            dataset.gen_data(steps, agent)

    # once finished, safe final policy
    agent.save_state(online=True, run=2)

    logger.plot()
    logger.save(highest_reward_eval, highest_reward_behavioral)


def offline(params):
    """
    Train agent in offline environment for multiple runs
    :param params:
    :return:
    """
    replay_buffer = ReplayBuffer(params)
    # number of datasets is how often the buffer size can fit into the max_timesteps
    num_ds = math.ceil(params.max_timesteps / params.buffer_size)
    # path to datasets
    path = os.path.join("data", params.experiment, "dataset", "ds")

    highest_reward = 0

    for run in range(1, params.runs + 1):

        agent = get_agent(params)

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
                eval_reward = eval_policy(t, agent, logger, params, eval_episodes=params.eval_iters)

                # save highest performing agent
                if eval_reward > highest_reward:
                    highest_reward = eval_reward
                    agent.save_state(online=False, run=run)

        logger.plot()
        logger.save(highest_reward, 0)


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

    return avg_reward


def eval_random(params):
    agent = Random(params)
    logger = TrainLogger(agent, params, run=1, online=True)
    eval_reward = eval_policy(0, agent, logger, params, eval_episodes=params.eval_iters*10)
    with open(os.path.join("data", params.experiment, "logs","Random_info.csv"), "w") as f:
        f.write(f"env;reward;\n{params.env};{round(eval_reward,2)}\n")


def get_agent(params):
    if params.agent == "dqn":
        return DQN(params)
    elif params.agent == "bcq":
        return BCQ(params)
    elif params.agent == "rem":
        return REM(params)
    elif params.agent == "qrdqn":
        return QRDQN(params)
    raise ValueError(f"You must specify an offline agent to train from [dqn, bcq, rem, qrdqn], specified: {params.agent}")


if __name__ == "__main__":
    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--config", default="experiment")  # experiment config to load
    parser.add_argument("--online", action="store_true") # Train online and generate buffer
    parser.add_argument("--offline", action="store_true")  # Train online and generate buffer
    parser.add_argument("--agent", default="dqn") # which offline agent should be trained? options: 'dqn', 'bcq', 'rem' or 'qrdqn'
    parser.add_argument("--runs", default=3, type=int) # how many runs of offline agents (for creating std afterwards)
    args = parser.parse_args()

    if args.config == "experiment":
        print(bcolors.WARNING + "Warning: executing default experiment!" + bcolors.ENDC)

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


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
        eval_random(params)
        online(params)
    elif params.offline:
        offline(params)
    else:
        raise ValueError("You have to specify whether you want to train --online of --offline!")