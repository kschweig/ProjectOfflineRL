import time
import os
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from source.utils.utils import load_config, bcolors, ParameterManager
from source.utils.atari_wrapper import make_env
from source.agents.dqn import DQN
from source.agents.qrdqn import QRDQN
from source.agents.bcq import BCQ
from source.agents.rem import REM
from source.agents.random import Random
from source.utils.state_estimation import estimate_randenc, estimate_sklearn
from source.utils.replay_buffer import ReplayBuffer


# render game with different policies!

if __name__ == "__main__":
    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--config", default="experiment")  # experiment config to load
    parser.add_argument("--online", action="store_true")  # use online agent, otherwise offline
    parser.add_argument("--agent", default="dqn") # which agent should be visualized? options: 'dqn', 'bcq', 'rem', 'qrdqn', 'random' or 'behavioral' (online dqn)
    parser.add_argument("--coverage", action="store_true") # estimate state coverage of respective config dataset
    parser.add_argument("--run", default=1, type=int) # which run should be taken?
    args = parser.parse_args()

    if args.config == "experiment":
        print(bcolors.WARNING + "Warning: executing default experiment!" + bcolors.ENDC)


    atari_pp = load_config("atari_preprocessing")
    config = load_config(args.config)
    env = make_env(config.env, atari_pp)

    # set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    # set experiment and action_space
    config.set_value("experiment", args.config)
    config.set_value("action_space", env.action_space.n)

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # unified access to all parameters
    params = ParameterManager(config, atari_pp, args, device)

    # if we want to get coverage of dataset
    if params.coverage:
        samples = 10000
        num_ds = math.ceil(params.max_timesteps / params.buffer_size)
        states, rewards = [],[]

        for ds in tqdm(range(num_ds), desc="loading samples"):
            replay_buffer = ReplayBuffer(params)
            path = os.path.join("data", params.experiment, "dataset", "ds")
            replay_buffer.load(path, ds)

            state, _, _, reward, _ = replay_buffer.sample(batch_size=samples//num_ds)
            states.append(state)
            rewards.append(reward)

        estimate_randenc(torch.cat(states, dim=0), torch.cat(rewards, dim=0), params, k=10, mesh=int(np.sqrt(samples)))
        estimate_sklearn(torch.cat(states, dim=0), torch.cat(rewards, dim=0), params, mesh=int(np.sqrt(samples)))

    # else we show performance in environment
    else:

        if params.agent == "dqn":
            agent = DQN(params)
        elif params.agent == "rem":
            agent = REM(params)
        elif params.agent == "qrdqn":
            agent = QRDQN(params)
        elif params.agent == "bcq":
            agent = BCQ(params)
        elif params.agent == "random":
            agent = Random(params)
        else:
            agent = Random(params)
            print("invalid agent given, load random agent")
        agent.load_state(online=params.online, run=params.run)

        state, done = env.reset(), False
        reward = 0
        while not done:
            env.render()
            action, _, _ = agent.policy(state, eval=False, eps=0)
            state, r, done, _ = env.step(action)
            reward += r
            time.sleep(.03)
        env.close()
        print("reward: ", reward)