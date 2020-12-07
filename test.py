from source.utils.AtariWrapper import make_env
from source.agents.DQN import DQN
from source.agents.BCQ import BCQ
from source.agents.REM import REM
from source.agents.QRDQN import QRDQN
from source.agents.Random import Random
from source.utils.utils import load_config
import argparse
import torch
import numpy as np


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
        print("Warning: executing default experiment!")

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

    #define action space and framestack
    action_space = env.action_space.n
    frames = atari_pp.frame_stack

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
        pass
    elif args.offline:
        pass
    else:
        # TODO: first do online with dqn agent, then offline with all agents
        #online
        agents = [DQN()]

        #offline
        agents = []
        agents.append(DQN())
        agents.append(BCQ())
        agents.append(REM())
        agents.append(QRDQN())
        agents.append(Random())

        pass
