import time
import torch
import argparse
from source.utils.utils import load_config, bcolors, ParameterManager
from source.utils.atari_wrapper import make_env
from source.agents.dqn import DQN


# render game with different policies!

if __name__ == "__main__":
    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Breakout")  # OpenAI gym environment name
    parser.add_argument("--seed", default=42, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--config", default="experiment")  # experiment config to load
    parser.add_argument("--render", action="store_true")  # render agent
    parser.add_argument("--plot", action="store_true")  # plot performance of offline agents compared to online agent
    parser.add_argument("--online", action="store_true") # visualize the given agent online, if not specified, offline
    parser.add_argument("--agent", default="dqn") # which agent should be visualized? options: 'dqn', 'bcq', 'rem', 'qrdqn', 'behavioral' or 'all'
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

    # for now, just show me the online agent
    agent = DQN(params)
    agent.load_state(online=params.online, run=1)

    state, done = env.reset(), False
    while not done:
        env.render()
        action, _, _ = agent.policy(state, eval=True)
        state, r, done, _ = env.step(action)
        time.sleep(.03)
    env.close()