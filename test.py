import os
import argparse
from source.utils.utils import load_config


# render game with different policies!

if __name__ == "__main__":
    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Breakout")  # OpenAI gym environment name
    parser.add_argument("--seed", default=42, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--config", default="experiment")  # experiment config to load
    parser.add_argument("--render", action="store_true")  # render agent
    parser.add_argument("--plot", action="store_true")  # plot performance of offline agents compared to online agent
    parser.add_argument("--agent", default="dqn") # which agent should be visualized? options: 'dqn', 'bcq', 'rem', 'qrdqn', 'behavioral' or 'all'
    args = parser.parse_args()

    if args.config == "experiment":
        print("Warning: executing default experiment!")

    config = load_config(args.config)

