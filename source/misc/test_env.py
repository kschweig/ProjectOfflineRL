import sys
import torch
from source.utils.atari_wrapper import make_env
from source.utils.utils import load_config
import numpy as np

"""
Just a few tests, to see if environment is usable and to check for some errors
"""

print (sys.version)
print(torch.cuda.is_available())

atari_pp = load_config("atari_preprocessing")

env = make_env("Breakout", atari_pp)

env.reset()
print(env.action_space.n)
print(env.observation_space.shape)
print(env.unwrapped.get_action_meanings())


for i in range(300):
    env.render()
    action = 0
    state, r, done, lives = env.step(action) # take a random action
    print(i, env.unwrapped.get_action_meanings()[action], r, done, lives)


action = env.action_space.sample()
observation, reward, done, info = env.step(action)
print(observation.shape, reward, done, info)

env.close()

# cuda?
print("cuda available:", torch.cuda.is_available())

