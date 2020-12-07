import sys
import torch
from source.utils.AtariWrapper import make_env
from source.utils.utils import load_config

"""
Just a view tests, to see if environment is usable and to check for some errors
"""

print (sys.version)
print(torch.cuda.is_available())

atari_pp = load_config("atari_preprocessing")

env = make_env("Breakout", atari_pp)

env.reset()
print(env.action_space.n)
print(env.observation_space.shape)
print(env.unwrapped.get_action_meanings())


for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action


action = env.action_space.sample()
observation, reward, done, info = env.step(action)
print(observation.shape, reward, done, info)

env.close()


#cuda?
print("cuda available:", torch.cuda.is_available())

