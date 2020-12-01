import sys
import torch
import gym

"""
Just a view tests, to see if environment is usable and to check for some errors
"""

print (sys.version)
print(torch.cuda.is_available())
env = gym.make('Breakout-v0')
env.reset()
print(env.action_space.n)
print(env.unwrapped.get_action_meanings())
for _ in range(2000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()