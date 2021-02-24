import numpy as np
import gym
import cv2


# Atari Preprocessing
# Code is based on https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py
# and in turns on https://github.com/sfujim/BCQ/blob/master/discrete_BCQ/utils.py
# which is the implementation from scott fujimoto (https://arxiv.org/abs/1910.01708)
# one of the two papers I aimed to compare.
# main contribution is in handling resets and live losses with several 'FIRE' actions.
class AtariPreprocessing(gym.Wrapper):

    def __init__(self, env, config):
        super(AtariPreprocessing, self).__init__(env)

        self.done_on_life_loss = config.done_on_life_loss
        self.frame_skip = config.frame_skip
        self.frame_size = config.frame_size
        self.reward_clipping = config.reward_clipping
        self._max_episode_steps = config.max_episode_steps
        self.observation_space = np.zeros((self.frame_size, self.frame_size))
        self.action_space = self.env.action_space

        self.lives = 0
        self.episode_length = 0

        # Track previous 2 frames
        self.frame_buffer = np.zeros(
            (2,
             self.env.observation_space.shape[0],
             self.env.observation_space.shape[1]),
            dtype=np.uint8
        )

        # Tracks previous 4 states
        self.state_buffer = np.zeros((config.frame_stack, self.frame_size, self.frame_size), dtype=np.uint8)

    def reset(self):
        """
        Resets environment
        """
        self.env.reset()
        ## FireRestEnv
        for _ in range(6):
            _, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        ##

        self.lives = self.env.ale.lives()
        self.episode_length = 0
        self.env.ale.getScreenGrayscale(self.frame_buffer[0])
        self.frame_buffer[1] = 0

        self.state_buffer[0] = self.adjust_frame()
        self.state_buffer[1:] = 0
        return self.state_buffer

    def step(self, action):
        """
        Takes single action is repeated for frame_skip frames (usually 4)
        Reward is accumulated over those frames
        """
        total_reward = 0.
        self.episode_length += 1

        for frame in range(self.frame_skip):
            _, reward, done, _ = self.env.step(action)
            total_reward += reward

            if self.done_on_life_loss:
                current_lives = self.env.ale.lives()
                done = True if current_lives < self.lives else done
                self.lives = current_lives

            # if not done on life loss and still have lifes left, issue FIRE action to resume game (for example in breakout)
            else:
                if self.env.ale.lives() < self.lives:
                    # make environment aware that loosing a life is bad
                    #total_reward = -100
                    self.lives = self.env.ale.lives()
                    if self.env.unwrapped.get_action_meanings()[1] == 'FIRE':
                        # restart again
                        for _ in range(6):
                            _, reward, done, _ = self.env.step(1)

            if done:
                break

            # Second last and last frame
            f = frame + 2 - self.frame_skip
            if f >= 0:
                # fill frame_buffer with grayscale image at this point
                self.env.ale.getScreenGrayscale(self.frame_buffer[f])

        self.state_buffer[1:] = self.state_buffer[:-1]
        self.state_buffer[0] = self.adjust_frame()

        if self.episode_length >= self._max_episode_steps:
            done = True

        # clip reward to -1,1
        return self.state_buffer, np.clip(total_reward, -1, 1), done, (self.env.ale.lives(), self.lives)

    def adjust_frame(self):
        """
        Take maximum over two consecutive frames as state
        Resize to 84x84
        :return:
        """

        # Take maximum over last two frames
        np.maximum(
            self.frame_buffer[0],
            self.frame_buffer[1],
            out=self.frame_buffer[0]
        )

        # Resize
        image = cv2.resize(
            self.frame_buffer[0],
            (self.frame_size, self.frame_size),
            interpolation=cv2.INTER_AREA
        )
        # return resized, maxed frame
        return np.array(image, dtype=np.uint8)


def make_env(env_name, config):
    """
    Create environment, add wrapper if necessary
    :param env_name:
    :param config:
    :return: Wrapped Environment
    """

    # custom wrapper with standard atari preprocessing
    assert not "NoFrameskip-v" in env_name, "Just pass game name without additional specifications like 'NoFrameskip-v0' which is added internally."
    env = gym.make(env_name + "NoFrameskip-v0")
    env = AtariPreprocessing(env, config)
    # try seeding env
    try:
        env.seed(config.seed)
    except:
        env.seed(42)

    return env
