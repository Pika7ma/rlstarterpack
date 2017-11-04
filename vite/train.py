import gym
import numpy as np


class ValueIteration(object):

    def __init__(self, env_name):
        """
        initialize gym and random seed
        """
        self.env = gym.make(env_name)
        self.env.seed(0)

    def random_play(self, time_step):
        self.env.reset()
        for _ in range(time_step):
            self.env.render()
            ob, rew, done, _ = self.env.step(self.env.action_space.sample())
            if done:
                break


def main():
    vite = ValueIteration("FrozenLake-v0")
    vite.random_play(1000)


if __name__ == '__main__':
    main()
