import gym
import numpy as np
from dqn.memory import Memory
from dqn.brain import Brain


class DQN(object):
    def __init__(self, env, batch_size=64, discount=0.99, target_update_freq=300,
                 starting_point=1000, memory_capacity=1000, initial_eps=1.0, final_eps=0.02, discount_eps=0.99,
                 mlp_units=(64,), mlp_ac="relu", render=False, max_step=int(1e5), random_seed=0):
        self.env = env
        self.target_update_freq = target_update_freq
        self.starting_point = starting_point if batch_size <= starting_point else batch_size
        self.batch_size = batch_size
        self.discount = discount
        self.memory_capacity = memory_capacity
        self.render = render
        self.epsilon = initial_eps
        self.final_eps = final_eps
        self.discount_eps = discount_eps
        self.mlp_units = mlp_units
        self.obs_num = env.observation_space.shape[0]
        self.act_num = env.action_space.n
        self.rng = np.random.RandomState(random_seed)
        self.memory = Memory(
            capacity=self.memory_capacity,
            obs_dim=self.obs_num,
            act_dim=self.act_num,
            random_seed=random_seed
        )
        self.brain = Brain(
            mlp_units=mlp_units,
            mlp_ac=mlp_ac,
            obs_num=self.obs_num,
            act_num=self.act_num,
            discount=discount
        )
        self.max_step = max_step

    def greedy_act(self):
        if self.epsilon > self.final_eps:
            self.epsilon *= self.discount_eps
        return self.rng.rand() > self.epsilon

    def choose_act(self, obs):
        if self.greedy_act():
            # act greedily
            return np.asscalar(self.brain.get_opt_action(obs))
        else:
            return self.rng.randint(self.act_num)

    def optimize(self):
        batch = self.memory.random_recall(self.batch_size)
        return self.brain.optimize(batch)

    def pretrain(self):
        obs = self.env.reset()
        for time_step in range(self.starting_point):
            act = self.rng.randint(self.act_num)
            obs_, rew, done, _ = self.env.step(act)
            self.memory.remember(obs, act, rew, done)
            obs = obs_
            if done:
                obs = self.env.reset()

    def train(self):
        self.pretrain()
        episode = 0
        obs = self.env.reset()
        episode_rew = 0
        for time_step in range(self.max_step):
            if self.render:
                self.env.render()
            act = self.choose_act(obs)
            obs_, rew, done, _ = self.env.step(act)
            self.memory.remember(obs, act, rew, done)
            self.optimize()
            episode_rew += rew
            obs = obs_
            if time_step % self.target_update_freq == 0:
                self.brain.eval2target()
                print("param updated")
            if done:
                episode += 1
                print("reward in episode %d is %f, eps is %f" % (episode, episode_rew, self.epsilon))
                episode_rew = 0
                obs = self.env.reset()


def main():
    env = gym.make("CartPole-v0")
    dqn = DQN(env, render=False)
    dqn.train()


if __name__ == '__main__':
    main()
