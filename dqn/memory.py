import numpy as np


class Memory(object):
    def __init__(
            self,
            obs_dim,
            act_dim,
            obs_dtype=np.float32,
            act_dtype=np.int32,
            rew_dtype=np.float32,
            capacity=1000,
            random_seed=0
    ):

        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.b_obs = np.zeros(
            (capacity, obs_dim),
            dtype=obs_dtype
        )
        self.b_act = np.zeros(
            (capacity,),
            dtype=act_dtype
        )
        self.b_rew = np.zeros(
            (capacity,),
            dtype=rew_dtype
        )
        self.b_done = np.zeros(
            (capacity,),
            dtype=np.bool_
        )

        self.obs_dtype = obs_dtype,
        self.act_dtype = act_dtype,
        self.rew_dtype = rew_dtype,
        self.bottom = 0
        self.top = 0
        self.size = 0

        self.rng = np.random.RandomState(random_seed)

    def __len__(self):
        return self.size

    def remember(self, obs, act, rew, done):
        self.b_obs[self.top] = obs
        self.b_act[self.top] = act
        self.b_rew[self.top] = rew
        self.b_done[self.top] = done
        self._adjust_size()
        self._increment_top()

    def random_recall(self, batch_size):
        b_obs = np.zeros(
            (batch_size, self.obs_dim),
            dtype=self.obs_dtype[0]
        )
        b_act = np.zeros(
            (batch_size,),
            dtype=self.act_dtype[0]
        )
        b_rew = np.zeros(
            (batch_size,),
            dtype=self.rew_dtype[0]
        )
        b_done = np.zeros(
            (batch_size,),
            dtype=np.bool_
        )
        b_obs_ = np.zeros(
            (batch_size, self.obs_dim),
            dtype=self.obs_dtype[0]
        )

        for count in range(batch_size):
            idx = self.rng.randint(
                self.bottom,
                self.bottom + self.size
            )

            next_index = idx + 1

            if self.b_done.take(idx, mode='wrap'):
                b_obs_[count] = self.b_obs.take(next_index, axis=0, mode='wrap')
            else:
                b_obs_[count] = self.b_obs.take(next_index, axis=0, mode='wrap')

            b_obs[count] = self.b_obs.take(idx, axis=0, mode='wrap')
            b_act[count] = self.b_act.take(idx, axis=0, mode='wrap')
            b_rew[count] = self.b_rew.take(idx, mode='wrap')
            b_done[count] = self.b_done.take(idx, mode='wrap')

        return dict(
            b_obs=b_obs,
            b_act=b_act,
            b_rew=b_rew,
            b_obs_=b_obs_,
            b_done=b_done
        )

    def _increment_top(self):
        self.top = (self.top + 1) % self.capacity

    def _adjust_size(self):
        if self.size == self.capacity:
            self.bottom = (self.bottom + 1) % self.capacity
        else:
            self.size += 1


def main():
    m = Memory(10, 10)
    m.remember([0] * 10, [0] * 10, 1, False)
    m.remember([1] * 10, [1] * 10, 1, False)
    m.remember([2] * 10, [2] * 10, 1, False)
    print(m.random_recall(2))


if __name__ == '__main__':
    main()
