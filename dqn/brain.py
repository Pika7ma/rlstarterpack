import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


class Brain(object):
    def __init__(self, mlp_units, obs_num, act_num, mlp_ac="relu", discount=0.99, cnn_units=None, mlp_opt='adam', learning_rate=0.01):
        self.mlp_units = mlp_units
        self.obs_num = obs_num
        self.act_num = act_num
        self.cnn_units = cnn_units
        self.discount = discount
        self.learning_rate = learning_rate
        self.setup_nn(mlp_ac, mlp_opt, learning_rate)

    def optimize(self, batch):
        return self.sess.run([self.optimizer], feed_dict={
            self.obs: batch["b_obs"],
            self.obs_: batch["b_obs_"],
            self.rew: batch["b_rew"],
            self.done: batch["b_done"],
            self.act: batch["b_act"]
        })

    def setup_observation(self):
        with tf.variable_scope("observation"):
            self.obs = tf.placeholder(tf.float32, shape=[None, self.obs_num])
            self.obs_ = tf.placeholder(tf.float32, shape=[None, self.obs_num])

    def setup_action(self):
        with tf.variable_scope("action"):
            self.act = tf.placeholder(tf.int32, shape=[None, ])

    def setup_optimizer(self, mlp_opt, learning_rate):
        with tf.variable_scope("optimizer"):
            optimizers = {
                "adam": tf.train.AdamOptimizer,
            }
            self.optimizer = optimizers[mlp_opt](learning_rate).minimize(self.loss)

    def setup_activation(self, mlp_ac):
        with tf.variable_scope("activation"):
            activations = {
                "relu": tf.nn.relu,
                "tanh": tf.nn.tanh,
            }
            self.mlp_ac = activations[mlp_ac]

    def setup_done(self):
        with tf.variable_scope("done"):
            self.done = tf.placeholder(tf.bool, shape=[None, ])

    def setup_reward(self):
        with tf.variable_scope("reward"):
            self.rew = tf.placeholder(tf.float32, shape=[None, ])

    def setup_param_update(self):
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evaluation')
        with tf.variable_scope('param_update'):
            self.param_update = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def eval2target(self):
        self.sess.run(self.param_update)

    def setup_nn(self, mlp_ac, mlp_opt, learning_rate):

        self.setup_observation()
        self.setup_action()
        self.setup_activation(mlp_ac)
        self.setup_done()
        self.setup_reward()

        self.q = self.mlp("evaluation", self.obs)
        self.qt = tf.stop_gradient(self.mlp("target", self.obs_))

        self.setup_labels()
        self.setup_loss()
        self.setup_optimizer(mlp_opt, learning_rate)
        self.setup_param_update()

        self.sess = tf.Session()
        summary_writer = tf.summary.FileWriter('/tmp/dqn_logs', self.sess.graph)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def setup_labels(self):
        with tf.variable_scope("labels"):
            self.labels = tf.where(self.done, tf.zeros_like(self.rew), self.discount * self.get_target())
            self.labels += self.rew

    def get_target(self):
        return tf.reduce_max(self.qt, axis=1)

    def get_evaluation(self):
        return tf.argmax(self.q, axis=1)

    def get_opt_action(self, obs):
        return self.sess.run(self.get_evaluation(), feed_dict={self.obs: obs.reshape((1, self.obs_num))})

    def get_evaluation_wrt_a(self):
        a_indices = tf.stack([tf.range(tf.shape(self.act)[0], dtype=tf.int32), self.act], axis=1)
        return tf.gather_nd(params=self.q, indices=a_indices)

    def setup_loss(self):
        with tf.variable_scope("loss"):
            self.loss = tf.losses.huber_loss(
                labels=self.labels,
                predictions=self.get_evaluation_wrt_a(),
                reduction=tf.losses.Reduction.MEAN
            )

    def mlp(self, scope, state, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            layer = state
            W_init = layers.xavier_initializer()
            b_init = tf.zeros_initializer()
            with tf.variable_scope("hidden"):
                for idx, units in enumerate(self.mlp_units):
                    layer = tf.layers.dense(
                        inputs=layer,
                        units=units,
                        kernel_initializer=W_init,
                        bias_initializer=b_init,
                        activation=self.mlp_ac,
                        reuse=reuse,
                        name=str(idx)
                    )
            with tf.variable_scope("output"):
                logits = tf.layers.dense(
                    inputs=layer,
                    units=self.act_num,
                    kernel_initializer=W_init,
                    bias_initializer=b_init,
                    activation=None,
                    reuse=reuse
                )
            return logits


def main():
    b = Brain([64, 64], 10, 10)
    b.param_update()


if __name__ == '__main__':
    main()