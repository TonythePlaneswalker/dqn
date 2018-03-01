import gym
import numpy as np
import tensorflow as tf
from q_networks import dqn


class OnlineAgent:

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, gamma):
        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.env = gym.make(environment_name)
        self.state = tf.placeholder(tf.float32,
                                    shape=self.env.observation_space.shape,
                                    name='state')
        self.q_values = dqn(self.state, self.env.action_space.n)

        self.reward = tf.placeholder(tf.bool, shape=(), name='reward')
        self.next_state = tf.placeholder(tf.float32,
                                         shape=self.env.observation_space.shape,
                                         name='next_state')
        self.is_terminal = tf.placeholder(tf.bool, shape=(), name='is_terminal')
        q_next = dqn(self.next_state, self.env.action_space.n)
        self.target_value = tf.cond(self.is_terminal,
                                    lambda: self.reward,
                                    lambda: self.reward + gamma * tf.reduce_max(q_next))
        self.target = tf.placeholder(tf.float32, shape=(), name='target')
        self.action = tf.placeholder(tf.int32, shape=(), name='action')
        self.loss = (self.target - tf.gather(self.q_values, self.action)) ** 2

    def policy(self, q_values, epsilon=0.05):
        best_action = np.argmax(q_values)
        u = np.random.uniform()
        if u > epsilon:
            return best_action
        else:
            return self.env.action_space.sample()

    def train(self):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        trainer = tf.train.AdamOptimizer(0.0001)
        train_op = trainer.minimize(self.loss)

        i = 0
        state = self.env.reset()
        while i < 10:
            q_values = sess.run(self.q_values, feed_dict={self.state: state})
            action = self.policy(q_values)
            obs, reward, done, info = self.env.step(action)
            target_value = sess.run(self.target_value,
                                    feed_dict={self.reward: reward,
                                               self.next_state: obs,
                                               self.is_terminal: done})
            _, loss = sess.run([train_op, self.loss],
                               feed_dict={self.target: target_value,
                                          self.action: action})

    def test(self, model_file=None):

    # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
    # Here you need to interact with the environment, irrespective of whether you are using a memory.


class ReplayAgent(OnlineAgent):
    def burn_in_memory:
        # Initialize your replay memory with a burn_in number of episodes / transitions.

        pass
