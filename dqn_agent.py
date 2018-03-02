import gym
import numpy as np
import os
import tensorflow as tf
from q_networks import dqn


# def get_learning_rate(base_lr, ):
#     tf.train.exponential_decay(base_lr, global_step,
#                                lr_decay_steps, lr_decay_rate,
#                                staircase=True, name='learning_rate')


# def get_epsilon(initial, final, steps):
    # tf.train.polynomial_decay()


class DQNAgent:

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, args):
        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.env = gym.make(args.env_name)
        self.state = tf.placeholder(tf.float32,
                                    shape=(None,) + self.env.observation_space.shape,
                                    name='state')
        self.q_pred = dqn(self.state, self.env.action_space.n)

        self.reward = tf.placeholder(tf.float32, shape=(None,), name='reward')
        self.next_state = tf.placeholder(tf.float32,
                                         shape=(None,) + self.env.observation_space.shape,
                                         name='next_state')
        self.is_terminal = tf.placeholder(tf.float32, shape=(None,), name='is_terminal')
        self.q_target = dqn(self.next_state, self.env.action_space.n)
        self.target_value = self.reward + args.gamma * tf.reduce_max(self.q_target) * self.is_terminal
        self.target = tf.placeholder(tf.float32, shape=(None,), name='target')
        self.action = tf.placeholder(tf.int32, shape=(None,), name='action')
        self.loss = tf.reduce_mean((self.target - tf.gather(self.q_pred, self.action, axis=1)) ** 2)

        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.avg_reward = tf.placeholder(tf.float32, shape=(), name='avg_reward')
        self.reward_summary = tf.summary.scalar('average reward', self.avg_reward)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(args.base_lr, self.global_step,
                                                   args.lr_decay_steps, args.lr_decay_rate,
                                                   staircase=True, name='learning_rate')
        learning_rate = tf.maximum(learning_rate, args.lr_clip)
        trainer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = trainer.minimize(self.loss, self.global_step)

        self.epsilon = tf.train.polynomial_decay(args.init_epsilon, self.global_step,
                                                 args.epsilon_decay_steps, args.final_epsilon)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)

        saver = tf.train.Saver()
        if args.restore:
            saver.restore(self.sess, tf.train.latest_checkpoint(args.log_dir))
        else:
            self.sess.run(tf.global_variables_initializer())
        self.save_path = os.path.join(args.log_dir, 'checkpoints', 'model')

    def policy(self, q_values, epsilon=0.05):
        best_action = np.argmax(q_values)
        u = np.random.uniform()
        if u > epsilon:
            return best_action
        else:
            return self.env.action_space.sample()

    def evaluate(self, env, step, num_episodes, epsilon):
        total_reward = 0.
        for i in range(num_episodes):
            done = False
            state = env.reset()
            episode_reward = 0.
            while not done:
                q_values = self.sess.run(self.q_pred, feed_dict={self.state: np.expand_dims(state, 0)})
                action = self.policy(q_values[0], epsilon)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
            total_reward += episode_reward
        avg_reward = total_reward / num_episodes
        summary = self.sess.run(self.reward_summary, feed_dict={self.avg_reward: avg_reward})
        self.writer.add_summary(summary, step)
        print('Step: %d    Average reward: %f' % (step, avg_reward))

    def train(self, args):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        i = 0
        state = self.env.reset()
        while i < args.max_iter:
            i += 1
            q_values, epsilon = self.sess.run([self.q_pred, self.epsilon],
                                              feed_dict={self.state: [state]})
            action = self.policy(q_values[0], epsilon)
            next_state, reward, done, info = self.env.step(action)
            target_value = self.sess.run(self.target_value,
                                         feed_dict={self.reward: [reward],
                                                    self.next_state: [next_state],
                                                    self.is_terminal: [done]})
            _, loss, summary = self.sess.run([self.train_op, self.loss, self.loss_summary],
                                             feed_dict={self.state: [state],
                                                        self.target: target_value,
                                                        self.action: [action]})
            self.writer.add_summary(summary, i)
            if done:
                state = self.env.reset()
            else:
                state = next_state
            if i % args.steps_per_eval == 0:
                env = gym.make(args.env_name)
                self.evaluate(env, i, args.eval_episodes, args.final_epsilon)
            if i % args.steps_per_save == 0:
                saver = tf.train.Saver()
                saver.save(self.sess, self.save_path, self.global_step)

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        pass


class ReplayDQNAgent(DQNAgent):
    # def __init__(self, environment_name, gamma):
    #     super(ReplayDQNAgent, self).__init__(environment_name, gamma)

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.

        pass
