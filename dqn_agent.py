import gym
import numpy as np
import os
import tensorflow as tf
from q_networks import dqn


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

    def evaluate(self, env_name, num_episodes, epsilon):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        env = gym.make(env_name)
        rewards = []
        for i in range(num_episodes):
            done = False
            state = env.reset()
            episode_reward = 0.
            while not done:
                action = self.policy(state, epsilon)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
            rewards.append(episode_reward)
        return rewards

    def policy(self, state, epsilon=0.05):
        q_values = self.sess.run(self.q_pred, feed_dict={self.state: [state]})
        best_action = np.argmax(q_values[0])
        u = np.random.uniform()
        if u > epsilon:
            return best_action
        else:
            return self.env.action_space.sample()

    def record(self, env_name, epsilon, video_dir):
        env = gym.make(env_name)
        env = gym.wrappers.Monitor(env, video_dir, force=True)
        done = False
        state = env.reset()
        while not done:
            action = self.policy(state, epsilon)
            next_state, reward, done, info = env.step(action)
        env.close()

    def train(self, args):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        saver = tf.train.Saver()
        steps_per_save = args.max_iter // 3
        save_path = os.path.join(args.log_dir, 'checkpoints', 'model')
        if args.restore:
            saver.restore(self.sess, tf.train.latest_checkpoint(args.log_dir))
        else:
            self.sess.run(tf.global_variables_initializer())

        rewards = self.evaluate(args.env_name, args.eval_episodes, args.final_epsilon)
        summary = self.sess.run(self.reward_summary,
                                feed_dict={self.avg_reward: np.mean(rewards)})
        self.writer.add_summary(summary, 0)
        print('Step: %d    Average reward: %f' % (0, np.mean(rewards)))

        video_num = 0
        steps_per_capture = args.max_iter // 3
        self.record(args.env_name, args.final_epsilon,
                    os.path.join(args.video_dir, '0'))

        i = 0
        state = self.env.reset()
        while i < args.max_iter:
            epsilon = self.sess.run(self.epsilon)
            action = self.policy(state, epsilon)
            next_state, reward, done, info = self.env.step(action)
            target_value = self.sess.run(self.target_value,
                                         feed_dict={self.reward: [reward],
                                                    self.next_state: [next_state],
                                                    self.is_terminal: [done]})
            _, loss, summary = self.sess.run([self.train_op, self.loss, self.loss_summary],
                                             feed_dict={self.state: [state],
                                                        self.target: target_value,
                                                        self.action: [action]})
            i += 1
            self.writer.add_summary(summary, i)
            if done:
                state = self.env.reset()
            else:
                state = next_state
            if i % args.steps_per_eval == 0:
                rewards = self.evaluate(args.env_name, args.eval_episodes, args.final_epsilon)
                summary = self.sess.run(self.reward_summary,
                                        feed_dict={self.avg_reward: np.mean(rewards)})
                self.writer.add_summary(summary, i)
                print('Step: %d    Average reward: %f' % (i, np.mean(rewards)))
            if i % steps_per_capture == 0:
                video_num += 1
                self.record(args.env_name, args.final_epsilon,
                            os.path.join(args.video_dir, str(video_num)))
            if i % steps_per_save == 0:
                saver.save(self.sess, save_path, self.global_step)
        saver.save(self.sess, save_path, self.global_step)

class ReplayDQNAgent(DQNAgent):
    # def __init__(self, environment_name, gamma):
    #     super(ReplayDQNAgent, self).__init__(environment_name, gamma)

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.

        pass
