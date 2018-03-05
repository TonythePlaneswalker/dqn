import gym
import numpy as np
import os
import tensorflow as tf
from q_networks import dqn
from replay_memory import ReplayMemory


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
        self.q_pred = dqn(self.state, self.env.action_space.n, num_hidden=[10, 10, 10])

        self.reward = tf.placeholder(tf.float32, shape=(None,), name='reward')
        self.next_state = tf.placeholder(tf.float32,
                                         shape=(None,) + self.env.observation_space.shape,
                                         name='next_state')
        self.is_terminal = tf.placeholder(tf.float32, shape=(None,), name='is_terminal')
        self.q_target = dqn(self.next_state, self.env.action_space.n, num_hidden=[10, 10, 10])
        self.target_value = self.reward + args.gamma * tf.reduce_max(self.q_target) * (1 - self.is_terminal)
        self.target = tf.placeholder(tf.float32, shape=(None,), name='target')
        self.action = tf.placeholder(tf.int32, shape=(None,), name='action')
        self.loss = tf.reduce_mean((self.target - tf.gather(self.q_pred, self.action, axis=1)) ** 2)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

    def evaluate(self, env_name, num_episodes, epsilon):
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

    def policy(self, state, epsilon):
        q_values = self.sess.run(self.q_pred, feed_dict={self.state: [state]})
        best_action = np.argmax(q_values[0])
        u = np.random.uniform()
        if u > epsilon:
            return best_action
        else:
            return self.env.action_space.sample()

    def train(self, args):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        if args.replay:
            replay = ReplayMemory(args.memory_size, args.burn_in, args.env_name)

        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(args.base_lr, global_step,
                                                   args.lr_decay_steps, args.lr_decay_rate,
                                                   staircase=True, name='lr')
        learning_rate = tf.maximum(learning_rate, args.lr_clip)
        # learning_rate = tf.constant(args.base_lr, dtype=tf.float32, shape=(), name='learning_rate')
        train_epsilon = tf.train.polynomial_decay(args.init_epsilon, global_step,
                                                  args.epsilon_decay_steps, args.final_epsilon)

        loss_summary = tf.summary.scalar('loss', self.loss)
        lr_summary = tf.summary.scalar('learning_rate', learning_rate)
        epsilon_summary = tf.summary.scalar('epsilon', train_epsilon)
        avg_reward_train = tf.placeholder(tf.float32, shape=(), name='avg_reward_train')
        r_summary = tf.summary.scalar('training average reward', avg_reward_train)
        train_summary = tf.summary.merge([loss_summary, lr_summary, epsilon_summary])
        episode_length = tf.placeholder(tf.int32, shape=(), name='episode_length')
        length_summary = tf.summary.scalar('episode_length', episode_length)
        avg_reward = tf.placeholder(tf.float32, shape=(), name='avg_reward')
        reward_summary = tf.summary.scalar('average reward', avg_reward)
        writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)

        trainer = tf.train.AdamOptimizer(learning_rate)
        train_op = trainer.minimize(self.loss, global_step)

        saver = tf.train.Saver()
        if args.restore:
            saver.restore(self.sess, tf.train.latest_checkpoint(args.log_dir))
        else:
            self.sess.run(tf.global_variables_initializer())
        save_path = os.path.join(args.log_dir, 'checkpoints', 'model')
        saver.save(self.sess, save_path, global_step)
        steps_per_save = args.max_iter // 3

        i = 0
        episode_start = i
        avg_r_train = 0
        state = self.env.reset()
        while i < args.max_iter:
            epsilon = self.sess.run(train_epsilon)
            action = self.policy(state, epsilon)
            next_state, reward, is_terminal, info = self.env.step(action)
            if args.replay:
                replay.append((state, action, reward, next_state, is_terminal))
                states, actions, rewards, next_states, is_terminals = replay.sample(args.batch_size)
                target_values = self.sess.run(self.target_value,
                                              feed_dict={self.reward: rewards,
                                                         self.next_state: next_states,
                                                         self.is_terminal: is_terminals})
                _, loss, summary = self.sess.run([train_op, self.loss, train_summary],
                                                 feed_dict={self.state: states,
                                                            self.reward: rewards,
                                                            self.target: target_values,
                                                            self.action: actions})
            else:
                target_value = self.sess.run(self.target_value,
                                             feed_dict={self.reward: [reward],
                                                        self.next_state: [next_state],
                                                        self.is_terminal: [is_terminal]})
                _, loss, summary = self.sess.run([train_op, self.loss, train_summary],
                                                 feed_dict={self.state: [state],
                                                            self.reward: [reward],
                                                            self.target: target_value,
                                                            self.action: [action]})
            i += 1
            writer.add_summary(summary, i)
            avg_r_train += (reward - avg_r_train) / i
            summary = self.sess.run(r_summary, feed_dict={avg_reward_train: avg_r_train})
            writer.add_summary(summary, i)
            if is_terminal:
                state = self.env.reset()
                summary = self.sess.run(length_summary,
                                        feed_dict={episode_length: i - episode_start})
                writer.add_summary(summary, i)
                episode_start = i
            else:
                state = next_state
            if i % args.steps_per_eval == 0:
                rewards = self.evaluate(args.env_name, args.eval_episodes, args.final_epsilon)
                summary = self.sess.run(reward_summary,
                                        feed_dict={avg_reward: np.mean(rewards)})
                writer.add_summary(summary, i)
                print('Step: %d    Average reward: %f' % (i, np.mean(rewards)))
            if i % steps_per_save == 0:
                saver.save(self.sess, save_path, global_step)
        saver.save(self.sess, save_path, global_step)
