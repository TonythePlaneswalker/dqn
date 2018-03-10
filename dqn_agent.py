import gym
import importlib
import numpy as np
import os
import tensorflow as tf
from atari_wrapper import AtariWrapper
from replay_memory import ReplayMemory


class DQNAgent:
    '''An agent that learns an epsilon-greedy policy using Q-learning.'''
    def __init__(self, args):
        # Set up environment and network input
        self.env = gym.make(args.env_name)
        if args.record:
            self.env = gym.wrappers.Monitor(self.env, args.video_dir, force=True)
        if args.env_name == 'SpaceInvaders-v0':
            self.env = AtariWrapper(self.env, args.num_frames, (args.width, args.height))
            self.state = tf.placeholder(tf.float32,
                                        shape=(None, args.width, args.height, args.num_frames),
                                        name='state')
        else:
            self.state = tf.placeholder(tf.float32,
                                        shape=(None,) + self.env.observation_space.shape,
                                        name='state')
        self.next_state = tf.placeholder(tf.float32, shape=self.state.get_shape(), name='next_state')

        # Build prediction and target networks
        model = importlib.import_module(args.model_file)
        self.q_pred = model.get_model(self.state, self.env.action_space.n, scope='q_pred')
        if args.fix_target:
            self.q_target = model.get_model(self.next_state, self.env.action_space.n, scope='q_target')
            self.update_q_target = [tf.assign(w_target, w) for w_target, w in zip(
                                    tf.trainable_variables(scope='q_target'),
                                    tf.trainable_variables(scope='q_pred'))]
        else:
            self.q_target = tf.placeholder(tf.float32, shape=(None, self.env.action_space.n), name='q_target')

        # Calculate TD target
        self.action = tf.placeholder(tf.int32, shape=(None,), name='action')
        self.reward = tf.placeholder(tf.float32, shape=(None,), name='reward')
        self.is_terminal = tf.placeholder(tf.float32, shape=(None,), name='is_terminal')
        self.q_next = tf.placeholder(tf.float32, shape=(None, self.env.action_space.n), name='q_next')
        if args.double_q:
            q_eval = tf.diag_part(tf.gather(self.q_target, tf.argmax(self.q_next, axis=1), axis=1))
        else:
            q_eval = tf.reduce_max(self.q_target, axis=1)
        target = self.reward + args.gamma * q_eval * (1 - self.is_terminal)
        self.loss = tf.reduce_mean((target - tf.diag_part(tf.gather(self.q_pred, self.action, axis=1))) ** 2)

        # Set up session for training/testing
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

    def evaluate(self, num_episodes, epsilon):
        rewards = np.zeros(num_episodes)
        for i in range(num_episodes):
            done = False
            state = self.env.reset()
            episode_reward = 0
            while not done:
                action = self.policy(state, epsilon)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                state = next_state
            rewards[i] = episode_reward
        return rewards

    def policy(self, state, epsilon):
        q_values = self.sess.run(self.q_pred, feed_dict={self.state: [state]})
        best_action = np.argmax(q_values[0])
        u = np.random.rand()
        if u > epsilon:
            return best_action
        else:
            return self.env.action_space.sample()

    def train(self, args):
        # Set up training parameters and trainer
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(args.base_lr, global_step,
                                                   args.lr_decay_steps, args.lr_decay_rate,
                                                   staircase=True, name='lr')
        learning_rate = tf.maximum(learning_rate, args.lr_clip)
        train_epsilon = tf.train.polynomial_decay(args.init_epsilon, global_step,
                                                  args.epsilon_decay_steps, args.final_epsilon)
        trainer = tf.train.AdamOptimizer(learning_rate)
        grad = trainer.compute_gradients(self.loss, var_list=tf.trainable_variables(scope='q_pred'))
        if args.grad_clip:
            # Clip the gradients
            grad = [(tf.clip_by_value(grad, -args.grad_clip, args.grad_clip), var) for grad, var in grad]
        train_op = trainer.apply_gradients(grad, global_step)

        # Summary for tensorboard
        loss_summary = tf.summary.scalar('loss', self.loss)
        lr_summary = tf.summary.scalar('learning_rate', learning_rate)
        epsilon_summary = tf.summary.scalar('epsilon', train_epsilon)
        train_summary = tf.summary.merge([loss_summary, lr_summary, epsilon_summary])
        episode_length = tf.placeholder(tf.int32, shape=(), name='episode_length')
        length_summary = tf.summary.scalar('episode_length', episode_length)
        avg_reward = tf.placeholder(tf.float32, shape=(), name='avg_reward')
        reward_summary = tf.summary.scalar('average reward', avg_reward)

        # Set up saving and logging
        saver = tf.train.Saver(max_to_keep=10)
        save_path = os.path.join(args.log_dir, 'checkpoints', 'model')
        steps_per_save = args.max_iter // 9
        if args.restore:
            self.restore(args.checkpoint)
        else:
            self.sess.run(tf.global_variables_initializer())
            if args.fix_target:
                self.sess.run(self.update_q_target)
            if os.path.exists(args.log_dir):
                delete_key = input('%s exists. Delete? [y (or enter)/N]' % args.log_dir)
                if delete_key == 'y' or delete_key == "":
                    os.system('rm -rf %s/*' % args.log_dir)
            os.makedirs(os.path.join(args.log_dir, 'checkpoints'), exist_ok=True)
            saver.save(self.sess, save_path, global_step)
        writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)

        # Burn in some transitions using the randomly initialized agent into the replay memory
        if args.replay:
            replay = ReplayMemory(args.memory_size)
            epsilon = self.sess.run(train_epsilon)
            i = 0
            while i < args.burn_in:
                done = False
                state = self.env.reset()
                while not done and i < args.burn_in:
                    action = self.policy(state, epsilon)
                    next_state, reward, done, info = self.env.step(action)
                    replay.append((state, action, reward, next_state, done))
                    state = next_state
                    i += 1

        # Training
        i = self.sess.run(global_step)
        episode_start = i
        state = self.env.reset()
        while i < args.max_iter:
            epsilon = self.sess.run(train_epsilon)
            action = self.policy(state, epsilon)
            next_state, reward, is_terminal, info = self.env.step(action)
            if args.replay:
                replay.append((state, action, reward, next_state, is_terminal))
                states, actions, rewards, next_states, is_terminals = replay.sample(args.batch_size)
            else:
                states = [state]
                actions = [action]
                rewards = [reward]
                next_states = [next_state]
                is_terminals = [is_terminal]
            if args.fix_target:
                feed_dict = {self.state: states, self.action: actions, self.reward: rewards,
                             self.next_state: next_states, self.is_terminal: is_terminals}
            else:
                q_target = self.sess.run(self.q_pred, feed_dict={self.state: next_states})
                feed_dict = {self.state: states, self.action: actions, self.reward: rewards,
                             self.is_terminal: is_terminals, self.q_target: q_target}
            if args.double_q:
                q_next = self.sess.run(self.q_pred, feed_dict={self.state: next_states})
                feed_dict.update({self.q_next: q_next})
            _, loss, summary = self.sess.run([train_op, self.loss, train_summary], feed_dict=feed_dict)
            i += 1
            writer.add_summary(summary, i)
            if is_terminal:
                summary = self.sess.run(length_summary, feed_dict={episode_length: i - episode_start})
                writer.add_summary(summary, i)
                episode_start = i
                state = self.env.reset()
            else:
                state = next_state
            if i % args.steps_per_eval == 0:
                rewards = self.evaluate(args.eval_episodes, args.final_epsilon)
                summary = self.sess.run(reward_summary,
                                        feed_dict={avg_reward: rewards.mean()})
                writer.add_summary(summary, i)
                print('Step: %d   Average reward: %f   Loss: %f' % (i, rewards.mean(), loss))
                np.set_printoptions(precision=10)
                print('Q values:', self.sess.run(self.q_pred, feed_dict={self.state: [state]}))
                print('Rewards:', rewards)
                episode_start = i
                state = self.env.reset()
            if args.fix_target and i % args.steps_per_update == 0:
                self.sess.run(self.update_q_target)
            if i % steps_per_save == 0:
                saver.save(self.sess, save_path, global_step)

    def restore(self, checkpoint):
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint)
