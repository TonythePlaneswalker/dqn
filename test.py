import argparse
import gym
import numpy as np
import tensorflow as tf
from dqn_agent import DQNAgent


def main(args):
    agent = DQNAgent(args)
    saver = tf.train.Saver()
    saver.restore(agent.sess, args.checkpoint)
    rewards = agent.evaluate(args.num_episodes, args.epsilon)
    print(np.mean(rewards), np.std(rewards))
    if args.video_dir is not None:
        env = gym.make(args.env_name)
        env = gym.wrappers.Monitor(env, args.video_dir, force=True)
        done = False
        state = env.reset()
        while not done:
            action = agent.policy(state, args.epsilon)
            next_state, reward, done, info = env.step(action)
            state = next_state
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--video_dir', type=str)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--double_q', action='store_true')
    args = parser.parse_args()
    main(args)
