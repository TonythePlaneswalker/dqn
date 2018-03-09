#!/usr/bin/env python3
import argparse
from dqn_agent import DQNAgent


def main(args):
    agent = DQNAgent(args)
    if args.test:
        agent.restore(args.checkpoint)
        rewards = agent.evaluate(args.eval_episodes, args.final_epsilon)
        print('Reward mean: %f   std: %f' % (rewards.mean(), rewards.std()))
    else:
        agent.train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--video_dir', type=str)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--num_frames', type=int, default=4)
    parser.add_argument('--width', type=int, default=84)
    parser.add_argument('--height', type=int, default=84)
    parser.add_argument('--double_q', action='store_true')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--memory_size', type=int, default=50000)
    parser.add_argument('--burn_in', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_steps', type=int, default=200000)
    parser.add_argument('--lr_clip', type=float, default=0.000001)
    parser.add_argument('--init_epsilon', type=float, default=0.5)
    parser.add_argument('--final_epsilon', type=float, default=0.05)
    parser.add_argument('--epsilon_decay_steps', type=int, default=100000)
    parser.add_argument('--steps_per_eval', type=int, default=10000)
    parser.add_argument('--eval_episodes', type=int, default=20)
    parser.add_argument('--steps_per_update', type=int, default=10000)
    parser.add_argument('--max_iter', type=int, default=1000000)
    args = parser.parse_args()
    main(args)
