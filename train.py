import argparse
import os
from dqn_agent import DQNAgent


def main(args):
    if not args.restore and os.path.exists(args.log_dir):
        delete_key = input('%s exists. Delete? [y (or enter)/N]' % args.log_dir)
        if delete_key == 'y' or delete_key == "":
            os.system('rm -rf %s/*' % args.log_dir)
    os.makedirs(os.path.join(args.log_dir, 'checkpoints'), exist_ok=True)
    agent = DQNAgent(args)
    agent.train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_steps', type=int, default=200000)
    parser.add_argument('--lr_clip', type=float, default=0.000001)
    parser.add_argument('--init_epsilon', type=float, default=0.5)
    parser.add_argument('--final_epsilon', type=float, default=0.05)
    parser.add_argument('--epsilon_decay_steps', type=int, default=100000)
    parser.add_argument('--max_iter', type=int, default=1000000)
    parser.add_argument('--steps_per_eval', type=int, default=10000)
    parser.add_argument('--eval_episodes', type=int, default=20)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--memory_size', type=int, default=50000)
    parser.add_argument('--burn_in', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    main(args)
