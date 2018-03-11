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
    parser.add_argument('--env_name', type=str, help='name of the gym environment')
    parser.add_argument('--model_file', type=str, help='name of the model to use')
    parser.add_argument('--log_dir', type=str, help='log directory where the checkpoints and summaries are saved.')
    parser.add_argument('--test', action='store_true', help='turn on test mode')
    parser.add_argument('--restore', action='store_true', help='restore from previous checkpoint')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path')
    parser.add_argument('--record', action='store_true', help='turn on video recording')
    parser.add_argument('--video_dir', type=str, help='directory to put recorded video')
    parser.add_argument('--gamma', type=float, default=1.0, help='discount factor')
    parser.add_argument('--init_epsilon', type=float, default=0.5, help='initial epsilon')
    parser.add_argument('--final_epsilon', type=float, default=0.05, help='final epsilon (used for evaluation)')
    parser.add_argument('--epsilon_decay_steps', type=int, default=100000,
                        help='number of steps over which epsilon is decayed from init_epsilon to final_epsilon')
    parser.add_argument('--replay', action='store_true', help='use replay memory')
    parser.add_argument('--memory_size', type=int, default=50000, help='maximum size of the replay memory')
    parser.add_argument('--burn_in', type=int, default=10000, help='initial size of the replay memory')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--fix_target', action='store_true', help='use fixed target network')
    parser.add_argument('--steps_per_update', type=int, default=10000,
                        help='number of steps per update of the target network')
    parser.add_argument('--double_q', action='store_true', help='use double q-learning')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='learning rate decay rate')
    parser.add_argument('--lr_decay_steps', type=int, default=200000, help='steps per learning rate decay')
    parser.add_argument('--lr_clip', type=float, default=0.000001, help='minimum learning rate')
    parser.add_argument('--grad_clip', type=float, help='maximum gradient')
    parser.add_argument('--steps_per_eval', type=int, default=10000, help='steps per evaluation')
    parser.add_argument('--eval_episodes', type=int, default=20, help='number of evaluation episodes')
    parser.add_argument('--max_iter', type=int, default=1000000, help='number of iterations for training')
    parser.add_argument('--num_frames', type=int, default=4,
                        help='number of frames that are stacked for input to the convolutinal network')
    parser.add_argument('--width', type=int, default=84, help='width for resized image (for convnet input)')
    parser.add_argument('--height', type=int, default=84, help='width for resized image (for convnet input)')
    args = parser.parse_args()
    main(args)
