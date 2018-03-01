import argparse
import gym


def main(args):
    # Make the environment
    env = gym.make(args.env_name)

    # Record the environment
    env = gym.wrappers.Monitor(env, 'recordings', force=True)

    for episode in range(10):
        done = False
        obs = env.reset()
        print(obs)

        while not done:  # Start with while True
            env.render()

            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env_name', type=str)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    args = parser.parse_args()
    main(args)
