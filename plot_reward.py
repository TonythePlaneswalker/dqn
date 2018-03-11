import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str)
parser.add_argument('--csv_file', type=str)
parser.add_argument('--fig_path', type=str)
args = parser.parse_args()

x = []
y = []
with open(args.csv_file) as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        x.append(int(row['Step']))
        y.append(float(row['Value']))
plt.figure(figsize=(15, 4))
plt.plot(x, y, linewidth=2)
plt.xlabel('Step')
plt.ylabel('Reward')
if args.env_name == 'CartPole-v0':
    plt.ylim([0, 220])
elif args.env_name == 'MountainCar-v0':
    plt.ylim([-220, -100])
elif args.env_name == 'SpaceInvaders-v0':
    plt.ylim([0, 400])
plt.xticks(np.arange(11) * 100000)
plt.savefig(args.fig_path)
