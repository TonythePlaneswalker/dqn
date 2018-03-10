import numpy as np
from skimage import color, transform


class AtariWrapper:
    def __init__(self, env, num_frames, input_size):
        self.env = env
        self.action_space = self.env.action_space
        self.input_size = input_size
        self.num_frames = num_frames
        self.frames = []

    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        self.frames = self.frames[1:] + [color.rgb2gray(transform.resize(frame, self.input_size))]
        next_state = np.stack(self.frames, axis=2)
        return next_state, reward, done, info

    def reset(self):
        state = self.env.reset()
        self.frames = [color.rgb2gray(transform.resize(state, self.input_size))]
        for j in range(self.num_frames - 1):
            state, reward, done, info = self.env.step(self.env.action_space.sample())
            self.frames.append(color.rgb2gray(transform.resize(state, self.input_size)))
        return np.stack(self.frames, axis=2)
