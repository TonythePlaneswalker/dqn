import numpy as np
from skimage import color, transform


class AtariWrapper:
    '''A wrapper for Atari game environments that resizes the frames,
       converts them into grayscale and stacks them.'''
    def __init__(self, env, num_frames, size):
        '''
        :param env: gym environment
        :param num_frames: number of frames to stack
        :param size: output size of the images
        '''
        self.env = env
        self.action_space = self.env.action_space
        self.size = size
        self.num_frames = num_frames
        self.frames = []

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.frames = self.frames[1:] + [self.process_frame(state)]
        next_state = np.stack(self.frames, axis=2)
        return next_state, reward, done, info

    def reset(self):
        state = self.env.reset()
        self.frames = [self.process_frame(state)]
        for j in range(self.num_frames - 1):
            state, reward, done, info = self.env.step(self.env.action_space.sample())
            self.frames += [self.process_frame(state)]
        return np.stack(self.frames, axis=2)

    def process_frame(self, frame):
        return (color.rgb2gray(transform.resize(frame, self.size)) * 255).astype(np.uint8)
