import numpy as np


class ReplayMemory:
    '''
    Stores transitions recorded from the agent taking actions in the environment.
    '''
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = []
        self.oldest = 0

    def sample(self, batch_size=32):
        '''
        Returns a batch of randomly sampled transitions -
        i.e. state, action, reward, next state, terminal flag tuples.
        '''
        idx = np.random.choice(len(self.memory), batch_size)
        return zip(*np.array(self.memory)[idx])

    def append(self, transition):
        '''Appends transition to the memory.'''
        if len(self.memory) < self.memory_size:
            self.memory.append(transition)
        else:
            self.memory[self.oldest] = transition
            self.oldest = (self.oldest + 1) % self.memory_size
