import collections
import numpy as np

Experience = collections.namedtuple('Experience',
                                    field_names=['state', 'action', 'reward', 'next_state', 'done'])


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop()
        self.buffer.append(experience)

    def sample(self, batch_size, device):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        zipped = list(zip(*[self.buffer[i] for i in indices]))
        return zipped
