import random
from simplerl.core import Experience

"""This basic experience buffer stores tuples of state, action, reward, and next state.

    It provides methods for adding new experiences, and sampling a batch of experiences, 
    as well as indexing into the buffer.
"""
class BasicExperienceBuffer():
    def __init__(self, size, batch_size):
        self.size = size
        self.batch_size = batch_size
        self.buffer = []

    def add(self, s, a, r, s_):
        self.buffer.append(Experience(s, a, r, s_))
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def sample(self):
        return random.sample(self.buffer, self.batch_size)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]


