import random
from collections import namedtuple

class BasicExperienceBuffer():
    def __init__(self, size, batch_size):
        self.size = size
        self.batch_size = batch_size
        self.buffer = []

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def sample(self):
        return random.sample(self.buffer, self.batch_size)