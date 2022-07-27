from inspect import isabstract

from numpy import size
from simplerl.core import Agent, Policy
from simplerl.buffers import BasicExperienceBuffer

class TestCore:
    def test_abstracts(self):
        pass
        # assert isabstract(Agent)
        # assert isabstract(MonoTrainer) == False
    def test_buffers(self):
        exp_buffer = BasicExperienceBuffer(size=10, batch_size=5)
