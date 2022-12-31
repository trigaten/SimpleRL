
from simplerl import Policy, MaxQPolicy
import torch.nn as nn
import torch

class LinearNet(nn.Module):
    """Dummy network"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 2, bias=False)

    def forward(self, x):
        return self.fc(x)

class TestMaxQPolicy:
    """Test the correctness of the MaxQPolicy class."""
    def test_output(self):
        # dummy net
        net = LinearNet()

        # set weights
        weights  = net.state_dict()
        weights['fc.weight'] = torch.tensor([[2],[-1]])
        net.load_state_dict(weights)

        # check network works properly
        assert torch.equal(net(torch.FloatTensor([[2]])), torch.FloatTensor([[ 4., -2.]]))

        # create policy
        policy = MaxQPolicy(net, 2)

        # check that policy output is correct
        assert policy(torch.FloatTensor([[1.0]])) == 0
        assert policy(torch.FloatTensor([[-1.0]])) == 1


