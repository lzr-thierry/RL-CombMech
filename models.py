import torch
import torch.nn as nn
import torch.nn.functional as F


# start to define the network...
class SL_Network(nn.Module):
    def __init__(self, inputs_shape, outputs_shape):
        super(SL_Network, self).__init__()
        self.linear = nn.Linear(inputs_shape, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, outputs_shape)

        # init the networks....
        self.linear4.weight.data.mul_(0.1)
        self.linear4.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class Local_Network(nn.Module):
    def __init__(self, inputs_shape, outputs_shape):
        super(Local_Network, self).__init__()
        self.linear = nn.Linear(inputs_shape, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, outputs_shape)

        # init the networks....
        self.linear4.weight.data.mul_(0.1)
        self.linear4.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class Global_Network(nn.Module):
    def __init__(self, inputs_shape, outputs_shape):
        super(Global_Network, self).__init__()
        self.linear = nn.Linear(inputs_shape, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, outputs_shape)

        # init the networks....
        self.linear4.weight.data.mul_(0.1)
        self.linear4.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x