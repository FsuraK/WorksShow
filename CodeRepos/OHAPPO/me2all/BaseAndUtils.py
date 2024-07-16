import torch
import torch.nn as nn


class net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden1_dim = 128
        self.hidden2_dim = 128
        self.linear1 = nn.Linear(self.input_dim, self.hidden1_dim)
        self.linear2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        self.linear3 = nn.Linear(self.hidden2_dim, self.output_dim)
        self.linear1.weight.data.normal_(0, 0.1)
        self.linear2.weight.data.normal_(0, 0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.tanh(self.linear3(x))
        return x


class ReplyBuffer():
    pass