import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.state_dim = input_dim
        self.action_dim = output_dim

        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim + output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        action = self.actor(state)
        value = self.critic(state)
        zero = torch.tensor([0.0])
        action = torch.cat((action[:1], zero, action[1:]))
        return action, value

    def get_q_value(self, state, a):
        state = torch.cat([state, a])
        value = self.critic(state)
        return value

    def get_action(self, state):
        action = self.actor(state)
        return action



class net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden1_dim = 128  # model 4 5: 128-64
        self.hidden2_dim = 64
        self.linear1 = nn.Linear(self.input_dim, self.hidden1_dim)
        self.linear2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        self.linear3 = nn.Linear(self.hidden2_dim, self.output_dim)
        # self.linear1.weight.data.normal_(0, 0.1)
        # self.linear2.weight.data.normal_(0, 0.1)
        # self.linear3.weight.data.normal_(0, 0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        # x = self.tanh(self.linear3(x))
        x = self.linear3(x)
        return x


class ReplyBuffer():
    pass


# Bidirectional recurrent neural network (many-to-one)
class BiLSTM(nn.Module):
    def __init__(self, num_classes, device, input_size=2, hidden_size=6, num_layers=1):
        super(BiLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection
        self.fc.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(self.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    input_size = 10
    hidden_size = 128
    num_layers = 2
    num_classes = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lstm = BiLSTM(input_size, hidden_size, num_layers, num_classes, device)

    x1 = torch.ones((3, 5, 1), dtype=torch.float)
    x2 = torch.ones((3, 7, 1), dtype=torch.float)

    a = lstm.forward(x1)
    b = lstm.forward(x2)
