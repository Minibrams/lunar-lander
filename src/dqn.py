import torch as tt
import torch.nn as nn
import torch.nn.functional as ff
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, learning_rate, state_size, hidden_1_dims, hidden_2_dims,
                 n_actions):
        super(DQN, self).__init__()

        self.input_dims = state_size
        self.hidden_1_dims = hidden_1_dims
        self.hidden_2_dims = hidden_2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.hidden_1_dims)
        self.fc2 = nn.Linear(self.hidden_1_dims, self.hidden_2_dims)
        self.fc3 = nn.Linear(self.hidden_2_dims, self.n_actions)

        self.loss = nn.HuberLoss()
        self.device = 'cuda' if tt.cuda.is_available() else 'cpu'
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        print(f'Sending DQN to device {self.device}')
        self.to(self.device)

    def forward(self, state):
        x = ff.relu(self.fc1(state))
        x = ff.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

