import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

class Connect4Net(nn.Module):
    def __init__(self):
        super(Connect4Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4)
        self.fc1 = nn.Linear(32*3*4, 64)
        self.out = nn.Linear(64, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

class DQNAgent:
    def __init__(self):
        self.policy_net = Connect4Net()
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999

    def act(self, state, valid_locations):
        if random.random() < self.epsilon:
            # Exploration
            action = random.choice(valid_locations)
            return action
        else:
            # Exploitation
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
                q_value = self.policy_net(state_tensor)
                q_list = q_value[0].numpy()
                max_q = -math.inf
                best_action = valid_locations[0]
                for c in valid_locations:
                    q = q_list[c]
                    if q > max_q:
                        max_q = q
                        best_action = c

                return best_action

