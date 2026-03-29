import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
from collections import deque


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
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

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

    def learn(self, memory, batch_size):
        if len(memory) < batch_size:
            return

        batch = memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32).unsqueeze(1)
        donse = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        current_q_value = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q_values = self.policy_net(next_states).max(1)[0].unsqueeze(1)

        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
        loss = self.loss_fn(current_q_value, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class ReplayMemory():
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
