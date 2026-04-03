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
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 7, 256)
        self.out = nn.Linear(256, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = Connect4Net().to(self.device)
        self.target_net = Connect4Net().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99995
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
                state_tensor = state
                q_value = self.policy_net(state_tensor)[0].cpu().numpy()
            max_q = -math.inf
            best_action = valid_locations[0]
            for c in valid_locations:
                q = q_value[c]
                if q > max_q:
                    max_q = q
                    best_action = c

            return best_action

    def learn(self, memory, batch_size):
        if len(memory) < batch_size:
            return

        batch = memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.cat(states, dim=0).to(torch.float32).to(self.device)
        next_states = torch.cat(next_states, dim=0).to(torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        current_q_value = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            # next_states shape: [64, 2, 6, 7]
            invalid_move_mask = (next_states[:, 0, 5, :] != 0) | (next_states[:, 1, 5, :] != 0)
            next_q_values[invalid_move_mask] = -1e9
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)

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
