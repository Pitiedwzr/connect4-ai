import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
from collections import deque

class DuelingConnect4Net(nn.Module):
    def __init__(self):
        super(DuelingConnect4Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.val_fc = nn.Linear(128 * 6 * 7, 128)
        self.val_out = nn.Linear(128, 1)

        self.adv_fc = nn.Linear(128 * 6 * 7, 128)
        self.adv_out = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)

        val = F.relu(self.val_fc(x))
        val = self.val_out(val)

        adv = F.relu(self.adv_fc(x))
        adv = self.adv_out(adv)

        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q

class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingConnect4Net().to(self.device)
        self.target_net = DuelingConnect4Net().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9999
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003)

    def act(self, state_tensor, valid_locations, board=None, my_piece=None, opp_piece=None, eval_mode=False):
        # 1-ply tactical check: take immediate winning move or block opponent's winning move
        if board is not None and my_piece is not None and opp_piece is not None:
            from game import get_immediate_winning_move, is_suicide_move
            win_move = get_immediate_winning_move(board, my_piece)
            if win_move is not None:
                return win_move
            block_move = get_immediate_winning_move(board, opp_piece)
            if block_move is not None and block_move in valid_locations:
                return block_move

            safe_moves = []
            for col in valid_locations:
                if not is_suicide_move(board, col, my_piece, opp_piece):
                    safe_moves.append(col)

            if len(safe_moves) > 0:
                valid_locations = safe_moves

        current_eps = 0.0 if eval_mode else self.epsilon
        if random.random() < self.epsilon:
            # Exploration
            action = random.choice(valid_locations)
            return action
        else:
            # Exploitation
            self.policy_net.eval()
            with torch.no_grad():
                state_tensor = state_tensor.to(self.device)
                q_values = self.policy_net(state_tensor)[0].cpu().numpy()
            self.policy_net.train()
            best_action = valid_locations[0]
            max_q = -math.inf
            for c in valid_locations:
                if q_values[c] > max_q:
                    max_q = q_values[c]
                    best_action = c
            return best_action

    def update_target_network(self, tau=0.01):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

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
            # next_states shape: [64, 2, 6, 7]
            invalid_move_mask = (next_states[:, 0, 5, :] != 0) | (next_states[:, 1, 5, :] != 0)

            policy_next_q = self.policy_net(next_states)
            policy_next_q[invalid_move_mask] = -1e9
            next_actions = policy_next_q.argmax(dim=1, keepdim=True)

            target_next_q = self.target_net(next_states)
            max_next_q_values = target_next_q.gather(1, next_actions)

            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        loss = F.smooth_l1_loss(current_q_value, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

class ReplayMemory():
    def __init__(self, capacity=50000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def push_with_symmetry(self, state, action, reward, next_state, done):
        """Saves both the original and horizontally mirrored transition."""
        self.push(state, action, reward, next_state, done)
        state_flip = torch.flip(state, dims=[3])
        next_state_flip = torch.flip(next_state, dims=[3])
        action_flip = 6 - action
        self.push(state_flip, action_flip, reward, next_state_flip, done)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
