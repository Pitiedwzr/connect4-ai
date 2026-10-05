import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
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
    def __init__(self, accelerator=None, device=None):
        self.accelerator = accelerator
        self.device = (accelerator.device if accelerator is not None else
                       torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))
        self.policy_net = DuelingConnect4Net().to(self.device)
        self.target_net = DuelingConnect4Net().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9997
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003)

    def _get_unwrapped_policy(self):
        """Returns the underlying model without DDP wrapper."""
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.policy_net)
        if hasattr(self.policy_net, "module"):
            return self.policy_net.module
        return self.policy_net

    def act(self, state_tensor, valid_locations, board=None, my_piece=None, opp_piece=None, eval_mode=False):
        from game import get_candidate_moves
        if board is not None and my_piece is not None and opp_piece is not None:
            candidates = get_candidate_moves(board, my_piece, opp_piece)
        else:
            candidates = policy_candidates_from_state(state_tensor)
        valid_locations = [c for c in candidates if c in valid_locations]
        if not valid_locations:
            return None
        if len(valid_locations) == 1:
            return valid_locations[0]
        current_eps = 0.0 if eval_mode else self.epsilon
        if random.random() < current_eps:
            # Exploration
            return random.choice(valid_locations)
        else:
            # Exploitation: evaluate with unwrapped model in eval mode
            unwrapped = self._get_unwrapped_policy()
            unwrapped.eval()
            with torch.no_grad():
                state_tensor = state_tensor.to(self.device)
                q_values = unwrapped(state_tensor)[0].cpu().numpy()

            return max(valid_locations, key=lambda c: q_values[c])

    def update_target_network(self, tau=0.01):
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        unwrapped = self._get_unwrapped_policy()
        if tau >= 1.0:
            self.target_net.load_state_dict(unwrapped.state_dict())
        else:
            with torch.no_grad():
                for target_param, policy_param in zip(self.target_net.parameters(), unwrapped.parameters()):
                    target_param.lerp_(policy_param, tau)
                # Running statistics describe the policy's current activations;
                # copy these buffers rather than combining weights with stale BN.
                for target_buffer, policy_buffer in zip(self.target_net.buffers(), unwrapped.buffers()):
                    target_buffer.copy_(policy_buffer)

    def learn(self, memory, batch_size):
        if len(memory) < batch_size:
            return

        batch = memory.sample(batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        states = torch.cat(states, dim=0).to(torch.float32).to(self.device)
        next_states = torch.cat(next_states, dim=0).to(torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_masks = torch.stack(next_masks).to(self.device)

        # 1. Main forward pass (train mode)
        self.policy_net.train()
        current_q_value = self.policy_net(states).gather(1, actions)

        # 2. Target Q calculation
        with torch.no_grad():
            next_values = torch.zeros_like(rewards)
            continuing = (dones[:, 0] == 0) & next_masks.any(dim=1)
            unwrapped = self._get_unwrapped_policy()
            unwrapped.eval()
            if continuing.any():
                next_batch = next_states[continuing]
                policy_next_q = unwrapped(next_batch)
                policy_next_q = policy_next_q.masked_fill(~next_masks[continuing], -torch.inf)
                next_actions = policy_next_q.argmax(dim=1, keepdim=True)
                target_next_q = self.target_net(next_batch)
                # Only terminal +/-1 rewards exist; bound approximate bootstrap
                # values to the feasible return range to prevent runaway targets.
                next_values[continuing] = target_next_q.gather(1, next_actions).clamp(-1.0, 1.0)
            target_q_values = rewards + self.gamma * next_values

        # 3. Loss & Backward
        loss = F.smooth_l1_loss(current_q_value, target_q_values)
        self.optimizer.zero_grad()

        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        if self.accelerator is not None:
            self.accelerator.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        else:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()


def policy_candidates_from_state(state):
    """Decode an own/opponent tensor for the same policy used during play."""
    from game import get_candidate_moves
    channels = state.detach().cpu()[0].numpy()
    board = (channels[0] != 0).astype("int64") + 2 * (channels[1] != 0)
    return get_candidate_moves(board, 1, 2)

class ReplayMemory():
    def __init__(self, capacity=50000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask=None):
        if next_mask is None:
            next_mask = torch.zeros(7, dtype=torch.bool)
            if not done:
                next_mask[policy_candidates_from_state(next_state)] = True
        self.memory.append((state, action, reward, next_state, done, next_mask))

    def push_with_symmetry(self, state, action, reward, next_state, done):
        """Saves both the original and horizontally mirrored transition."""
        self.push(state, action, reward, next_state, done)
        state_flip = torch.flip(state, dims=[3])
        next_state_flip = torch.flip(next_state, dims=[3])
        action_flip = 6 - action
        self.push(state_flip, action_flip, reward, next_state_flip, done,
                  torch.flip(self.memory[-1][5], dims=[0]))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
