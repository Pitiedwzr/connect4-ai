import torch
import torch.nn as nn
import torch.nn.functional as F

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
