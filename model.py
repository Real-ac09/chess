import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    """
    SE Block: Reweights channels to emphasize important features globally.
    Cost: negligible. Gain: +30-50 Elo.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # Global Average Pooling: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Apply SE before adding residual
        out += residual
        out = self.relu(out)
        return out

class ChessNet(nn.Module):
    def __init__(self, num_blocks=10, channels=256):
        super().__init__()
        # Input: 19 planes (P1, P2, Rep, Color, Castling)
        self.conv_input = nn.Sequential(
            nn.Conv2d(19, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        # Tower of Residual Blocks
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])
        
        # Policy Head (Action prediction)
        # Reduces 256 channels -> 2 channels -> Flatten -> 4672 moves (or 4096 simplified)
        # We use 4096 (64*64) to match dataset.py
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1), # Reduce depth
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4096) 
        )
        
        # Value Head (Win/Loss prediction)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1), # Reduce to 1 channel
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh() # Output range [-1, 1]
        )

    def forward(self, x):
        x = self.conv_input(x)
        for block in self.res_blocks:
            x = block(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
