"""Versioned game and network configuration shared by every backend."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    rows: int = 6
    cols: int = 7
    connect: int = 4
    channels: int = 32
    blocks: int = 2

    def __post_init__(self):
        if min(self.rows, self.cols, self.channels) < 1 or self.blocks < 0:
            raise ValueError("Dimensions/channels must be positive; blocks must be nonnegative")
        if not 2 <= self.connect <= max(self.rows, self.cols):
            raise ValueError("connect must be between 2 and the longest board dimension")
