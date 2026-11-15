"""EEG-SimpleConv, ported from https://github.com/elouayas/EEGSimpleConv (commit a732d9a).

Module names match the official code. Changes: single logit output (binary task), no subject head.
The official block condition `i > 0 - 1` is always true, so every block max-pools by 2.
"""
import torch
import torch.nn as nn
from torchaudio.transforms import Resample


class EEGSimpleConv(nn.Module):
    def __init__(self, n_channels, sfreq, fm, n_convs, resampling, kernel_size):
        super().__init__()
        self.rs = Resample(orig_freq=sfreq, new_freq=resampling)
        self.conv = nn.Conv1d(n_channels, fm, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(fm)
        blocks = []
        old_fm = new_fm = fm
        for i in range(n_convs):
            if i > 0:
                new_fm = int(1.414 * new_fm)
            blocks.append(nn.Sequential(
                nn.Conv1d(old_fm, new_fm, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(new_fm),
                nn.MaxPool1d(2),
                nn.ReLU(),
                nn.Conv1d(new_fm, new_fm, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(new_fm),
                nn.ReLU(),
            ))
            old_fm = new_fm
        self.blocks = nn.ModuleList(blocks)
        self.fc = nn.Linear(old_fm, 1)

    def forward(self, x):  # x: (B, C, T) -> logits (B,)
        y = torch.relu(self.bn(self.conv(self.rs(x))))
        for block in self.blocks:
            y = block(y)
        return self.fc(y.mean(dim=2)).squeeze(-1)
