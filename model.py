import torch.nn as nn
import torch
import math

from block import ResBlock
from block import AttentionModule


class ResAttentionModel(nn.Module):
    def __init__(self, num_label):
        super(ResAttentionModel, self).__init__()

        start_ch = 1
        self.first_part = list()

        for _ in range(3): # length : 128-> 64-> 32 -> 16, ch 1 -> 2 -> 4 -> 8
            self.first_part.append(ResBlock(start_ch, start_ch))
            self.first_part.append(nn.Conv1d(start_ch, start_ch * 2, kernel_size=3, padding=1, stride=2))
            start_ch = 2 * start_ch

        self.first_part = nn.Sequential(*self.first_part)

        self.attention_module = AttentionModule(16)
        self.second_part = list()

        #print(start_ch)
        for _ in range(2): # length : 16 -> 8 -> 4, ch 8 -> 16 -> 32
            self.second_part.append(ResBlock(start_ch, start_ch))
            self.second_part.append(nn.Conv1d(start_ch, start_ch * 2, kernel_size=3, padding=1, stride=2))
            start_ch = 2 * start_ch

        self.second_part = nn.Sequential(*self.second_part)
        self.fc = nn.Linear(4 * 32, num_label)

    def forward(self, x):

        x = self.first_part(x)
        #print('after first_part : ', x.size())
        x = self.attention_module(x)
        #print('after attention_module : ', x.size())
        x = self.second_part(x)
        b, _, _ = x.size()
        x = x.view(b, -1)
        #print('after second_part : ', x.size())
        return self.fc(x)
