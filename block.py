import torch.nn as nn
import torch
import math


class ResBlock(nn.Module):
    """Residual Block with batch normalization."""
    def __init__(self,  in_channels, out_channels):
        super(ResBlock, self).__init__()
        assert in_channels == out_channels
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels, affine=True, track_running_stats=True))

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        return self.main(x) + self.conv(x)


class AttentionModule(nn.Module):
    """Attention Module for sound classification."""
    def __init__(self,  channels):
        super(AttentionModule, self).__init__()
        self.w_q = nn.Linear(channels, channels)
        self.w_k = nn.Linear(channels, channels)
        self.w_v = nn.Linear(channels, channels)
        self.scailing_factor = math.sqrt(channels)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, feature):

        #print('feature_size : ', feature.size())
        q_result = self.w_q(feature)
        k_result = self.w_k(feature)
        v_result = self.w_v(feature)

        #print('q_result : ', q_result.size()) # k_result :  torch.Size([2, 4, 512])
        #print('k_result : ', k_result.size()) # k_result :  torch.Size([2, 4, 512])
        #print('v_result : ', v_result.size()) # k_result :  torch.Size([2, 4, 512])

        k_result = k_result.permute(0, 2, 1)
        attention_map = torch.bmm(q_result, k_result)
        #print('attention_map : ', attention_map.size()) # attention_map :  torch.Size([2, 512, 512])
        attention_map = self.softmax(attention_map) / self.scailing_factor
        v_star = torch.bmm(attention_map, v_result)
        #print('v_star.size() : ', v_star.size()) # v_star.size() :  torch.Size([2, 4, 512])

        v_sum = (feature + v_star)#.permute(0, 2, 1)
        #print('v_sum.size() : ', v_sum.size()) # v_sum.size() :  torch.Size([2, 4, 512])
        return v_sum
