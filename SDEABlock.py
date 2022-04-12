import torch
from torch import nn


class SDEABlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, maxdisp: int, stride=1):
        super().__init__()
        self.maxdisp = maxdisp
        self.g1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.g2 = nn.Conv2d(out_channels, 1, 1, 1, 0)

        self.channel_scaling = None
        if in_channels != out_channels:
            self.channel_scaling = nn.Conv2d(in_channels, out_channels, stride, 1)

    def forward(self, left, right):
        # two convolution layers G1
        left_g1 = self.g1(left)
        right_g1 = self.g1(right)

        # scaling the input channels
        if self.channel_scaling:
            left = self.channel_scaling(left)
            right = self.channel_scaling(right)

        # 1x1 convolution layer G2
        left_g2 = self.g2(left_g1)
        right_g2 = self.g2(right_g1)

        # Weight calculation
        left_weight = weight_matrix_calculation(left_g2, right_g2, self.maxdisp)
        right_weight = weight_matrix_calculation(right_g2, left_g2, self.maxdisp)

        # Elementwise multiplication
        left_out = left_g1 * left_weight
        right_out = right_g1 * right_weight

        # Elementwise sum
        left_out = left_out + left
        right_out = right_out + right

        return left_out, right_out


@torch.jit.script
def weight_matrix_calculation(left, right, maxdisp: int):
    """
    Then we are on one G2, for each point x on G2,
    we find the point x with the minimum difference
    on the other G2 in themax-disp range
    """

    weight_volume = torch.empty_like(left)

    for j in range(left.shape[3]):
        left_bound = max(0, j - maxdisp)
        right_bound = min(left.shape[3], j + maxdisp)
        diff = right[:, 0, :, left_bound:right_bound] - left[:, 0, :, j : j + 1]
        diff = torch.abs(diff)
        v, _ = diff.min(2)
        weight_volume[:, 0, :, j] = v
    weight_volume = 1 - torch.sigmoid(weight_volume)
    return weight_volume
