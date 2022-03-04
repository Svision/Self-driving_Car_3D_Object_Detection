from torch import Tensor, nn


class SELayer(nn.Module):
    # ref: https://github.com/hujie-frank/SENet
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Residual Block.

    References:
        He et al. (2015). Deep Residual Learning for Image Recognition.
            https://arxiv.org/abs/1512.03385
    """

    def __init__(self, channels: int) -> None:
        """Initialization.

        Args:
            channels: Number of input/output channels.
        """
        super(ResidualBlock, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

        self.se = SELayer(channels, 8)

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass of the residual block.

        Args:
            x: A batch_size x C x H x W tensor.

        Returns:
            A batch_size x C x H x W tensor.
        """
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.se(x)
        x = x + residual
        out = self.relu(x)
        return out
