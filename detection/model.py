from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
import numpy as np

from detection.modules.loss_function import DetectionLossConfig
from detection.modules.residual_block import ResidualBlock
from detection.modules.voxelizer import VoxelizerConfig
from detection.types import Detections


@dataclass
class DetectionModelConfig:
    """Detection model configuration."""

    voxelizer: VoxelizerConfig = field(
        default_factory=lambda: VoxelizerConfig(
            x_range=(-76.0, 76.0),
            y_range=(-50.0, 50.0),
            z_range=(0.0, 10.0),
            step=0.25,
        )
    )
    loss: DetectionLossConfig = field(
        default_factory=lambda: DetectionLossConfig(
            heatmap_loss_weight=100.0,
            offset_loss_weight=10.0,
            size_loss_weight=1.0,
            heading_loss_weight=100.0,
            heatmap_threshold=0.01,
            heatmap_norm_scale=20.0,
        )
    )


class DetectionModel(nn.Module):
    """A basic object detection model."""

    def __init__(self, config: DetectionModelConfig) -> None:
        super(DetectionModel, self).__init__()

        D, _, _ = config.voxelizer.bev_size
        self._backbone = nn.Sequential(
            nn.Conv2d(D, 32, 3, 1, 1),  # 1x
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1),  # 2x
            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, 2, 1),  # 4x
            ResidualBlock(128),
            nn.Conv2d(128, 256, 3, 2, 1),  # 8x
            ResidualBlock(256),
            nn.Conv2d(256, 512, 3, 2, 1),  # 16x
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(512, 256, 3, 1, 1),  # 8x
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 256, 3, 1, 1),  # 4x
        )

        self._head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 7, 3, 1, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),  # 1x
        )

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass of the model's neural network.

        Args:
            x: A [batch_size x D x H x W] tensor, representing the input LiDAR
                point cloud in a bird's eye view voxel representation.

        Returns:
            A [batch_size x 7 x H x W] tensor, representing the dense detection outputs.
                The 7 channels are (heatmap, offset_x, offset_y, x_size, y_size, sin_theta, cos_theta).
        """
        return self._head(self._backbone(x))

    @torch.no_grad()
    def inference(
        self, bev_lidar: Tensor, k: int = 100, score_threshold: float = 0.05
    ) -> Detections:
        """Predict a set of 2D bounding box detections from the given LiDAR point cloud.

        To predict a set of 2D bounding box detections, we use the following algorithm:
        1. Run the model's neural network to produce a [7 x H x W] prediction tensor.
            The 7 channels at each pixel (i, j) in the [H x W] BEV image are
            (heatmap, offset_x, offset_y, x_size, y_size, sin_theta, cos_theta).
        2. Find the coordinates of the local maximums in the predicted heatmap and keep the top K.
            We define the value at pixel (i, j) to be a local maximum if it is the maximum
            in a [5 x 5] window centered on (i, j). This gives us a [K x 2] tensor of
            coordinates in the [H x W] BEV image, where each row represents a detection.
            Each detection's score is given by its corresponding heatmap value.
        3. For each of the K detections, compute its centers by adding its predicted
            (offset_x, offset_y) to the detection's coordinates (i, j) from step 2.
            For example, if a detection has coordinates (100, 100) and its predicted
            offsets are (0.1, 0.2), then its center is (100.1, 100.2).
        4. For each of the K detections, set its bounding box size equal to the
            (x_size, y_size) values predicted at coordinates (i, j).
        5. For each of the K detections, set its heading equal to atan2(sin_theta, cos_theta),
            where (sin_theta, cos_theta) are the values predicted at coordinates (i, j).
        6. Remove all detections with a score less than or equal to `score_threshold`.

        Args:
            bev_lidar: A [D x H x W] tensor containing the bird's eye view voxel
                representation for one LiDAR point cloud. Note that batch inference
                is not supported!
            k: The maximum number of detections to keep; defaults to 100.
            score_threshold: Keep only detections with a score exceeding `score_threshold`.
                Defaults to 0.05.

        Returns:
            A set of 2D bounding box detections.
        """
        # TODO: Replace this stub code.
        D, H, W = bev_lidar.shape
        # step1:
        # (heatmap, offset_x, offset_y, x_size, y_size, sin_theta, cos_theta)
        X = self.forward(bev_lidar.view((1, D, H, W)))[0]  # 7 x H x W

        # step2:
        heatmap = X[None, 0, :, :]  # 1 x H x W
        maxpool2d = torch.nn.MaxPool2d(5, stride=1, padding=2)
        maxpooled_heatmap = maxpool2d(heatmap)
        mask = torch.eq(heatmap, maxpooled_heatmap)
        scores, indices = torch.topk(input=(heatmap * mask).flatten(), k=k)
        top_k_scores = scores.reshape((k, 1))
        # top_k_indices = Tensor(np.unravel_index(indices=indices.numpy(), shape=(H, W))).T.long()  # k x 2 -> (i, j)
        top_k_indices = Tensor([[torch.div(index, W, rounding_mode='floor'), index % W] for index in indices]).long().cuda()

        # step3:
        offsets_xy = torch.stack([X[1, :, :], X[2, :, :]], dim=0).permute(1, 2, 0)  # H x W x 2
        centroids = top_k_indices + offsets_xy[top_k_indices[:, 0], top_k_indices[:, 1]]  # k x 2

        # step4:
        sizes_xy = torch.stack([X[3, :, :], X[4, :, :]], dim=0).permute(1, 2, 0)
        boxes = sizes_xy[top_k_indices[:, 0], top_k_indices[:, 1]]  # k x 2

        # step5:
        sin = X[5:6, :, :].permute(1, 2, 0)
        cos = X[6:7, :, :].permute(1, 2, 0)
        yaws = torch.atan2(sin[top_k_indices[:, 0], top_k_indices[:, 1]], cos[top_k_indices[:, 0], top_k_indices[:, 1]])  # k x 1

        # step6:
        # ref: https://stackoverflow.com/questions/57570043/filter-data-in-pytorch-tensor
        raw = torch.cat([centroids, yaws, boxes, top_k_scores], dim=-1)  # k x 6
        print("raw:", raw.shape)
        filtered = raw[raw[:, 5] > score_threshold]
        print("filtered:", filtered.shape)
        print("Detections:", filtered[:, 0:2].shape, filtered[:, 2].shape, filtered[:, 3:5].shape, filtered[:, 5].shape)

        return Detections(
            filtered[:, 0:2], filtered[:, 2], filtered[:, 3:5], filtered[:, 5]
        )
