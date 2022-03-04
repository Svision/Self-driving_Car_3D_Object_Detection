from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


def heatmap_weighted_mse_loss(
        targets: Tensor, predictions: Tensor, heatmap: Tensor, heatmap_threshold: float
) -> Tensor:
    """Compute the mean squared error (MSE) loss between `predictions` and `targets`, weighted by a heatmap.

    Specifically, the heatmap-weighted MSE loss can be computed as follows:
    1. Compute the MSE loss between `predictions` and `targets` along the C dimension.
        This should give us a [batch_size x H x W] tensor `mse_loss`.
    2. Compute a binary mask based on whether the values of `heatmap` exceeds `heatmap_threshold`.
        This should give us a [batch_size x H x W] tensor `mask`.
    3. Compute the mean of `mse_loss` weighted by `heatmap` and masked by `mask`.
        This gives us our final scalar loss.

    Args:
        targets: A [batch_size x C x H x W] tensor, containing the ground truth targets.
        predictions: A [batch_size x C x H x W] tensor, containing the predictions.
        heatmap: A [batch_size x 1 x H x W] tensor, representing the ground truth heatmap.
        heatmap_threshold: We compute MSE loss for only elements where `heatmap > heatmap_threshold`.

    Returns:
        A scalar MSE loss between `predictions` and `targets`, aggregated as a
            weighted average using the provided `heatmap`.
    """
    # DONE: Replace this stub code.
    N, C, H, W = targets.shape
    # step 1
    mse_loss = torch.sum((predictions - targets) ** 2, dim=1)  # NxHxW
    # step 2
    bit_mask = (heatmap > heatmap_threshold).long()  # Nx1xHxW

    # step 3
    masked_weight = (bit_mask * heatmap).reshape((N, H, W))  # NxHxW
    mse_final = torch.mean(mse_loss * masked_weight)

    return mse_final


def focal_loss(
        targets: Tensor, predictions: Tensor, alpha: float = 0.25, gamma: float = 2
) -> Tensor:
    """Compute the focal loss between `predictions` and `targets`, weighted by a heatmap.
    Specifically, the heatmap-weighted focal loss can be computed as follows:
    1. Compute the focal loss between `predictions` and `targets` along the C dimension.
        This should give us a [batch_size x H x W] tensor `mse_loss`.
    2. Compute a binary mask based on whether the values of `heatmap` exceeds `heatmap_threshold`.
        This should give us a [batch_size x H x W] tensor `mask`.
    3. Compute the mean of `focal_loss` weighted by `heatmap` and masked by `mask`.
        This gives us our final scalar loss.
    Args:
        targets: A [batch_size x C x H x W] tensor, containing the ground truth targets.
        predictions: A [batch_size x C x H x W] tensor, containing the predictions.
        gamma: controls the shape of the curve. The higher the value of γ,
               the lower the loss for well-classified examples.
        alpha: give high weights to the rare class and small weights to the dominating or common class.
    Returns:
        A scalar focal loss between `predictions` and `targets`.
    """
    # ref: https://amaarora.github.io/2020/06/29/FocalLoss.html
    alpha = torch.tensor([alpha, 1-alpha]).cuda()
    BCE_loss = F.binary_cross_entropy_with_logits(predictions, targets)
    targets = targets.type(torch.long)
    at = alpha.gather(0, targets.data.view(-1))
    pt = torch.exp(-BCE_loss)
    focal_loss = at * (1 - pt) ** gamma * BCE_loss
    focal_loss = focal_loss / torch.max(focal_loss)

    return focal_loss.mean()


def focal_loss_v2(
        targets: Tensor, predictions: Tensor, alpha: float = 2.0, beta: float = 4.0
) -> Tensor:
    # ref: https://arxiv.org/pdf/1904.07850.pdf
    predictions = predictions / predictions.max()
    loss = torch.zeros(predictions.shape).cuda()
    positive_label_indices = targets == 1.0
    negative_label_indices = targets < 1.0
    loss[positive_label_indices] = (1 - predictions[positive_label_indices]) ** alpha * torch.log(
        predictions[positive_label_indices])
    loss[negative_label_indices] = (1 - targets[negative_label_indices]) ** beta * predictions[
        negative_label_indices] ** alpha * torch.log(1 - predictions[negative_label_indices])

    final_loss = torch.sum(loss, dim=1)
    final_loss = final_loss / final_loss.max()

    return - final_loss.mean()


def negative_hard_mining(
        targets: Tensor, predictions: Tensor, k: int = 100
) -> Tensor:
    """
    Simplified method which compute the total loss over the top k values of the heatmap that have high individual
    losses.
    Args:
        targets: A [batch_size x C x H x W] tensor, containing the ground truth targets.
        predictions: A [batch_size x C x H x W] tensor, containing the predictions.
        k: top k value to calculate
    Returns:
        A scalar MSE loss between `predictions` and `targets`.
    """
    mse_loss = (targets - predictions) ** 2
    hard_mining, indices = torch.topk(mse_loss.flatten(), k=k)

    hard_mining = hard_mining / torch.max(hard_mining)

    return hard_mining.mean()


@dataclass
class DetectionLossConfig:
    """Detection loss function configuration.

    Attributes:
        heatmap_loss_weight: The multiplicative weight of the heatmap loss.
        offset_loss_weight: The multiplicative weight of the offset loss.
        size_loss_weight: The multiplicative weight of the size loss.
        heading_loss_weight: The multiplicative weight of the heading loss.
        heatmap_threshold: A scalar threshold that controls whether we ignore the loss
            at a given location. In particular, we ignore the loss for all locations
            where the ground truth heatmap has a value less than or equal to `heatmap_threshold`.
        heatmap_norm_scale: A scalar value that scales the spread of a heatmap.
            The larger the value, the smaller the spread of the heatmap.
            See `detection/modules/loss_target.py` for usage details.
    """

    heatmap_loss_weight: float
    offset_loss_weight: float
    size_loss_weight: float
    heading_loss_weight: float
    heatmap_threshold: float
    heatmap_norm_scale: float


@dataclass
class DetectionLossMetadata:
    """Detailed breakdown of the detection loss."""

    total_loss: torch.Tensor
    heatmap_loss: torch.Tensor
    offset_loss: torch.Tensor
    size_loss: torch.Tensor
    heading_loss: torch.Tensor


class DetectionLossFunction(torch.nn.Module):
    """A loss function to train a detection model."""

    def __init__(self, config: DetectionLossConfig) -> None:
        super(DetectionLossFunction, self).__init__()
        self._heatmap_loss_weight = config.heatmap_loss_weight
        self._offset_loss_weight = config.offset_loss_weight
        self._size_loss_weight = config.size_loss_weight
        self._heading_loss_weight = config.heading_loss_weight
        self._heatmap_threshold = config.heatmap_threshold

    def forward(
            self, predictions: Tensor, targets: Tensor
    ) -> Tuple[torch.Tensor, DetectionLossMetadata]:
        """Compute the loss between the predicted detections and target labels.

        Args:
            predictions: A [batch_size x 7 x H x W] tensor containing the outputs of `DetectionModel`.
                The 7 channels are (heatmap, offset_x, offset_y, length, width, sin_theta, cos_theta).
            targets: A [batch_size x 7 x H x W] tensor containing the ground truth output.
                The 7 channels are (heatmap, offset_x, offset_y, length, width, sin_theta, cos_theta).

        Returns:
            The scalar tensor containing the weighted loss between `predictions` and `targets`.
        """
        # 1. Unpack the targets tensor.
        target_heatmap = targets[:, 0:1]  # [B x 1 x H x W]
        target_offsets = targets[:, 1:3]  # [B x 2 x H x W]
        target_sizes = targets[:, 3:5]  # [B x 2 x H x W]
        target_headings = targets[:, 5:7]  # [B x 2 x H x W]

        # 2. Unpack the predictions tensor.
        predicted_heatmap = torch.sigmoid(predictions[:, 0:1])  # [B x 1 x H x W]
        predicted_offsets = predictions[:, 1:3]  # [B x 2 x H x W]
        predicted_sizes = predictions[:, 3:5]  # [B x 2 x H x W]
        predicted_headings = predictions[:, 5:7]  # [B x 2 x H x W]

        # 3. Compute individual loss terms for heatmap, offset, size, and heading.
        # heatmap_loss = ((target_heatmap - predicted_heatmap) ** 2).mean()
        # heatmap_loss = negative_hard_mining(target_heatmap, predicted_heatmap)
        heatmap_loss = focal_loss_v2(target_heatmap, predicted_heatmap)
        offset_loss = heatmap_weighted_mse_loss(
            target_offsets, predicted_offsets, target_heatmap, self._heatmap_threshold
        )
        size_loss = heatmap_weighted_mse_loss(
            target_sizes, predicted_sizes, target_heatmap, self._heatmap_threshold
        )
        heading_loss = heatmap_weighted_mse_loss(
            target_headings, predicted_headings, target_heatmap, self._heatmap_threshold
        )

        # 4. Aggregate losses using the configured weights.
        total_loss = (
                heatmap_loss * self._heatmap_loss_weight
                + offset_loss * self._offset_loss_weight
                + size_loss * self._size_loss_weight
                + heading_loss * self._heading_loss_weight
        )

        loss_metadata = DetectionLossMetadata(
            total_loss, heatmap_loss, offset_loss, size_loss, heading_loss
        )
        return total_loss, loss_metadata
