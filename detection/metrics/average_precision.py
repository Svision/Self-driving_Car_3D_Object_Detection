from dataclasses import dataclass
from typing import List

import torch

from detection.metrics.types import EvaluationFrame


@dataclass
class PRCurve:
    """A precision/recall curve.
    Attributes:
        precision: [N] vector of precision values, where N is the total number of detections.
            The element at index n denotes the precision of the top n detections when ordered by
            decreasing detection scores.
        recall: [N] vector of recall values, where N is the total number of detections.
            The element at index n denotes the recall of the top n detections when ordered by
            decreasing detection scores.
    """

    precision: torch.Tensor
    recall: torch.Tensor


@dataclass
class AveragePrecisionMetric:
    """Stores average precision and its associate precision-recall curve."""

    ap: float
    pr_curve: PRCurve


def compute_precision_recall_curve(
        frames: List[EvaluationFrame], threshold: float
) -> PRCurve:
    """Compute a precision/recall curve over a batch of evaluation frames.
    The PR curve plots the trade-off between precision and recall when sweeping
    across different score thresholds for your detections. To compute precision
    and recall for a score threshold s_i, consider the set of detections with
    scores greater than or equal to s_i. A detection is a true positive if it
    matches a ground truth label; it is a false positive if it does not.
    With this, we define precision = TP / (TP + FP) and recall = TP / (TP + FN),
    where TP is the number of true positive detections, FP is the number of false
    positive detections, and FN is the number of false negative labels (i.e. the
    number of ground truth labels that did not match any detections). By varying
    the score threshold s_i over all detection scores, we have the PR curve.
    What does it mean for a detection to match a ground truth label? In this assignment, we use
    the following definition: A detection matches a ground truth label if: (1) the Euclidean
    distance between their centers is at most `threshold`; and (2) no higher scoring detection
    satisfies condition (1) with respect to the same label.
    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.
    Returns:
        A precision/recall curve.
    """

    # TODO: Replace this stub code.
    @dataclass
    class Matching:
        scores: torch.Tensor  # N dimensional, scores of all detections
        true_positives: torch.Tensor  # N dimensional, detections with a match has a 1, otherwise 0
        false_negatives: torch.Tensor  # M dimensional, labels without a match has a 1, otherwise 0

    @dataclass
    class BatchedMatching:
        scores: torch.Tensor  # sum(len(a single matching.scores)) dimensional, scores of all detections in the batch
        true_positives: torch.Tensor  # sum(len(a single matching.true_positives)) dimensional
        false_negatives: torch.Tensor  # sum(len(a single matching.false_negatives)) dimensional

        def __init__(self, matchings: List[Matching]):
            self.scores = torch.cat([matching.scores for matching in matchings])
            self.true_positives = torch.cat([matching.true_positives for matching in matchings])
            self.false_negatives = torch.cat([matching.false_negatives for matching in matchings])

    def to_key(tensor: torch.Tensor) -> tuple:
        return (round(float(tensor[0]), 4), round(float(tensor[1]), 4))


    matching_list = []
    for frame in frames:
        detections = frame.detections
        labels = frame.labels
        scores = detections.scores
        # sort all detections with scores in descending order:
        sorted_scores, sorted_scores_indices = torch.sort(scores, descending=True)
        detections.centroids = detections.centroids[sorted_scores_indices]

        true_positives = torch.zeros(len(detections))
        false_negatives = torch.ones(len(labels))

        label_detections_map = {}  # a dictionary store all detections within the threshold of the
        label_scores_map = {}  # a dictionary store the scores of all detections within the threshold of the label
        # parse the label_detections_map and label_highest_score_map
        for i, label_centroid in enumerate(labels.centroids):
            distance_map = torch.linalg.norm(detections.centroids - label_centroid, dim=1)  # N dimensional
            true_map = distance_map <= threshold  # N dimensional
            if torch.count_nonzero(true_map):
                indices = torch.nonzero(true_map)
                label_scores_map[to_key(label_centroid)], ascending_indices = detections.scores[indices].sort()
                label_detections_map[to_key(label_centroid)] = detections.centroids[indices][ascending_indices].flatten()
            else:
                label_detections_map[to_key(label_centroid)] = torch.tensor([])

        detection_closest_label_map = {}  # a dictionary maps a detection with its closest label
        detection_labels_map = {}  # a dictionary maps a detection with all labels within distance threshold
        for j, detection_centroid in enumerate(detections.centroids):
            distance_map = torch.linalg.norm(labels.centroids - detection_centroid, dim=1)  # M dimensional
            true_map = distance_map <= threshold  # M dimension
            if torch.count_nonzero(true_map):
                indices = torch.nonzero(true_map)
                detection_labels_map[to_key(detection_centroid)] = labels.centroids[indices].flatten()
                min_distance_coord = torch.argmin(distance_map)
                detection_closest_label_map[to_key(detection_centroid)] = labels.centroids[min_distance_coord]

        # for each detection, generate the Matching class
        for k, detection_centroid in enumerate(detections.centroids):
            if to_key(detection_centroid) not in detection_closest_label_map:
                continue
            closest_label = detection_closest_label_map[to_key(detection_centroid)]
            if sorted_scores[k] == label_scores_map[to_key(closest_label)][-1]:  # the last element is the highest score
                true_positives[k] = 1
                closest_label_index = (labels.centroids == closest_label).nonzero()[0]
                false_negatives[closest_label_index] = 0  # a match should flip the label to a TP
                label_detections_map.pop(to_key(closest_label))
                label_scores_map.pop(to_key(closest_label))

        matching_list.append(Matching(sorted_scores, true_positives, false_negatives))

    # Having batch_size of Matching, generate a BatchedMatching class for precision/recall calculation
    all_matching = BatchedMatching(matching_list)
    all_matching.scores, sorted_indices = torch.sort(all_matching.scores, descending=True)
    all_matching.true_positives = all_matching.true_positives[sorted_indices]
    precisions = torch.zeros(len(all_matching.scores))  # precision = TP / (TP + FP)
    recalls = torch.zeros(len(all_matching.scores))  # recall = TP / (TP + FN)
    for n in range(len(all_matching.scores)):
        TP = torch.count_nonzero(all_matching.true_positives[:n + 1])
        precisions[n] = TP / (n + 1)
        recalls[n] = TP / len(all_matching.false_negatives)
    return PRCurve(precisions, recalls)


def compute_area_under_curve(curve: PRCurve) -> float:
    """Return the area under the given curve.
    Given a `PRCurve` curve, this function computes the area under the curve as:
        AP = \sum_{i = 1}^{n} (r_i - r_{i - 1}) * p_i
    where r_i (resp. p_i) is the recall (resp. precision) of the top i detections,
    n is the total number of detections, and we set r_0 = 0.0. Intuitively, this
    is computing the integral of the step function defined by the PRCurve.
    Args:
        curve: The precision/recall curve.
    Returns:
        The area under the curve, as defined above.
    """
    # Done: Replace this stub code.
    # N dimensional, each is (r_i - r_{i - 1})
    recall_diff = curve.recall - torch.cat((torch.Tensor([0]), curve.recall[: len(curve.recall) - 1]))
    area = torch.sum(curve.precision * recall_diff).float()
    return area


def compute_average_precision(
    frames: List[EvaluationFrame], threshold: float
) -> AveragePrecisionMetric:
    """Compute average precision over a batch of evaluation frames.
    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.
    Returns:
        A dataclass consisting of a PRCurve and its average precision.
    """
    # DONE: Replace this stub code.
    pr_curve = compute_precision_recall_curve(frames, threshold)
    ap = compute_area_under_curve(pr_curve)
    return AveragePrecisionMetric(ap, pr_curve)
