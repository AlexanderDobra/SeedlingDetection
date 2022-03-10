"""
A file containing modifications to some functions of the torchvision packages
"""
import logging
from collections import OrderedDict

import warnings
from typing import Union

import torchvision
from torch.nn import AdaptiveAvgPool2d
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.roi_heads import fastrcnn_loss, RoIHeads
import torch
from torch import nn, Tensor
from torch.jit.annotations import Optional, List, Dict, Tuple
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.ops.poolers import _onnx_merge_levels, initLevelMapper, LevelMapper, MultiScaleRoIAlign


HEIGHT_MEAN = 0.003977019805461168
HEIGHT_STD = 0.007557219825685024

logger = logging.getLogger(__name__)

class RoIHeadsVanilla(RoIHeads):
    """
    The RoIHeads class for integration of height data in the final NN layer
    """
    def forward(self,
                features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            # Get proposals and match them to targets
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None
        # Extract the feature maps for each proposal using roi pooling
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        # This has two fully connected layers
        box_features = self.box_head(box_features)
        # We now have the two heads - one for classification and one for the boxes of each class
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            # Calculate the loss
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        # else:
        # logger.log(logging.INFO, box_regression)
        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )
        # We are not dealing with masks or keypoints
        assert not self.has_mask(), "This functionality is not supported"
        assert self.keypoint_roi_pool is None, "This functionality is not supported"
        return result, losses

class RoIHeadsEndHeights(RoIHeads):
    def forward(self,
                features,      # type: Dict[str, Tensor]
                heights,       # type: [Tensor]
                proposals,     # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'
        if self.training:
            # Get proposals and match them to targets
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None
        attrs = vars(self.box_roi_pool)
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        ### Now we extract the features from the heights
        # First the heights need to be in the correct format.
        heights = [height_im.unsqueeze(0) for height_im in heights]
        heights = torch.cat(heights, 0)
        #Normalize the heights
        heights = (heights - HEIGHT_MEAN) / HEIGHT_STD
        heights = {"heights": heights}
        box_heights = self.box_roi_pool(heights, proposals, image_shapes)
        # Now we concatenate the features together
        new_features = torch.cat((box_heights, box_features), 1)
        # This has two fully connected layers
        box_features = self.box_head(new_features)
        # We now have the two heads - one for classification and one for the boxes of each class
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            # Calculate the loss
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        # else:
        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )
        # We are not dealing with masks or keypoints
        assert not self.has_mask(), "This functionality is not supported"
        assert self.keypoint_roi_pool is None, "This functionality is not supported"
        return result, losses

class FasterRCNNEndHeights(FasterRCNN):
    def forward(self, images, heights, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        # This is the only difference to the forward function - we feed the heights parameter in here
        detections, detector_losses = self.roi_heads(features, heights, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections)

    def get_new_weights(self):
        """
        Get the new weights from the final model.

        This needs to be done with a function because shared memory pointers are not conserved with pickle
        :return: The new weights in this model
        """
        extended_weights = self.roi_heads.box_head.fc6.weight
        start = self.existing_weights_shape[1]
        length = extended_weights.shape[1] - self.existing_weights_shape[1]
        new_weights = torch.narrow(extended_weights, 1, start, length)
        return nn.ParameterList([nn.Parameter(new_weights)])

class FasterRCNNVanilla(FasterRCNN):
    """
    Wrapper around the standard class that can take the height argument (and ignore it)
    """
    def forward(self, images, heights, targets=None):
        return super().forward(images, targets)

    def get_new_weights(self):
        return nn.ParameterList()

class FasterRCNNStartHeights(FasterRCNN):
    def forward(self, images, heights, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # Add an extra channel to the images here
        im_num = 0
        for image, height_map in zip(images, heights):
            image = torch.cat((image, height_map), 0)
            images[im_num] = image
            im_num += 1
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        # No heights needed here - the height data was fed directly into the backbone
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections)

    def get_new_weights(self):
        """
        Get the new weights from the final model.

        This needs to be done with a function because shared memory pointers are not conserved with pickle
        :return: The new weights in this model
        """
        extended_weights = self.backbone.body.conv1.weight
        start = self.existing_weights_shape[1]
        length = extended_weights.shape[1] - self.existing_weights_shape[1]
        new_weights = torch.narrow(extended_weights, 1, start, length)
        return nn.ParameterList([nn.Parameter(new_weights)])

class FasterRCNNPreRpn(FasterRCNN):
    def forward(self, images, heights, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        # Add the heights to the features here
        features = self.add_heights(heights, features)
        proposals, proposal_losses = self.rpn(images, features, targets)
        # This is the only difference to the forward function - we feed the heights parameter in here
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections)

    def add_heights(self, heights, features):
        # Put heights in the correct format
        heights = [height_im.unsqueeze(0) for height_im in heights]
        heights = torch.cat(heights, 0)
        new_features = OrderedDict()
        for key, feature in features.items():
            feature_size = feature.shape[2:]
            # Resize the heights image to this new shape
            avg_pool = AdaptiveAvgPool2d(feature_size)
            new_heights = avg_pool(heights)
            # Now concatenate the heights
            new_feature = torch.cat((feature, new_heights), dim=1)
            new_features[key] = new_feature
        return new_features

    def get_new_weights(self):
        """
        Get the new weights from the final model.

        This needs to be done with a function because shared memory pointers are not conserved with pickle
        :return: The new weights in this model
        """
        return [torch.zeros(20)]