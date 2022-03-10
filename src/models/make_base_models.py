import torch
import torchvision
from torch.nn.init import xavier_normal_
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.ops import misc as misc_nn_ops
import numpy as np

#from src.models.modifications
from models.modifications import RoIHeadsEndHeights, FasterRCNNEndHeights, RoIHeadsVanilla, FasterRCNNStartHeights, \
    FasterRCNNVanilla, FasterRCNNPreRpn
from torch import nn

NUM_CLASSES = 2
IMAGE_MEAN = [0.513498842716217, 0.5408999919891357, 0.5676814913749695, 0.003977019805461168]
IMAGE_STD = [0.21669578552246094, 0.22595597803592682, 0.2860477566719055, 0.007557219825685024]

#
def make_vanilla_model(model_dir, pretrained=True, trainable_backbone_layers=0, pretrained_backbone=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,
                                                   trainable_backbone_layers=trainable_backbone_layers,
                                                                 pretrained_backbone=pretrained_backbone)
    # This model has no new weights
    model.new_weights = nn.ParameterList()
    # Assign model new class to override forward function so it can take height input (and ignore it)
    model.__class__ = FasterRCNNVanilla
    # Replace the model's RoIHead with our version
    model.roi_heads.__class__ = RoIHeadsVanilla
    # Assign a new class prediction layer for two classes
    representation_size = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = nn.Linear(representation_size, NUM_CLASSES)
    # TODO: Better this way because the bounding boxes are set for each class for some reason????
    # box_predictor = FastRCNNPredictor(representation_size,num_classes)
    if pretrained:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_pretrained.pt"
    elif pretrained_backbone:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_pretrained_backbone.pt"
    else:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_no_pretraining.pt"
    torch.save(model, path)

def make_final_layer_model(model_dir, pretrained=True, trainable_backbone_layers=0, pretrained_backbone=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,
                                                                 trainable_backbone_layers=trainable_backbone_layers,
                                                                 pretrained_backbone=pretrained_backbone)
    # Change the model class so the model uses our forward function
    model.__class__ = FasterRCNNEndHeights
    # Change the roi heads class so they use our forward function
    model.roi_heads.__class__ = RoIHeadsEndHeights
    # Rebuild add extra weights to the fc layer in the roi head to handle the new features
    resolution = model.roi_heads.box_roi_pool.output_size[0]
    representation_size = model.roi_heads.box_predictor.cls_score.in_features
    # Extend the existing box head with weights for the height layer
    num_new_features = resolution ** 2
    existing_weights = model.roi_heads.box_head.fc6.weight
    new_weights = torch.zeros(representation_size, num_new_features)
    xavier_normal_(new_weights)
    extended_weights = torch.cat((existing_weights, new_weights), 1)
    model.roi_heads.box_head.fc6.weight = nn.Parameter(extended_weights)
    model.roi_heads.box_head.fc6.in_features = extended_weights.shape[1]
    model.roi_heads.box_roi_pool.featmap_names.append('heights')
    # Log the new weights that we added so they can be tracked
    model.existing_weights_shape = existing_weights.shape

    new_weights = torch.narrow(extended_weights, 1, 0, existing_weights.shape[1])
    # new_weights = torch.narrow(extended_weights, 1, existing_weights.shape[1], new_weights.shape[1])


    model.new_weights = nn.ParameterList([nn.Parameter(new_weights)])
    # Assign a new class prediction layer for two classes
    model.roi_heads.box_predictor.cls_score = nn.Linear(representation_size, NUM_CLASSES)
    if pretrained:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_pretrained_final.pt"
    elif pretrained_backbone:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_pretrained_backbone_final.pt"
    else:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_no_pretraining_final.pt"
    torch.save(model, path)


def make_first_layer_model(model_dir, pretrained=True, trainable_backbone_layers=0, pretrained_backbone=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,
                                                                 trainable_backbone_layers=trainable_backbone_layers,
                                                                 pretrained_backbone=pretrained_backbone)
    # Change the model class so the model uses our forward function
    model.__class__ = FasterRCNNStartHeights
    # Don't need to change the roi_heads. Everything is normal there
    model.roi_heads.__class__ = RoIHeadsVanilla
    # Rebuild first conv layer of the backbone to accept one extra layer
    new_shape = list(model.backbone.body.conv1.weight.shape)
    new_shape[1] = 1
    new_weights = torch.zeros(new_shape)
    xavier_normal_(new_weights)
    existing_weights = model.backbone.body.conv1.weight
    extended_weights = torch.cat((existing_weights, new_weights), 1)
    model.backbone.body.conv1.weight = nn.Parameter(extended_weights)
    model.existing_weights_shape = existing_weights.shape
    # Now change the transformation to accomodate 4d transformations
    model.transform.image_mean = IMAGE_MEAN
    model.transform.image_std = IMAGE_STD
    # Assign a new class prediction layer for two classes
    representation_size = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = nn.Linear(representation_size, NUM_CLASSES)
    if pretrained:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_pretrained_first.pt"
    elif pretrained_backbone:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_pretrained_backbone_first.pt"
    else:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_no_pretraining_first.pt"
    torch.save(model, path)

def make_normal_backbone_model(model_dir, pretrained=True, trainable_backbone_layers=0):
    # Assign the new FPN that only takes the final feature map layer
    backbone = torchvision.models.resnet50(pretrained=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    returned_layers = [4]
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    backbone_features = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=None)
    backbone_features.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=None)
    # An anchor generator that is appropriate for a single feature map
    rpn_anchor_generator = AnchorGenerator()
    # This won't have pretrained weights in the RPN
    model = torchvision.models.detection.FasterRCNN(backbone_features, NUM_CLASSES, rpn_anchor_generator=rpn_anchor_generator)
    # Assign the trainable layers (this is usually done during initialisation)
    # This model has no new weights
    model.new_weights = nn.ParameterList()
    # Assign model new class to override forward function so it can take height input (and ignore it)
    model.__class__ = FasterRCNNVanilla
    # Replace the model's RoIHead with our version
    model.roi_heads.__class__ = RoIHeadsVanilla
    if pretrained:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_pretrained_basic_backbone.pt"
    else:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_no_pretraining_basic_backbone.pt"
    torch.save(model, path)

def make_pre_roi_model(model_dir, returned_layers, pretrained=True, pooling_layer=False, out_channels=256):
    assert min(returned_layers) > 0 and max(
        returned_layers) < 5, "Returned layers must correspond to layers in the resnet"
    # TODO: Add option to change this with checks for returned layers being correct
    trainable_backbone_layers = 5
    # Assign the new FPN that only takes the final feature map layer
    backbone = torchvision.models.resnet50(pretrained=pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    backbone_features = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
    # Make sure that there is no pooling layer returned
    if not pooling_layer:
        backbone_features.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list,
                                                      out_channels=out_channels,
                                                      extra_blocks=None)
    # Make a note if the pooling layer is included
    if pooling_layer:
        returned_layers.append(5)
    # An anchor generator that is appropriate for a single feature map
    # Only add the necessary layers to the anchor generation
    ANCHOR_SIZES = [(32,), (64,), (128,), (256,), (512,)]
    our_anchor_sizes = [ANCHOR_SIZES[ind-1] for ind in returned_layers]
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(our_anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        our_anchor_sizes, aspect_ratios)
    # Change the number of out channels to include the height layer
    backbone_features.out_channels += 1
    # This won't have pretrained weights in the RPN
    model = torchvision.models.detection.FasterRCNN(backbone_features, NUM_CLASSES,
                                                    rpn_anchor_generator=rpn_anchor_generator)
    # Change the number of out channels to include the height layer
    # Assign the trainable layers (this is usually done during initialisation)
    # This model has no new weights
    model.new_weights = nn.ParameterList()
    # Assign model new class to override forward function so it can take height input (and ignore it)
    model.__class__ = FasterRCNNPreRpn
    # Replace the model's RoIHead with our version
    model.roi_heads.__class__ = RoIHeadsVanilla

    if pretrained:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_pretrained_pre_rpn_{returned_layers}_out_channels_{out_channels}.pt"
    else:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_no_pretraining_pre_rpn_{returned_layers}_out_channels_{out_channels}.pt"
    torch.save(model, path)


def make_pre_rpn_pretrained_model(model_dir, returned_layers, pooling_layer=False, out_channels=256):
    assert min(returned_layers) > 0 and max(
        returned_layers) < 5, "Returned layers must correspond to layers in the resnet"
    num_orig_out_channels = 256
    returned_layers = returned_layers.copy()
    rpn_returned = returned_layers.copy()
    if pooling_layer:
        rpn_returned.append(5)
    assert out_channels <= num_orig_out_channels, "The original model only had 256 layers per backbone FPN feature map"
    trainable_backbone_layers = 5
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                 trainable_backbone_layers=trainable_backbone_layers)
    # If we keep the pooling layer that is the fifth and final layer of the FPN output
    ####################################################################################################################
    # The FPN of the backbone needs to be adapted
    fpn_outputs_to_keep = np.array(returned_layers) - 1
    # The missing layers need to be removed from the list of backbone layers to access
    missing_layers = [f'layer{k}' for k in [1, 2, 3, 4] if k not in returned_layers]
    if not pooling_layer:
        model.backbone.fpn.extra_blocks = None
    for missing_layer in missing_layers:
        del model.backbone.body.return_layers[missing_layer]
    # Only keep the specified FPN layers
    new_inner = [x for i, x in enumerate(model.backbone.fpn.inner_blocks) if i in fpn_outputs_to_keep]
    model.backbone.fpn.inner_blocks = torch.nn.ModuleList(new_inner)
    new_layer = [x for i, x in enumerate(model.backbone.fpn.layer_blocks) if i in fpn_outputs_to_keep]
    model.backbone.fpn.layer_blocks = torch.nn.ModuleList(new_layer)
    # Now loop through the kept layers and reduce them to the correct number of out channels
    for i in range(len(model.backbone.fpn.layer_blocks)):
        conv = model.backbone.fpn.layer_blocks[i]
        conv.weight = torch.nn.Parameter(conv.weight[:out_channels])
        conv.bias = torch.nn.Parameter(conv.bias[:out_channels])
        conv.out_channels = out_channels
    ####################################################################################################################
    # We now need to alter the RPN to accomodate the new size
    #
    ANCHOR_SIZES = [(32,), (64,), (128,), (256,), (512,)]
    our_anchor_sizes = [ANCHOR_SIZES[ind-1] for ind in rpn_returned]
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(our_anchor_sizes)
    model.rpn.anchor_generator = AnchorGenerator(
        our_anchor_sizes, aspect_ratios)
    ####################################################################################################################
    # Now the rpn head
    cov_weight = model.rpn.head.conv.weight
    new_shape = list(cov_weight.shape)
    new_shape[1] = 1
    new_weights = torch.zeros(new_shape)
    xavier_normal_(new_weights)
    combined_weights = torch.cat([cov_weight[:, :out_channels], new_weights], dim=1)
    model.rpn.head.conv.weight = nn.Parameter(combined_weights)
    ####################################################################################################################
    # Now the RoI heads
    #
    # First the roi pooling
    featmap_names = fpn_outputs_to_keep.astype(str)
    box_roi_pool = MultiScaleRoIAlign(featmap_names, output_size=7, sampling_ratio=2)
    model.roi_heads.box_roi_pool = box_roi_pool
    # Now the box head
    resolution = box_roi_pool.output_size[0]
    box_head_weight = model.roi_heads.box_head.fc6.weight
    rep_size = box_head_weight.shape[0]
    end = out_channels * resolution ** 2
    # Now add the new weights for the extra layer
    new_weights = torch.zeros(rep_size, resolution ** 2)
    xavier_normal_(new_weights)
    combined_weights = torch.cat([box_head_weight[:, :end], new_weights], dim=1)
    model.roi_heads.box_head.fc6.weight = torch.nn.Parameter(combined_weights)
    # Assign model new class to override forward function so it can take height input (and ignore it)
    model.__class__ = FasterRCNNPreRpn
    # Replace the model's RoIHead with our version
    model.roi_heads.__class__ = RoIHeadsVanilla
    path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_really_pretrained_pre_rpn_{rpn_returned}_out_channels_{out_channels}.pt"
    torch.save(model, path)



def make_basic_pre_rpn(model_dir, pretrained, pretrained_backbone):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=False, trainable_backbone_layers=5, pretrained_backbone=pretrained_backbone)
    
    cov_weight = model.rpn.head.conv.weight
    new_shape = list(cov_weight.shape)
    new_shape[1] = 1
    new_weights = torch.zeros(new_shape)
    xavier_normal_(new_weights)
    combined_weights = torch.cat([cov_weight[:, :256], new_weights], dim=1)
    model.rpn.head.conv.weight = nn.Parameter(combined_weights)

    # Now the box head
    resolution = model.roi_heads.box_roi_pool.output_size[0]
    box_head_weight = model.roi_heads.box_head.fc6.weight
    rep_size = box_head_weight.shape[0]
    end = 256 * resolution ** 2
    # Now add the new weights for the extra layer
    new_weights = torch.zeros(rep_size, resolution ** 2)
    xavier_normal_(new_weights)
    combined_weights = torch.cat([box_head_weight[:, :end], new_weights], dim=1)
    model.roi_heads.box_head.fc6.weight = torch.nn.Parameter(combined_weights)

    # Assign model new class to override forward function so it can take height input
    model.__class__ = FasterRCNNPreRpn
    # Replace the model's RoIHead with our version
    model.roi_heads.__class__ = RoIHeadsVanilla

    representation_size = model.roi_heads.box_predictor.cls_score.in_features
    #model.roi_heads.box_predictor.cls_score = nn.Linear(representation_size, NUM_CLASSES)
    model.roi_heads.box_predictor = FastRCNNPredictor(representation_size, NUM_CLASSES)

    if pretrained and pretrained_backbone:
        path = model_dir / f"RCNN_basic_TRUE_TRUE_pre_rpn.pt"
    elif pretrained:
        path = model_dir / f"RCNN_basic_TRUE_FALSE_pre_rpn.pt"
    elif pretrained_backbone:
        path = model_dir / f"RCNN_basic_FALSE_TRUE_pre_rpn.pt"
    else:
        path = model_dir / f"RCNN_basic_FALSE_FALSE_pre_rpn.pt"
    torch.save(model, path)

def make_basic_vanilla(model_dir, pretrained, pretrained_backbone):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=False, trainable_backbone_layers=5, pretrained_backbone=pretrained_backbone)

    model.__class__ = FasterRCNNVanilla
    # Replace the model's RoIHead with our version
    model.roi_heads.__class__ = RoIHeadsVanilla
    
    # Assign a new class prediction layer for two classes
    representation_size = model.roi_heads.box_predictor.cls_score.in_features
    #model.roi_heads.box_predictor.cls_score = nn.Linear(representation_size, NUM_CLASSES)
    model.roi_heads.box_predictor = FastRCNNPredictor(representation_size, NUM_CLASSES)

    if pretrained and pretrained_backbone:
        path = model_dir / f"RCNN_basic_TRUE_TRUE_vanilla.pt"
    elif pretrained:
        path = model_dir / f"RCNN_basic_TRUE_FALSE_vanilla.pt"
    elif pretrained_backbone:
        path = model_dir / f"RCNN_basic_FALSE_TRUE_vanilla.pt"
    else:
        path = model_dir / f"RCNN_basic_FALSE_FALSE_vanilla.pt"
    torch.save(model, path)

def make_basic_roi_model(model_dir, returned_layers, pretrained, pooling_layer=True, out_channels=256):

    assert min(returned_layers) > 0 and max(
        returned_layers) < 5, "Returned layers must correspond to layers in the resnet"
    # TODO: Add option to change this with checks for returned layers being correct
    trainable_backbone_layers = 5
    # Assign the new FPN that only takes the final feature map layer
    backbone = torchvision.models.resnet50(pretrained=pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    backbone_features = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
    # Make sure that there is no pooling layer returned
    if not pooling_layer:
        backbone_features.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list,
                                                      out_channels=out_channels,
                                                      extra_blocks=None)
    # Make a note if the pooling layer is included
    if pooling_layer:
        returned_layers.append(5)
    # An anchor generator that is appropriate for a single feature map
    # Only add the necessary layers to the anchor generation
    ANCHOR_SIZES = [(32,), (64,), (128,), (256,), (512,)]
    our_anchor_sizes = [ANCHOR_SIZES[ind-1] for ind in returned_layers]
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(our_anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        our_anchor_sizes, aspect_ratios)
    # Change the number of out channels to include the height layer
    backbone_features.out_channels += 1
    # This won't have pretrained weights in the RPN
    model = torchvision.models.detection.FasterRCNN(backbone_features, NUM_CLASSES,
                                                    rpn_anchor_generator=rpn_anchor_generator)
    # Change the number of out channels to include the height layer
    # Assign the trainable layers (this is usually done during initialisation)
    # This model has no new weights
    #model.new_weights = nn.ParameterList()

    # Assign model new class to override forward function so it can take height input (and ignore it)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=False, trainable_backbone_layers=5, pretrained_backbone=True)
    
    model.__class__ = FasterRCNNPreRpn
    # Replace the model's RoIHead with our version
    model.roi_heads.__class__ = RoIHeadsVanilla

    # Assign a new class prediction layer for two classes
    representation_size = model.roi_heads.box_predictor.bbox_pred.in_features
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(representation_size, 91*4)

    if pretrained:
        path = model_dir / f"RCNN_basic_TRUE_roi.pt"
    else:
        path = model_dir / f"RCNN_basic_FALSE_roi.pt"
    torch.save(model, path)