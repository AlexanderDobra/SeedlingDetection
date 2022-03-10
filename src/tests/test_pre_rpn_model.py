import pathlib
from unittest import TestCase

import numpy as np
import torch

from make_base_models_copy import make_pre_rpn_pretrained_model


class PretrainedPreRPNTest(TestCase):

    def test_basic(self):
        """ Test that the FPN now outputs the correct number and size of feature maps"""
        module_path = pathlib.Path(__file__).parent
        base_dir = module_path.parent.parent.absolute()
        template_dir = base_dir / "src" / "tests" / "temp"
        out_channels = 64
        returned_layers = [1, 4]
        pooling_layer = True
        make_pre_rpn_pretrained_model(template_dir, returned_layers, pooling_layer=pooling_layer, out_channels=out_channels)
        # Read in the new model and test it
        if pooling_layer:
            name_layers = returned_layers.copy()
            name_layers.append(5)
        filename = f"RCNN-resnet-50_5_layer_really_pretrained_pre_rpn_{name_layers}_out_channels_{out_channels}.pt"
        model = torch.load(template_dir / filename)
        # Test the output shapes of the feature maps
        test_img = torch.randn(3, 256, 256)
        model.eval()
        targets = None
        images, _ = model.transform([test_img], targets)
        out = model.backbone(images.tensors)
        expected_names = list((np.array(returned_layers) - 1).astype(str))
        if pooling_layer:
            expected_names.append("pool")
        for name in out:
            self.assertTrue(name in expected_names)
            fm = out[name]
            self.assertTrue(fm.shape[1] == out_channels)

    def test_no_pooling(self):
        """
        Test the full model is able to run without exceptions
        """
        module_path = pathlib.Path(__file__).parent
        base_dir = module_path.parent.parent.absolute()
        template_dir = base_dir / "src" / "tests" / "temp"
        out_channels = 256
        returned_layers = [1, 2, 3, 4]
        pooling_layer = False
        make_pre_rpn_pretrained_model(template_dir, returned_layers, pooling_layer=pooling_layer,
                                      out_channels=out_channels)
        # Read in the new model and test it
        name_layers = returned_layers.copy()
        if pooling_layer:
            name_layers.append(5)
        filename = f"RCNN-resnet-50_5_layer_really_pretrained_pre_rpn_{name_layers}_out_channels_{out_channels}.pt"
        model = torch.load(template_dir / filename)
        # Test the output shapes of the feature maps
        test_img = torch.randn(3, 256, 256)
        test_height = torch.randn(1, 256, 256)
        targets = None
        # Check that there are no exceptions in the forward pass
        model.eval()
        out = model([test_img], [test_height], targets)




