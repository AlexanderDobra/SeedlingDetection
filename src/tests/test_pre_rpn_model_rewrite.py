import unittest

import numpy as np
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from make_base_models_copy import make_pre_rpn_pretrained_model


class PretrainedPreRPNTest(unittest.TestCase):

    def test_basic(self):
        """ Test that the FPN now outputs the correct number and size of feature maps"""
        
        out_channels = 64
        returned_layers = [1, 4]
        pooling_layer = True
        model = make_pre_rpn_pretrained_model(returned_layers, pooling_layer=pooling_layer, out_channels=out_channels)
        # Read in the new model and test it
        
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
        
        out_channels = 256
        returned_layers = [1, 2, 3, 4]
        pooling_layer = True
        model = make_pre_rpn_pretrained_model(returned_layers, pooling_layer=pooling_layer,
                                      out_channels=out_channels)

        #num_classes = 2
        #in_features = model.roi_heads.box_predictor.cls_score.in_features
        #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # Read in the new model and test it
        # TODO: Kind of a hack to deal with the two machines (my laptop and Rhodos)
        #if torch.cuda.device_count() == 2:
            #device = torch.device('cuda:0')
        #elif torch.cuda.device_count() == 1:
            #device = torch.device('cuda:0')
        #else:
            #assert False, "There should be either one or two devices available"
        device = torch.device('cuda')
        model = model.to(device)
        
        
        # Check that there are no exceptions in the forward pass
        
        model.train()
        ###
        images = torch.rand(2, 3, 256, 256)
        images = list(image.to(device) for image in images)

        boxes = np.array([[[0.29, 0.10, 0.43, 0.58],[0.24, 0.17, 0.33, 0.48]],[[0.04, 0.10, 0.63, 0.58],[0.29, 0.16, 0.43, 0.78]]])
        boxes = boxes.astype(np.float32)
        boxes = torch.from_numpy(boxes)
        boxes = [box.to(device) for box in boxes]

        labels = torch.randint(1, 2, (2, 2))
        labels = [label.to(device) for label in labels]

        height = torch.rand(2, 1, 256, 256)
        height = [x.to(device) for x in height]

        targets = []
        for i in range(len(images)):
            d = {}
            d['boxes'] = boxes[i]
            d['labels'] = labels[i]
            targets.append(d)
        
        batch_losses, batch_predictions = model(images, height, targets)
        print(batch_predictions)
        ###
        # Test the output shapes of the feature maps
        test_img = torch.randn(3, 256, 256)
        test_height = torch.randn(1, 256, 256)
        targets = None
        model.eval()
        out = model([test_img], [test_height], targets)

if __name__ == "__main__":
    print("Running Tests:")
    unittest.main()


