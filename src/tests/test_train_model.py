import copy
from unittest import TestCase
from src.models.train_model import *


class MapCalcTest(TestCase):

    def test_get_map(self):
        """
        Make a coco dataset with one image, two true bounding boxes and three predicted bounding boxes
        """
        img_size = (100,100)
        true_boxes = torch.Tensor([[0, 0, 10, 10],
                    [90, 90, 100, 100]
                                        ])
        true_anns = {0: {"boxes": true_boxes}}
        # Test no predicted boxes (returns -1 for invalid)
        pred_boxes = torch.Tensor([])
        pred_scores = torch.Tensor([])
        pred_anns = {0: {"boxes": pred_boxes, "scores": pred_scores}}
        self.assertAlmostEqual(get_MAP(true_anns, pred_anns, img_size), 0)


        # Test now the perfect case - You can get perfect recall for all data points
        pred_boxes = torch.Tensor([[0, 0, 9, 9],
                      [91, 91, 100, 100],
                      [40, 40, 60, 60]
                     ])
        pred_scores = torch.Tensor([0.9, 0.8, 0.5])
        pred_anns = {0: {"boxes": pred_boxes, "scores": pred_scores}}
        MAP = get_MAP(true_anns, pred_anns, img_size)
        self.assertAlmostEqual(MAP, 1)


        # Now test a different ordering with one misclassification in the middle
        pred_scores = torch.Tensor([0.9, 0.5, 0.8])
        pred_anns = {0: {"boxes": pred_boxes, "scores": pred_scores}}
        MAP = get_MAP(true_anns, pred_anns, img_size)
        self.assertAlmostEqual(MAP, 5.0 / 6, places=2)
