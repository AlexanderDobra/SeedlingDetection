import copy
import logging
import math
import pathlib
import re

import torch
from pycocotools.coco import COCO
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader

from mean_average_precision import MetricBuilder
from models.draw import draw_boxes

#import src.util.util as utils
import util.util as utils
import pandas as pd
#from src.data.data_classes import SeedlingDataset
from data.data_classes import SeedlingDataset
from sklearn.model_selection import train_test_split
import torch.optim.adam
from pycocotools.cocoeval import COCOeval
import sklearn.metrics
import numpy as np
import mlflow

import matplotlib.pyplot as plt
from models.MDE import MDE

MAP_IND = 1 # The index in all of the stats generated by COCOEval of the MAP @ IoU = 50 - our target
DEFAULT_NUM_WEIGHTS_TRACKED = 10
module_path = pathlib.Path(__file__).parent
base_dir = module_path.parent.parent.absolute()
logger = logging.getLogger(__name__)

depthmodel = torch.load('BeraSeedlings/models/ENDMDE101false3.pth')##'jason/src/models/48lastMDE101true3.pth'
depthmodel.to('cuda')
depthmodel.eval()

def get_dataloader(params):
    # First split into train and validation
    if "data_file" in params:
        train_file = params["data_file"]["train_file"]
    else:
        train_file = params["train_file"]
    if "train_neg_ratio" in params:
        assert "test_neg_ratio" in params
        # Use regular expressions to replace the existing ratio with the ratio being tested
        train_file = re.sub(r"(.*neg)_[^_]*_(.*\.csv)", fr"\1_{params['train_neg_ratio']}_\2", train_file)
    train_file_path = base_dir.joinpath(train_file)
    all_train = pd.read_csv(train_file_path)
    train, valid = train_test_split(all_train, test_size=params["valid_ratio"], random_state=1)
    train_dataset = SeedlingDataset(train)
    valid_dataset = SeedlingDataset(valid)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params["train_batch_size"],
        shuffle=True,
        num_workers=params["dataloader_num_workers"],
        collate_fn=utils.collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=params["eval_batch_size"],
        shuffle=False,
        num_workers=params["dataloader_num_workers"],
        collate_fn=utils.collate_fn
    )
    print(f'length of train dataset: {len(train_dataset)}')
    print(f'length of valid dataset: {len(valid_dataset)}')
    return train_dataloader, valid_dataloader

#TODO: Pass extra params for optimiser
def get_optimiser(model_params, params):
    if params["optimiser"] == "ADAM":
        optimiser = torch.optim.Adam(model_params, lr=params["learning_rate"])
    elif params["optimiser"] == "SGD":
        optimiser = torch.optim.SGD(model_params, lr=params["learning_rate"], momentum=params["momentum"],
                                    weight_decay=params["weight_decay"])
    return optimiser


# TODO: This could be more complicated later
def get_device(params):
    if "device" in params:
        return torch.device(params["device"])
    # TODO: Kind of a hack to deal with the two machines (my laptop and Rhodos)
    if torch.cuda.device_count() == 2:
        device = torch.device('cuda:1')
    elif torch.cuda.device_count() == 1:
        device = torch.device('cuda')
    else:
        assert False, "There should be either one or two devices available"
    return device


@utils.block_print
def get_MAP(gt_anns, det_anns, img_size):
    """Return the MAP at IoU=05"""
    coco_gt = get_dataset(gt_anns, img_size)
    coco_det = get_dataset(det_anns, img_size)
    coco_eval = COCOeval(coco_gt, coco_det, iouType="bbox") #64 (potted plant) 56 (broccoli)     ###cocoEval.params.catIds = [1] ##testing right now
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[MAP_IND]

@utils.block_print
def get_dataset(img_anns, img_size):
    """
    Create a COCO api object from a list of bounding boxes.
    The input takes the following format:
    {img_id_1: {"boxes": [N, 4], "scores": [N]}, img_id_M: {...}}

    That is a dictonary containing the annotations for M images, each of which have N separate bounding boxes stored as
    a tensor. Detections also have scores attached.
    """
    num_img = len(img_anns)
    images = []
    # Add the image annotations
    for i in range(num_img):
        img_dict = {'id': i, 'height': img_size[0], 'width': img_size[1]}
        images.append(img_dict)
    dataset = {'images': images, 'categories': [], 'annotations': []}
    ann_id = 1
    # Add the box annotations for each image
    for image in img_anns:
        boxes = img_anns[image]["boxes"].tolist()
        if "scores" in img_anns[image]:
            scores = img_anns[image]["scores"].tolist()
            assert len(scores) == len(boxes)
        else:
            scores = []
        for idx, box in enumerate(boxes):
            if not box:
                continue
            assert len(box) == 4
            # They need to be in WH format
            box[2] -= box[0]
            box[3] -= box[1]
            area = box[2] * box[3]
            ann = {"image_id": image, "bbox": box, "category_id": 1, "area": area,
                   "iscrowd": 0, "id": ann_id}
            if scores:
                ann["score"] = scores[idx]
            ann_id += 1
            cat = {"id": 1}
            dataset["annotations"].append(ann)
            dataset["categories"].append(cat)
    if not images:
        return
    coco_ds = COCO()
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def train_one_epoch(model, dataloader, opt, params, test=None):
    device = get_device(params)
    dets = {}
    gts = {}
    num_rows = math.ceil(len(dataloader.dataset) / dataloader.batch_size)
    losses = np.zeros((num_rows,4))
    batch_num = 0
    # Just a placeholder
    img_size = (256, 256)
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
    for batch_images, batch_heights, batch_targets, _ in dataloader:
        batch_size = len(batch_images)
        img_size = batch_images[0].shape
        # Save copies of the targets before they're pushed to the gpu
        batch_gts = {gt["image_id"].item(): copy.deepcopy(gt) for gt in batch_targets}
        gts.update(batch_gts)

        batch_images = list(img.to(device) for img in batch_images)

        batch_heights = list(depthmodel(img.unsqueeze(0)*255).squeeze().unsqueeze(0) for img in batch_images)
        
        #print(batch_heights[0].shape)
        #for x in range(len(batch_images)-2):
            #batch_images[x] = torch.zeros(3, 256, 256).to(device)
        #batch_heights = list(torch.zeros(1, 256, 256).to(device) for img in batch_heights)

        batch_targets = ([{k: v.to(device) for k, v in t.items()} for t in batch_targets])
        batch_losses, batch_predictions = model(batch_images, batch_heights, batch_targets)

        

        ''' for labels
        length = len(batch_images)
        for y in range(length):
            print(y, 'batch:', batch_predictions[y]['labels'])
        '''

        if test is not None:
            
            length = len(batch_images)

            for y in range(length):
                #print(y, 'batch:', batch_predictions[y]['labels'])
                # prediction scores & bounding boxes
                pred_scores = batch_predictions[y]['scores'].detach().cpu().numpy()
                pred_bboxes = batch_predictions[y]['boxes'].detach().cpu().numpy()

                #saveimage
                '''
                savename = batch_num*length+y
                imagetransform = batch_images[y].cpu().numpy().transpose(1, 2, 0)
                draw_boxes(batch_targets[y]['boxes'].cpu(), pred_bboxes, imagetransform, savename)
                '''
                #number of prediction & real boxes for this image
                x = len(pred_bboxes)
                x2 = len(batch_targets[y]['boxes'])
                
                #print(batch_targets[y]['boxes'])
                #print('length:', x2)

                #concatenate targetboxes for MAP
                if x2 == 0:
                    alltargets = np.array([])#np.zeros((1, 7))
                else:
                    alltargets = np.hstack((batch_targets[y]['boxes'].cpu(), [[0]]*x2, [[0]]*x2, [[0]]*x2))
                
                if x == 0:
                    metric_fn.add(np.array([]), alltargets) #for empty predictions
                else:
                    allpredictions = np.hstack((pred_bboxes, [[0]]*x, pred_scores[...,None]))
                    metric_fn.add(allpredictions, alltargets)

        batch_loss = sum(batch_losses.values())
        # Check if we're training here - if so we should update the gradients
        if opt:
            batch_loss.backward()
            # Gradient clipping
            if "clip" in params:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
            opt.step()
            opt.zero_grad()
        # Push the results back to cpu
        batch_outputs = [{k: v.detach().to("cpu") for k, v in t.items()} for t in batch_predictions]
        # Map IDs to outputs
        res = {target["image_id"].item(): output for target, output in zip(batch_targets, batch_outputs)}
        dets.update(res)
        # Push the losses back to the cpu
        if batch_loss:
            # Record the individual loss components
            losses[batch_num, :] = [val.item() * batch_size for val in batch_losses.values()]
        batch_num += 1
    # Now we average over the total number of values to get the average loss per sample
    av_loss = np.sum(losses, 0) / len(dataloader.dataset)
    # Convert back to dict with named losses
    named_losses = {}
    for ind, loss_name in enumerate(batch_losses.keys()):
        named_losses[loss_name] = av_loss[ind]
    named_losses["loss_total"] = np.sum(av_loss)
    # Sometimes its nice to comment out training to test the pipeline. This is for that case. Otherwise gts should always
    # be populated
    if gts:
        # An now we calculate the MAP
        MAP = get_MAP(gts, dets, img_size)
    else:
        MAP = 0
    if test is not None:
        print('new package MAP COCO: ', metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP'])
    return named_losses, MAP

def test_model(model, params):
    if "data_file" in params:
        test_file = params["data_file"]["test_file"]
    else:
        test_file = params["test_file"]
    # Handle negative ratios here
    if "test_neg_ratio" in params:
        assert "train_neg_ratio" in params
        # Use regular expressions to replace the existing ratio with the ratio being tested
        test_file = re.sub(r"(.*neg)_[^_]*_(.*\.csv)", fr"\1_{params['test_neg_ratio']}_\2", test_file)
    test_file_path = base_dir.joinpath(test_file)
    test_data = pd.read_csv(test_file_path)
    test_dataset = SeedlingDataset(test_data)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=params["eval_batch_size"],
        shuffle=False,
        num_workers=params["dataloader_num_workers"],
        collate_fn=utils.collate_fn
    )
    print(f'length of test dataset: {len(test_dataset)}')
    device = get_device(params)
    _, test_MAP = train_one_epoch(model.to(device), test_dataloader, False, params, True)
    return test_MAP


def fit(params):
    """
    Generic model fitting function
    """
    # Hack to get the RCNN to always return both the predictions and losses
    torch.jit.is_scripting = lambda: True
    device = get_device(params)
    train_dataloader, valid_dataloader = get_dataloader(params)
    model_path = base_dir.joinpath(params["base_model_path"])
    model = torch.load(model_path).to(device)
    opt = get_optimiser(model.parameters(), params)
    # Model saving vars
    best_MAP = -1
    best_model = None
    best_model_epoch = -1
    # Early stopping vars
    worse_model_count = 0
    best_valid_loss = math.inf
    # For tracking of a sample of weights
    weight_sample = None
    num_weights_tracked = params.pop("num_weights_tracked", DEFAULT_NUM_WEIGHTS_TRACKED)
    try:
        for epoch in range(params["epochs"]):
            model.train()
            train_av_losses, _ = train_one_epoch(model, train_dataloader, opt, params)
            # The validation needs to stay in training mode to get the validation LOSS - which will be used for early stopping
            with torch.no_grad():
                valid_av_losses, _ = train_one_epoch(model, valid_dataloader, False, params)
            model.eval()
            _, valid_MAP = train_one_epoch(model, valid_dataloader, False, params, True)
            train_MAP = -1
            #### Logging
            metrics = {}
            # TODO This is flawed. The metric is overwritten each time
            # Log new weights
            for param in model.get_new_weights():
                new_weights = torch.flatten(param)
                # We only want to take a small sample of the weights
                if weight_sample is None:
                    inds = np.arange(len(new_weights))
                    weight_sample = np.random.choice(inds, num_weights_tracked, replace=False)
                # Now record the sampled weights
                for i, sample_ind in enumerate(weight_sample):
                    new_weight = new_weights[sample_ind]
                    metrics[f"new_weight_{i}"] = new_weight.item()
            # Log the loss components
            for loss_name, av_loss in train_av_losses.items():
                metrics[f"train_{loss_name}"] = av_loss
            for loss_name, av_loss in valid_av_losses.items():
                metrics[f"valid_{loss_name}"] = av_loss
            # _, train_MAP = train_one_epoch(model, train_dataloader, False, params)
            metrics["train_MAP"] = train_MAP
            metrics["valid_MAP"] = valid_MAP
            ###mlflow.log_metrics(metrics, epoch)
            logger.log(logging.INFO,
                       f"EPOCH {epoch} valid loss: {valid_av_losses['loss_total']:.8f} | valid MAP: {valid_MAP:.3f} | train loss: "
            f"{train_av_losses['loss_total']:.8f} | train MAP: {train_MAP:.3f}")
            # keep a copy of the best model
            if valid_MAP > best_MAP:
                #TODO: This would be much better with the state dict
                # e.g. best_model_state_dict = {k:v.to('cpu') for k, v in model.state_dict().items()}
                logger.log(logging.INFO, f"New best model in epoch {epoch} with valid MAP score of {valid_MAP}")
                model = model.to("cpu")
                best_model = copy.deepcopy(model)
                model = model.to(device)
                best_model_epoch = epoch
                best_MAP = valid_MAP
            # Now implement early stopping
            if valid_av_losses['loss_total'] > best_valid_loss:
                worse_model_count += 1
                if not "patience" in params:
                    params["patience"] = params["epochs"]
                if worse_model_count >= params["patience"]:
                    logger.log(logging.INFO, f"Stopped training early at epoch {epoch}")
                    break
            else:
                best_valid_loss = valid_av_losses['loss_total']
                worse_model_count = 0
    except Exception as err:
        ###mlflow.log_metric("error", True)
        if params["develop"]:
            # If we're in development we want errors to halt execution (they can be silly errors)
            raise err
        else:
            # During deployment we want the model testing to be robust to unexpected errors
            logger.exception(err)
    ###mlflow.log_metric("best_epoch", best_model_epoch)
    # Test the final model on the test set
    if best_model:
        test_MAP = test_model(best_model, params).item()
    else:
        test_MAP = -1
    ###mlflow.log_metric("test_MAP", test_MAP)
    logger.log(logging.INFO, f"Final test score of model is {test_MAP}")
    return best_model, best_model_epoch, test_MAP
