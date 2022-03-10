import math
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import logging
import pandas as pd
#import src.util.util as utils
import util.util as utils
import xml.etree.ElementTree as ET
import albumentations as A
import numpy as np


COLOUR_MAX = 255.0
HEIGHT_MAX = 255.0
class SeedlingDataset(Dataset):
    def __init__(self, datafiles):
        super().__init__()
        if type(datafiles) is not pd.DataFrame:
            # This should be a csv file to read in results
            datafiles = pd.read_csv(datafiles)
        # datafiles structure:
        self.datafiles = datafiles
        self.img_boxes = self._read_boxes()

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            tuple_list = [self[i] for i in range(start, stop, step)]
            unzipped = list(zip(* tuple_list))
            return unzipped
        elif isinstance(key, int):
            return self.get_value(key)
        else:
            raise TypeError('Invalid argument type: {}'.format(type(key)))

    def get_value(self, index: int):
        id = self.datafiles["id"].iloc[index]
        records = self.img_boxes[self.img_boxes["id"] == id]
        # Read in the colour image
        paths = self.datafiles.loc[self.datafiles["id"] == id, ["im_filename", "height_filename"]]
        assert len(paths) == 1, "There should only be one match"
        img_path = paths.iloc[0, 0]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        height_path = paths.iloc[0, 1]
        height_image = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
        # Check that both images are in 8 bit format (as much as possible)
        if max(image.flatten()) > 256:
            raise RuntimeError(f"{img_path} is not in 8 bit format")
        if max(height_image.flatten()) > 256:
            raise RuntimeError(f"{height_path} is not in 8 bit format")
        assert image is not None, "The image should exist"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Scale to 0-1 space
        image = image / COLOUR_MAX
        image = torch.tensor(image, dtype=torch.float)
        # Rearrange image - pytorch standard models expect channel dimension first
        image = image.permute((2, 0, 1))
        # Read in the height map
        height_image = height_image / HEIGHT_MAX
        height_image = torch.tensor(height_image, dtype=torch.float)
        height_image = height_image.unsqueeze(0)
        # Now the labels
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        # TODO: Put this into coco format straight away? Might be some overhead with uploading unncessary stuff to the GPU...
        target = {}
        target["image_id"] = torch.tensor(index)
        target["boxes"] = torch.tensor(records[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float)
        areas = (records["xmax"] - records["xmin"]) * (records["ymax"] - records["ymin"])
        target["area"] = torch.tensor(areas.values, dtype=torch.float)
        target["labels"] = labels
        target["iscrowd"] = iscrowd
        return image, height_image, target, id

    def _read_boxes(self):
        # First get a list of boxes for each file
        file_boxes = [self._get_boxes(label_file) for label_file in self.datafiles["label_filename"]]
        df = pd.concat(file_boxes)
        return df

    def _get_boxes(self, filename):
        if pd.isna(filename):
            # Empty filenames simply contain no boxes
            return pd.DataFrame(columns=["id", "xmin", "xmax", "ymin", "ymax", "truncated"])
        # The xml parser reads in a tree structure
        this_id = utils.extract_base_id(filename)
        id_entries = []
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        truncateds = []
        root = ET.parse(filename).getroot()
        box_elems = root.findall("object")
        for ind, box_elem in enumerate(box_elems):
            id_entries.append(this_id)
            truncateds.append(box_elem.find("truncated").text == "1")
            box = box_elem.find("bndbox")
            names = ["xmin", "xmax", "ymin", "ymax"]
            vals = np.array([box.find(name).text for name in names]).astype(np.float)
            xmins.append(vals[0])
            xmaxs.append(vals[1])
            ymins.append(vals[2])
            ymaxs.append(vals[3])
        # Create the final dataframe
        df = pd.DataFrame(zip(id_entries, xmins, xmaxs, ymins, ymaxs, truncateds),
                      columns=["id", "xmin", "xmax", "ymin", "ymax", "truncated"])
        return df

    def __len__(self) -> int:
        return len(self.datafiles)