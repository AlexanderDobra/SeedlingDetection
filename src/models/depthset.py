#Uses data from lane 464, 460, 466 and new CHM (Cannopy-height-model)
#some Labels (XML) were deleted, because they didn't contain any Labels
#FOLDERS USED: labels, ortho, chm

import os
import numpy as np
import torch
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
from models.depthutils import get_edges, minmax_over_nonzero


class SeedlingDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.annotation_files = os.listdir(os.path.join(root_dir, 'labels'))
        self.dm_dim = (256, 256)
        
    def __len__(self):
        return len(self.annotation_files)
    
    def __getitem__(self, index):
        annotation_file = self.annotation_files[index]
        boxes, ortho_name, chm_name = self.read_annotation(os.path.join(self.root_dir, 'labels/', annotation_file))
        
        #images
        image = cv2.imread(os.path.join(self.root_dir, 'ortho/', ortho_name))
        edges = get_edges(image, self.dm_dim)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #labels (depth)
        label = np.array(Image.open(os.path.join(self.root_dir, 'chm/', chm_name))) ###---potentiyl type issue!

        #calc. range
        range = np.array([np.min(label[np.nonzero(label)]), np.max(label)]) ###--- why nonzero??? (better: arr[arr != 0])
        
        #if normalisation (local):
        label = minmax_over_nonzero(label)

        mask = (label >= 0).astype(int)  # 0 is smallest after minmax

        #if interpolate = TRUE
        if np.min(mask) == 0:
            label = interpolate_on_missing(label * mask)

        return {'image': image, 'depth': label, 'mask': mask, 'edges': edges, 'range_min': range[0], 'range_max': range[1]}
        
    def read_annotation(self, file):    
        tree = ET.parse(file)
        root = tree.getroot()
        
        filename = root.find('filename').text
        site = filename.split('_')[1]
        cut = filename.split('cut-')[1][:-4]
        ortho_name = 'site_{}_201710_030m_ortho_als11_3channels_buffer_removed_cut-{}.tif'.format(site, cut)
        chm_name = 'site_{}_201710_CHM10cm_large_buffer_removed_cut-{}.tif'.format(site, cut)

        list_with_all_boxes = []

        for boxes in root.iter('object'):
            ymin, xmin, ymax, xmax = None, None, None, None
            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)
            list_with_single_boxes = [xmin, ymin, xmax, ymax]
            list_with_all_boxes.append(list_with_single_boxes)

        return list_with_all_boxes, ortho_name, chm_name