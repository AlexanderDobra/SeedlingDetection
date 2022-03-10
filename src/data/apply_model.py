import logging
import math
import pathlib
from xml.dom import minidom

import cv2
import torch
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import src.util.util as utils

COLOUR_MAX = 255.0

logger = logging.getLogger(__name__)
module_path = pathlib.Path(__file__).parent
base_dir = module_path.parent.parent.absolute()

def get_img_df(im_dir):
    im_ids = pd.DataFrame(utils.get_filenames(im_dir, ".tif", keep_ext=True), columns=["im_filename"])
    # Transform the names so they match up with each other
    colour_ids = im_ids[im_ids["im_filename"].str.contains("030m")]
    return colour_ids


def detect_img_xml(model_path, device, im_dir):
    """
    Make predictions on the images and write xml files with the output
    """
    model = torch.load(model_path).to(device)
    ims = get_img_df(im_dir)
    for im_file in ims["im_filename"]:
        im_path = im_dir / im_file
        print(im_path)
        image = cv2.imread(str(im_path), cv2.IMREAD_COLOR)
        assert image is not None, "The image should exist"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Scale to 0-1 space
        image = image / COLOUR_MAX
        image = torch.tensor(image, dtype=torch.float)
        # Rearrange image - pytorch standard models expect channel dimension first
        image = image.permute((2, 0, 1))
        image = image.to(device)
        model.eval()
        batch_prediction = model([image], [], None)
        write_pred_xml(batch_prediction, im_path)


def write_pred_xml(prediction, path):
    ann = ET.Element('annotation')
    ET.SubElement(ann, 'folder').text = "images"
    ET.SubElement(ann, 'filename').text = str(path.name)
    ET.SubElement(ann, 'path').text = str(path)
    source = ET.SubElement(ann, 'source')
    ET.SubElement(source, 'database').text = "Unknown"
    size = ET.SubElement(ann, 'size')
    ET.SubElement(size, 'width').text = "256"
    ET.SubElement(size, 'height').text = "256"
    ET.SubElement(size, 'depth').text = "3"
    ET.SubElement(ann, 'segmented').text = "0"
    for ind, box in enumerate(prediction[0]["boxes"]):
        score = prediction[0]["scores"][ind]
        object = ET.SubElement(ann, 'object')
        ET.SubElement(object, 'score').text = str(score)
        ET.SubElement(object, 'name').text = "seedling"
        ET.SubElement(object, 'pose').text = "Unspecified"
        #  # Not sure about this one
        ET.SubElement(object, 'difficult').text = "0"
        bndbox = ET.SubElement(object, 'bndbox')
        pixels = np.round(box.detach().to("cpu").numpy())
        if any((pixels == 0) | (pixels == 256)):
            ET.SubElement(object, 'truncated').text = "1"
        else:
            ET.SubElement(object, 'truncated').text = "0"
        ET.SubElement(bndbox, 'xmin').text = str(pixels[0])
        ET.SubElement(bndbox, 'ymin').text = str(pixels[1])
        ET.SubElement(bndbox, 'xmax').text = str(pixels[2])
        ET.SubElement(bndbox, 'ymax').text = str(pixels[3])
    # Write the file
    doc_str = ET.tostring(ann, encoding='utf8', method='xml')
    pretty_str = minidom.parseString(doc_str).toprettyxml(indent="   ")
    filename = base_dir / "data" / "raw" / "labelling" / (path.stem + ".xml")
    myfile = open(filename, "w")
    myfile.write(pretty_str)
    myfile.close()


model_path = base_dir / "models" / "trained" / "2021-02-04_improve_stability_cv_24.pt"
device = "cuda:0"
im_dir = base_dir / "data" / "processed"
detect_img_xml(model_path, device, im_dir)