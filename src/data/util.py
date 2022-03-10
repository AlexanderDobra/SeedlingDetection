import contextlib
import os
import random

import sys
import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import gc
from types import ModuleType
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    return tuple(zip(*batch))


def get_filenames(directory, ext, keep_ext=False, full_path=False):
    """
    Get a list of file names without extensions within the directory
    :param directory: The directory to search in
    :return: A list of filenames stripped of extensions
    """
    #TODO: Correc tthis to using pathlib

    # First check if the directory exists
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        raise NotADirectoryError(f"{directory} is not a directory")
    # A quick way to get all files in directory
    (_, _, img_paths) = next(os.walk(directory))
    # Strip the file extensions
    img_paths = np.array(img_paths)
    # Check for empty results
    if not len(img_paths):
        return img_paths
    # Only match values with this extension
    img_paths = img_paths[np.char.endswith(img_paths, ext)]
    if not keep_ext:
        img_paths = np.char.rstrip(img_paths, ext)
    if full_path:
        img_paths = np.char.add(directory+"/", img_paths)
    return img_paths

def plot_image(image, boxes=[]):
    """
    Plot the boxes on the given image in matplotlib
    """
    assert len(image.shape) == 3 and 3 in image.shape, "We expect a three channel image"
    # Get the inputs in the correct format
    image = image.cpu()
    boxes = boxes.cpu().numpy().astype(np.int32)
    # Rearrange the image so that the rgb channel is last
    rgb_dim = image.shape.index(3)
    new_order = list(range(3))
    if rgb_dim == 0:
        new_order = [1, 2, 0]
    elif rgb_dim == 1:
        new_order = [0, 1, 2]
    # We need a copy of the numpy array (we don't want to modify the original)
    image = image.permute(new_order).numpy().copy()
    # Now make the plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    box_colour = (220, 0, 0)
    line_thickness = 3
    # Augment the image with boxes
    for box in boxes:
        cv2.rectangle(image,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      box_colour, line_thickness)
    # Plot the image
    ax.set_axis_off()
    ax.imshow(image)
    plt.show()


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def extract_base_id(file_string):
    #Remove a path if there is one
    file_string = os.path.basename(file_string)
    file_string = os.path.splitext(file_string)[0]
    file_string = file_string.replace("_CHM10cm_large_buffer_removed_cut", "")
    file_string = file_string.replace("_030m_ortho_als11_3channels_buffer_removed_cut", "")
    file_string = file_string.replace("_030m_ortho_als11_3channels_cut", "")
    return file_string


def set_seed(seed):
    # https://pytorch.org/docs/stable/notes/randomness.html for more info
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def block_print(func):
    """A function decorator to block printing within the function"""
    def non_print_function(*args, **kwargs):
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            result = func(*args, **kwargs)
        return result
    return non_print_function


def get_alive_tensors():
    alive_tensors = []
    for obj in gc.get_objects():
        # if type(obj) is ModuleType or type(obj) is torch.jit.annotations.Module:
        #     continue
        # try:
        #     torch.is_tensor(obj) #or (hasattr(obj, 'data') and torch.is_tensor(obj.data))
        # except:
        #     # This is just for investigation so it should be OK
        #     continue
        if torch.is_tensor(obj):
            alive_tensors.append(obj)
    return alive_tensors


def alive_tensors_string():
    """
    Get a string displaying a list of the currently alive tensors
    """
    tensors_strings = [f"Name: {type(obj)}, Size: {obj.size()}" for obj in get_alive_tensors()]
    return "\n".join(tensors_strings)
