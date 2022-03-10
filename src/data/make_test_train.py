import math
import pathlib

import cv2


#from src.data.data_classes import SeedlingDataset
#import src.util.util as utils
import util as utils
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
module_path = pathlib.Path(__file__).parent
base_dir = module_path.parent.parent.absolute()
"""
This is a script to generate a csv file with image and label file names for a train/test split"""
WHITE_CUTOFF = 0.05
TEST_SIZE = 0.2
DIR_INPUT = base_dir / "data" #BeraSeedlings MNIST
print(DIR_INPUT)
im_dir = DIR_INPUT / "processed" #ortho
label_dir = DIR_INPUT / "raw2/labels" #labels
# Find all image and label files in directories
im_ids = pd.DataFrame(utils.get_filenames(im_dir, ".tif"), columns=["im_filename"])
label_ids = pd.DataFrame(utils.get_filenames(label_dir, ".xml"), columns=["label_filename"])
# Transform the names so they match up with each other
height_ids = im_ids[im_ids["im_filename"].str.contains("CHM10cm")]
height_ids = height_ids.rename({"im_filename": "height_filename"}, axis=1)
colour_ids = im_ids[im_ids["im_filename"].str.contains("030m")]
height_ids["id"] = height_ids["height_filename"].map(utils.extract_base_id)
colour_ids["id"] = colour_ids["im_filename"].map(utils.extract_base_id)
label_ids["id"] = label_ids["label_filename"].map(utils.extract_base_id)
# Remove images that have white sections in them
colour_ids.loc[:, "blank"] = False
# If more than a threshold of pure white is present then the image is excldued (because lane 466 is at 45 degress) and
# contains a lot of white tiles
for im_filename in colour_ids["im_filename"]:
    full_filename = f"{str(im_dir)}/" + im_filename + ".tif"
    im = cv2.imread(full_filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    matched = (im == 255).all(axis=2).flatten()
    if matched.sum() / len(matched) > WHITE_CUTOFF:
        row = colour_ids["im_filename"] == im_filename
        colour_ids.loc[row, "blank"] = True
# Remove the rows with blank images
colour_ids = colour_ids.loc[~colour_ids["blank"]]
# Get the files that are shared across all
#TODO: This is strange. Height and colour don't have the same number of images and the smaller one (height) isn't even a subset.
union_ids = height_ids.merge(colour_ids, on="id", how="outer").merge(label_ids, on="id", how="outer")
# Warn for ids that are not shared
missing_labels = union_ids["label_filename"].isna().to_numpy().nonzero()[0]
for missing_label_ind in missing_labels:
    missing_sample = union_ids["id"][missing_label_ind]
    logger.warning(f"No label file present for sample {missing_sample}")
missing_height = union_ids["height_filename"].isna().to_numpy().nonzero()[0]
for missing_height_ind in missing_height:
    missing_sample = union_ids["id"][missing_height_ind]
    logger.warning(f"No height file present for sample {missing_sample}")
missing_ims = union_ids["im_filename"].isna().to_numpy().nonzero()[0]
for missing_im_ind in missing_ims:
    missing_sample = union_ids["id"][missing_im_ind]
    logger.warning(f"No image file present for sample {missing_sample}")
# Remove entries that don't have both height and colour images

valid_rows = ~union_ids["height_filename"].isna() & ~union_ids["im_filename"].isna()
shared_ids = union_ids.loc[valid_rows]
shared_ids["label_filename"] = f"{str(label_dir)}/" + shared_ids['label_filename'] + ".xml"
shared_ids["im_filename"] = f"{str(im_dir)}/" + shared_ids['im_filename'] + ".tif"
shared_ids["height_filename"] = f"{str(im_dir)}/" + shared_ids['height_filename'] + ".tif"
# Split into train test
neg_ratios = [0, 0.5]
# Maintain consistent splits even when renewed
np.random.seed(42)
for i in range(5):
    train, test = train_test_split(shared_ids, test_size=TEST_SIZE, random_state=i)
    train_neg_mask = train["label_filename"].isna()
    train_neg_inds = train_neg_mask.to_numpy().nonzero()[0]
    test_neg_mask = test["label_filename"].isna()
    test_neg_inds = test_neg_mask.to_numpy().nonzero()[0]
    # Make different files for different ratios of negative to positive samples
    for neg_ratio in neg_ratios:
        # Take a sample of the negative rows to remove
        if neg_ratio == "all":
            num_neg_train = len(train_neg_inds)
            num_neg_test = len(test_neg_inds)
        else:
            num_pos_train = (~train_neg_mask).sum()
            num_neg_train = round(num_pos_train * neg_ratio)
            num_pos_test = (~test_neg_mask).sum()
            num_neg_test = round(num_pos_test * neg_ratio)
        # Now sample negative samples to include
        sample_train_negs = np.random.choice(train_neg_inds, size=num_neg_train, replace=False)
        train_keep = ~train_neg_mask
        train_keep.iloc[sample_train_negs] = True
        balanced_train = train.loc[train_keep]
        balanced_train.to_csv(DIR_INPUT / f"464_neg_{neg_ratio}_train_{i}.csv")
        # Now the test set
        sample_test_negs = np.random.choice(test_neg_inds, size=num_neg_test, replace=False)
        test_keep = ~test_neg_mask
        test_keep.iloc[sample_test_negs] = True
        balanced_test = test.loc[test_keep]
        balanced_test.to_csv(DIR_INPUT / f"464_neg_{neg_ratio}_test_{i}.csv")
#

# Write the normal files without negatives as well
shared_ids = shared_ids.loc[~shared_ids["label_filename"].isna()]
for i in range(5):
    train, test = train_test_split(shared_ids, test_size=TEST_SIZE, random_state=i)
    train.to_csv(DIR_INPUT / f"464_train_{i}.csv")
    test.to_csv(DIR_INPUT / f"464_test_{i}.csv")
