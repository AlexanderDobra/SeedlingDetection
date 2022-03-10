# This isn't in notebook form at the moment because my jupyter in pycharm is broken and this is really very short. I might
# move it over later if I have the time

# I need to get the averages and sd from each channel in the dataset. At the moment this is hard coded into the model image
# transformer
import torch

from src.data.data_classes import SeedlingDataset
import pandas as pd
import pathlib

module_path = pathlib.Path(__file__).parent
base_dir = module_path.parent.absolute()
test_ims = pd.read_csv(base_dir / "data" / "site_464_201710_30_test.csv")
train_ims = pd.read_csv(base_dir / "data" / "site_464_201710_30_train.csv")
all_ims = pd.concat((test_ims, train_ims))
ds = SeedlingDataset(all_ims)
num_ims = 0
sum_chan = torch.zeros(3)
sum_height = torch.zeros(1)
for im, height, _, _ in ds:
    sum_chan += torch.mean(im, (1,2))
    sum_height += torch.mean(height, (1,2))
    num_ims += 1
av_chan = sum_chan / num_ims
av_height = sum_height / num_ims
# Now to calculate the standard deviation
var_chan = torch.zeros(3)
var_height = torch.zeros(1)
num_ims = 0
for im, height, _, _ in ds:
    # The squared differences for every pixel in the image
    square_sum_chan = (im - av_chan[:, None, None]) ** 2
    # Now take the average here (better not to sum cause we could overflow). Images are all the same size anyway
    var_chan += torch.mean(square_sum_chan, (1,2))
    square_sum_height = (av_height - height) ** 2
    var_height += torch.mean(square_sum_height, (1,2))
    num_ims += 1
var_chan = var_chan / num_ims
var_height = var_height / num_ims
sd_chan = var_chan ** 0.5
sd_height = var_height ** 0.5
means = av_chan.tolist()
means.append(av_height.item())
sds = sd_chan.tolist()
sds.append(sd_height.item())
print(f"Means: {means} | SDs: {sds}")