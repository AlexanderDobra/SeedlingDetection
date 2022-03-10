import math
import pathlib
import gdal
import matplotlib
import os
import skimage as skim
import skimage.transform
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import rasterio

EPS = 1e-1

module_path = pathlib.Path(__file__).parent
base_dir = module_path.parent.parent.resolve()


def extract_matching_image(big_image_path, small_image_path):
    """
    Take a large geotif image and a small geotif image and extract the slice corresponding to the small image from the
    large image
    :return: The sliced section of the larger image (not necesarily the same pixel size)
    """
    big_image_path = pathlib.Path(big_image_path)
    small_image_path = pathlib.Path(small_image_path)
    big_data = gdal.Open(str(big_image_path))
    small_data = gdal.Open(str(small_image_path))
    gt_big = big_data.GetGeoTransform()
    gt_small = small_data.GetGeoTransform()
    # Get the coordinates of the bottom right corner of the image
    small_size = (small_data.RasterXSize, small_data.RasterYSize)
    br_coords = pixel_to_geo(gt_small, small_size[0], small_size[1])
    tl_coords = (gt_small[0], gt_small[3])  # The top left coordinates are stored in the geotransform
    # Now we calculate the corresponding pixel values in the large file
    tl_big_pixel = geo_to_pixel(gt_big, tl_coords[0], tl_coords[1])
    br_big_pixel = geo_to_pixel(gt_big, br_coords[0], br_coords[1])
    width = br_big_pixel[0] - tl_big_pixel[0]
    height = br_big_pixel[1] - tl_big_pixel[1]
    # Now we can read in the relevant pixels of the large file
    big_slice = big_data.ReadAsArray(tl_big_pixel[0], tl_big_pixel[1], width, height)
    # Normalise to 0 value
    big_slice = big_slice - np.min(big_slice)
    # Alter the geotransform for the output file
    gt_new = list(gt_big)
    gt_new[0] = tl_coords[0]
    gt_new[3] = tl_coords[1]
    # Assumes the standard naming pattern
    lane_num = small_image_path.stem[5:8]
    # Now we save the file
    out_filename = f"site_{lane_num}_201710_CHM10cm.tif"
    out_filepath = base_dir / "data" / "interim" / out_filename
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(str(out_filepath), width, height, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(gt_new)
    # TODO: Understand what the projection is
    outdata.SetProjection(big_data.GetProjection())  # sets same projection as input
    outdata.GetRasterBand(1).WriteArray(big_slice)
    outdata.FlushCache()
    return out_filepath

def geo_to_pixel(gt, new_long, new_lat):
    # Read in the geological coordinates of the images
    long = gt[0]  # The latitude of the top left pixel
    long_spacing = gt[1]  # The spacing along the E-W axis
    rot = gt[2]  # The roation (along this axis???)
    lat = gt[3]  # The longitude of teh top left pixel
    rot2 = gt[4]  # Not entirely sure about why there are two rotation params.
    lat_spacing = gt[5]  # The spacing along the N-S axis
    assert not rot and not rot2, "There should be no image rotation"
    pixel_x = math.floor((new_long - long) / long_spacing)
    pixel_y = math.floor((new_lat - lat) / lat_spacing)
    return pixel_x, pixel_y

#TODO: Make it clear that this is reversed (long, lat) to match pixel x,y values
def pixel_to_geo(gt, x, y):
    # Read in the geological coordinates of the images
    long = gt[0]  # The latitude of the top left pixel
    long_spacing = gt[1]  # The spacing along the E-W axis
    rot = gt[2]  # The roation (along this axis???)
    lat = gt[3]  # The longitude of teh top left pixel
    rot2 = gt[4]  # Not entirely sure about why there are two rotation params.
    lat_spacing = gt[5]  # The spacing along the N-S axis
    assert not rot and not rot2, "There should be no image rotation"
    new_lat = lat + y * lat_spacing
    new_long = long + x * long_spacing
    return new_long, new_lat

def convert_to_3_layer(filename):
    # Generate a file with three layers instead of 4
    filename = pathlib.Path(filename)
    file = gdal.Open(str(filename))
    # Metadata
    gt = file.GetGeoTransform()
    proj = file.GetProjection()
    # Choose the rasters that we are interested in
    raster1 = file.GetRasterBand(1)
    raster2 = file.GetRasterBand(2)
    raster3 = file.GetRasterBand(3)
    width = raster1.XSize
    height = raster1.YSize
    arr1 = raster1.ReadAsArray()
    arr2 = raster2.ReadAsArray()
    arr3 = raster3.ReadAsArray()
    arr1 = arr1 // 256
    arr2 = arr2 // 256
    arr3 = arr3 // 256
    # Now write the file
    name_string = filename.stem + "_3channels" + filename.suffix
    out_filepath = base_dir / "data" / "interim" / name_string
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(str(out_filepath), width, height, 3, gdal.GDT_Byte)
    outdata.SetGeoTransform(gt)
    outdata.SetProjection(proj)
    outdata.GetRasterBand(1).WriteArray(arr1)
    outdata.GetRasterBand(2).WriteArray(arr2)
    outdata.GetRasterBand(3).WriteArray(arr3)
    outdata.FlushCache()
    return out_filepath


def resize_files(fn1, fn2):
    """
    Take two geotiff images and resize the second so they are on the same scale
    :return:
    """
    data1 = gdal.Open(str(fn1))
    data2 = gdal.Open(str(fn2))
    if not data1 or not data2:
        return # TODO: This could be better. Maybe an exception should be thrown
    gt1 = data1.GetGeoTransform()
    gt2 = data2.GetGeoTransform()
    # Entries 1 and 5 show the distance moved per pixel in x and y
    scale1x = gt1[1]
    scale1y = gt1[5]
    scale2x = gt2[1]
    scale2y = gt2[5]
    # Now we need the size of the second image to resize it
    x_size = data2.RasterXSize
    y_size = data2.RasterYSize
    # Now we calculate the new scale for the second image
    ratio_x = scale2x / scale1x
    ratio_y = scale2y / scale1y
    # Now we can rescale the image
    new_x = round(x_size * ratio_x)
    new_y = round(y_size * ratio_y)
    # Finally we need to actually write the image
    im_arr = data2.ReadAsArray()
    new_im_arr = skim.transform.resize(im_arr, (new_y, new_x), mode='constant')
    # Calc the new filename
    filename = fn2.stem
    new_filename = f"{filename}_large{fn2.suffix}"
    new_file = base_dir / "data" / "interim" / new_filename
    # The geotransform should now be identical to the first image
    gt = gt1
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(str(new_file), new_x, new_y, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(gt)
    # TODO: Understand what the projection is
    outdata.SetProjection(data2.GetProjection())  # sets same projection as input
    outdata.GetRasterBand(1).WriteArray(new_im_arr)
    outdata.FlushCache()
    return new_file

def split_images(filename, new_path,  slice_w=256, slice_h=256):
    """
    Split a large image into a series of small tiles and save them
    :return:
    """
    image = skim.io.imread(filename)
    im_h = image.shape[0]
    im_w = image.shape[1]
    pos_x = 0
    pos_y = 0
    while pos_y < im_h:
        while pos_x < im_w:
            im_sec = image[pos_y:pos_y+slice_h, pos_x:pos_x+slice_w]
            #TODO: This would be much prettier just in the loop
            if im_sec.shape[0] != slice_h or im_sec.shape[1] != slice_w:
                pos_x += slice_w
                continue
            new_filename = new_path.stem + f"_cut-{pos_x}-{pos_y}" + new_path.suffix
            new_filepath = new_path.parent / new_filename
            skim.io.imsave(new_filepath, im_sec)
            pos_x += slice_w
        pos_y += slice_h
        pos_x = 0


def remove_pad_tl(image_path, shift):
    image = skim.io.imread(image_path)
    im_new_filename = f"{image_path.stem}_buffer_removed"+image_path.suffix
    im_new_path = base_dir / "data" / "interim" / im_new_filename
    if image.ndim == 2:
        image = image[shift + 1:, shift + 1:]
    elif image.ndim == 3:
        image = image[shift + 1:, shift + 1:, :]
    else:
        assert False
    skim.io.imsave(im_new_path, image)
    return im_new_path


def process_images(image_path, height_path, large_height, slice_h=256, slice_w=256, clear=False):
    assert image_path.exists() and height_path.exists()
    image_path = pathlib.Path(image_path)
    height_path = pathlib.Path(height_path)
    # Convert the original lane to a 3-channel image
    new_image_path = convert_to_3_layer(image_path)
    if large_height:
        # Take the corresponding slice out of the large height map for the height data
        new_height_path = extract_matching_image(height_path, new_image_path)
    else:
        new_height_path = height_path
    new_height_path = resize_files(new_image_path, new_height_path)
    # Remove the 35 pixel top-left pad on both images (first 33 are white and there is another 2 pixels off in the
    # xml files)
    im_new_path = remove_pad_tl(new_image_path, 35)
    new_height_path = remove_pad_tl(new_height_path, 35)
    final_dir = base_dir / "data" / "processed"
    # clear all previous files
    if clear:
        prev_files = final_dir.glob("*_cut-*.tif")
        for f in prev_files:
            os.remove(f)
    # Split the images
    im_final_path = final_dir / im_new_path.name
    h_final_path = final_dir / new_height_path.name
    split_images(im_new_path, im_final_path, slice_w, slice_h)
    split_images(new_height_path, h_final_path, slice_w, slice_h)


#############################################################################################################
# Run for our lanes
gdal.UseExceptions()
raw_dir = base_dir / "data" / "raw" / "images"
# The high def data from 464
im_full = raw_dir / "site_464_201710_030m_ortho_als11.tif"
height_path = raw_dir / "old_height_464.tif"
process_images(im_full, height_path, False, clear=True)
# Now get the normal data for all lanes
lane_files = {460: "site_460_201710_030m_ortho_als11.tif",
              464: "site_464_201710_030m_ortho_als11.tif",
              466: "site_466_201710_030m_ortho_als11.tif"}

CHM_path = raw_dir / "KirbyLeafOff2017DSMEntireSite.tif"
for im_path in lane_files.values():
    im_full = raw_dir / im_path
    process_images(im_full, CHM_path, True, clear=False)



#### Assessing alignment


# def get_num_channels(filename):
#     src_ds = gdal.Open(str(filename))
#     if src_ds is not None:
#         return src_ds.RasterCount
#
# def plot_two_images(height_path, colour_path, ax, alpha=0.6, log=False, vmax=18):
#     """Plot a number of layered slices to correct the image shift"""
#     height_1 = gdal.Open(str(height_path))
#     height_1 = height_1.ReadAsArray()
#     # Histogram of heights
#     fig2, ax2 = plt.subplots()
#     ax2.hist(height_1.flatten())
#     fig2.show()
#     if log:
#         height_1 = np.log(height_1)
#     colour = gdal.Open(str(colour_path))
#     colour = colour.ReadAsArray()
#     # Add an alpha channel to the first plot
#     colour = np.moveaxis(colour, 0, -1)
#     # Because we're feeding in float alphas matplotlib needs the values in floats
#     if type(colour[0,0,0]) is np.uint8:
#         colour = colour / 256.0
#     elif type(colour[0,0,0]) is np.uint16:
#         colour = colour / 256.0 ** 2
#     else:
#         assert False, "We can't handle other data types"
#     new_shape = list(colour.shape)
#     new_shape[2] = 1
#     alpha = np.full(new_shape, alpha)
#     colour = np.concatenate((colour, alpha), axis=2)
#     if log:
#         vmax = np.log(vmax)
#     ax.imshow(height_1, vmin=0, vmax=vmax)
#     ax.imshow(colour)
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
# height_path = base_dir / "data" / "interim" / "site_460_GNDDIPC_3m_large_buffer_removed.tif"
# colour_path = base_dir / "data" / "interim" / "site_460_201710_030m_ortho_als11_3channels.tif"
#
# fig, ax = plt.subplots()
# plot_two_images(height_path, colour_path, ax)
# plt.show()
# height_path = base_dir / "data" / "interim" / "site_466_201710_CHM10cm_large_buffer_removed.tif"

