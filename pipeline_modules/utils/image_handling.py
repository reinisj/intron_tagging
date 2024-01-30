import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread,imsave
from skimage.exposure import adjust_gamma
from skimage.util import img_as_ubyte
from skimage.transform import resize as imgresize

def background_subtract(img, value):
    img_subtracted = img.astype("int32").copy()
    img_subtracted = img_subtracted - value
    img_subtracted[img_subtracted < 0] = 0
    return img_subtracted.astype("uint16")

def quantile_normalize(img, minq=0.005, maxq=0.9975):
    img_norm = img.copy()
    img_norm[img > np.quantile(img, maxq)] = np.quantile(img, maxq)
    img_norm[img < np.quantile(img, minq)] = np.quantile(img, minq)
    img_norm = img_norm - np.quantile(img, minq)
    img_norm = img_norm/img_norm.max()
    # bug fix when img_norm can be negative
    img_norm[img_norm<0] = 0
    return img_norm

# loads images, for each image (provided by path) subtracts background (one value per image provided), quantile normalizes
# optionally also performs gamma correction
def load_img(path, background_intensity, minq = 0, maxq = 1, gamma = None):
    img = imread(path)
    img = background_subtract(img, background_intensity)
    img = quantile_normalize(img, minq, maxq)
    if gamma:
        img = adjust_gamma(img, gamma)
    return img

# wrapper around the load_img function, loads different channels for a given FOV
# returns the loaded/preprocessed images as a tensor: (height, width, channel)
def load_image_channels(paths, background_intensities, minq, maxq, gamma = None):
    channels = [load_img(path, bg, minq, maxq, gamma) for path, bg in zip(paths, background_intensities)]
    channels = np.moveaxis(np.array(channels), 0, -1)
    return channels#[:,:,[1,0,2]]

def get_struct_chans(channels):
    m = channels[...,3] # ch4 = mAmetrine
    n = channels[...,4] # ch5 = miRFP

    # heuristics to make miRFP dark red and mAmetrine yellow/greenish
    composite = np.zeros((channels.shape[0],channels.shape[1],3))
    composite[:,:,0] += n + m/1.3
    composite[:,:,1] += n/2.7 + m
    # scale again back to (0,1)
    composite = quantile_normalize(composite)
    return composite

# create overlay for tagged channels
def get_tagged_chans(channels):
    return channels[:,:,[1,0,2]] # ch1 = GFP (green), ch2 = mScarlet (red), ch3 = BFP (blue)

def save_img(filename, image, newsize=None, quality=92):
    if not newsize:
        newsize = image.shape
    # img_as_ubyte to suppress warnings about conversion of the image from float to int in order to save
    if filename.endswith("png"):
        imsave(filename,
           img_as_ubyte(imgresize(image,newsize)))
    else:
        imsave(filename,
           img_as_ubyte(imgresize(image,newsize)),
           quality=quality,
           check_contrast=False)

# subtract proportion of img2 from img1 - bleedthrough correction
def subtract_channel(img1, img2, proportion):
    img1_subtr = img1.astype("int64") - (img2*proportion).astype("int64")
    img1_subtr[img1_subtr <=0] = 0
    return img1_subtr.astype("uint16")
