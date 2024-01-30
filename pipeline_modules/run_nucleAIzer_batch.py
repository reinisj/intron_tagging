#!/usr/bin/env python3

import argparse

import os
import pandas as pd
import numpy as np

from tqdm import tqdm
from imageio import imread,imwrite
from skimage.util import img_as_ubyte
from nucleaizer_backend.mrcnn_interface.predict import MaskRCNNSegmentation

from utils.image_handling import load_img

# load arguments from command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('-i', "--imagelist", required=True)
parser.add_argument('-N', "--nuclei_diameter", type=int, default=60)
parser.add_argument('-m', "--model", required=True) # path to saved weights for nucleaizer
parser.add_argument('-n', "--minq", type=float, default=0.05)
parser.add_argument('-x', "--maxq", type=float, default=0.9975)
parser.add_argument('-g', "--gamma", type=float, default=None)   
parser.add_argument('-b', "--background", nargs=5, type=int, default=[0,0,0,0,0]) # estimated background intensities of the 5 channels
parser.add_argument('-f', "--first", type=int, default=0)
parser.add_argument('-l', "--last", type=int)

args = parser.parse_args()

# load list of images to segment
# table which contains absolute paths (URL_x), folder paths (PathName_x), and filenames (FileName) of the input and output images TIFFs
imagelist = pd.read_csv(args.imagelist)

nuclei_diameter = args.nuclei_diameter
model_path = args.model
minq = args.minq
maxq = args.maxq
gamma = args.gamma
channels_background = args.background

# 0-based indexing; i.e. -f 0 -l 5  = [0,1,2,3,4]
first = args.first
last = args.last

if not last or last > imagelist.shape[0]:
    last = imagelist.shape[0]
assert first <= last

# get the subset of images we want to segment
imagelist_batch = imagelist.iloc[first:last]

# instantiate the nucleAIzer model
nucleaizer_instance = MaskRCNNSegmentation(model_path, default_image_size=2048, trained_object_size=nuclei_diameter)

# segment nuclei (ch5)
print("Segmenting nuclei ...")
paths = [x[5:] for x in imagelist_batch["URL_miRFP"]]
to_segment = [load_img(path, channels_background[4], minq, maxq, gamma) for path in paths]
masks = [nucleaizer_instance.segment(img_as_ubyte(x))[0] for x in tqdm(to_segment)]
for i, mask in enumerate(masks):
    imwrite(imagelist_batch.loc[first+i,"URL_segmented_nuclei_mask"][5:], masks[i])

