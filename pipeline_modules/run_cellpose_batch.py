#!/usr/bin/env python3

import argparse

import os
import pandas as pd
import numpy as np

#import matplotlib.pyplot as plt

from imageio import imread
from PIL import Image
from cellpose import models,io

# load arguments from command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('-i', "--imagelist", required=True)    
parser.add_argument('-m', "--nuclei_model", default="nuclei")
parser.add_argument('-c', "--cells_diameter", type=int, default=80)
parser.add_argument('-f', "--first", type=int, default=0)
parser.add_argument('-l', "--last", type=int)

args = parser.parse_args()

# load list of images to segment
# table which contains absolute paths (URL_x), folder paths (PathName_x), and filenames (FileName) of the input and output images TIFFs
imagelist = pd.read_csv(args.imagelist)

nuclei_model = args.nuclei_model
cell_diameter = args.cells_diameter

# 0-based indexing; i.e. -f 0 -l 5  = [0,1,2,3,4]
first = args.first
last = args.last

if not last or last > imagelist.shape[0]:
    last = imagelist.shape[0]
assert first <= last

# get the subset of images we want to segment
imagelist_batch = imagelist.iloc[first:last]

# segment cells (ch4)
print("Segmenting cells ...")
model = models.Cellpose(model_type='cyto')


imgs = [imread(x[5:]) for x in imagelist_batch["URL_mAmetrine"]]
channels = [0,0] # was [[0,0]] for some reason and that did not work
masks, flows, styles, diams = model.eval(imgs, diameter=cell_diameter, channels=channels, resample=True)
for i, mask in enumerate(masks):
    Image.fromarray(masks[i]).save(imagelist_batch.loc[first+i,"URL_segmented_cells_mask"][5:])

