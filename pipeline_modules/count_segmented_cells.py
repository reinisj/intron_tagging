#!/usr/bin/env python3

import argparse

import pandas as pd
import numpy as np
from skimage.io import imread
import re

def count_objects_above_threshold_size(mask, min_size=0):
    obj_sizes = np.array(np.unique(mask, return_counts=True)).T
    # count objects which are not 0 (background) and that are larger than min_size
    return np.sum(obj_sizes[obj_sizes[:,0] != 0][:,1]> min_size)

# load arguments from command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input_path", required=True)   
parser.add_argument('-o', "--output_path", required=True)
parser.add_argument('-r', "--run_name", default=None)
parser.add_argument('-Nm', "--nuclei_min_size", type=int, default = 0)
parser.add_argument('-Cm', "--cells_min_size", type=int, default = 0)

args = parser.parse_args()

# csv path
imagelist_csv_path = args.input_path
output_path = args.output_path
run_name = args.run_name
nuclei_min_size = args.nuclei_min_size
cells_min_size = args.cells_min_size

imagelist = pd.read_csv(imagelist_csv_path)

print(imagelist_csv_path)
print(len(imagelist), "FOVs", "\n")

# count cells using the mask files
print("Counting all detected cells")
detected_cells = [len(np.unique(imread(URL[5:])))-1 for URL in imagelist["URL_segmented_cells_mask"]]
print("Counting all detected nuclei")
detected_nuclei = [len(np.unique(imread(URL[5:])))-1 for URL in imagelist["URL_segmented_nuclei_mask"]]

# if minimal object sizes are provided, calculate the number of objects above threshold
if nuclei_min_size and cells_min_size:
    print(f"Count cells above {cells_min_size} and nuclei above {nuclei_min_size} area (pixels)")
    cells_nodebris = [count_objects_above_threshold_size(imread(URL[5:]), cells_min_size) for URL in imagelist["URL_segmented_cells_mask"]]
    nuclei_nodebris = [count_objects_above_threshold_size(imread(URL[5:]), nuclei_min_size) for URL in imagelist["URL_segmented_nuclei_mask"]]

print("Counting cells after 1:1 mapping")

filtered_cells = [len(np.unique(imread(URL[5:])))-1 for URL in imagelist["URL_filtered_cells"]]

# add info about the FOV (image)
image_metadata = imagelist[["FOV"]].copy()
image_metadata["run_name"] = run_name
image_metadata.rename(columns={"FOV":"image"}, inplace=True)
image_metadata["row"] = [re.search(r"r(\d+)c(\d+)f(\d+)p", x).groups()[0] for x in image_metadata["image"]]
image_metadata["column"] = [re.search(r"r(\d+)c(\d+)f(\d+)p", x).groups()[1] for x in image_metadata["image"]]
image_metadata["FOV"] = [re.search(r"r(\d+)c(\d+)f(\d+)p", x).groups()[2] for x in image_metadata["image"]]

image_metadata[['row', 'column', 'FOV']] =image_metadata[['row', 'column', 'FOV']].apply(pd.to_numeric)

# get counts
image_metadata["detected_cells"] = detected_cells
image_metadata["detected_nuclei"] = detected_nuclei
image_metadata["filtered_cells"] = filtered_cells
if nuclei_min_size and cells_min_size:
    image_metadata["cells_nodebris"] = cells_nodebris
    image_metadata["nuclei_nodebris"] = nuclei_nodebris

# reorder
if nuclei_min_size and cells_min_size:
    image_metadata = image_metadata[["run_name", "image", "row", "column", "FOV", "detected_cells", "detected_nuclei", "cells_nodebris", "nuclei_nodebris", "filtered_cells"]]
else:
    image_metadata = image_metadata[["run_name", "image", "row", "column", "FOV", "detected_cells", "detected_nuclei",  "filtered_cells"]]

# save
image_metadata.to_csv(output_path, index=None)
