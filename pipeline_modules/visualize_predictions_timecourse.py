#!/usr/bin/env python3

import pandas as pd
import numpy as np

import argparse
from matplotlib import pyplot as plt
import matplotlib.patheffects as patheffects
from tqdm import tqdm

from imageio import imread
from skimage.exposure import adjust_gamma
from utils.image_handling import background_subtract, quantile_normalize
from utils.misc_utils import get_color_codes_pred_probas, move_columns_to_front


# slice a part of the stitched image
def slice_stitch(img, size):
    for _ in range(img.shape[1]//size):
        part = img[:,:size]
        img = img[:, size:]
        yield part
        
# extract single (preprocessed) images from the stitch 
def split_stitch(stitch): yield from slice_stitch(stitch, 1080)

# reads all timepoints images together, stitches the timepoints, preprocesses the stitched image and then splits it again
def joint_normalize_FOV(paths_channels, background_intensities, minq, maxq, gamma):
    stitch_channels = []
    for k, channel in enumerate(paths_channels):
        to_stitch = [imread(path) for path in paths_channels[channel]]

        # standard image preprocessing, but for the stitched image
        stitch = np.concatenate(np.array(to_stitch),axis=1)
        stitch = background_subtract(stitch, background_intensities[k])
        stitch = quantile_normalize(stitch, minq, maxq)
        stitch = adjust_gamma(stitch, gamma=gamma)
        stitch_channels.append(stitch)
    color_stitch = np.stack(stitch_channels,axis=-1)[...,[1,0,2]]
    return list(split_stitch(color_stitch))

# load command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', "--imagelist", required=True)
parser.add_argument('-p', "--predictions", required=True)
parser.add_argument('-o', "--output_save_folder", required=True)
parser.add_argument('-b', "--background", nargs=5, type=int, default=[0,0,0,0,0]) # estimated background intensities of the 5 channels
parser.add_argument('-n', "--minq", type=float, default=0.05)
parser.add_argument('-x', "--maxq", type=float, default=0.9975)
parser.add_argument('-g', "--gamma", type=float, default=0.43)
parser.add_argument('-l', "--labels", type=int, default=1)
parser.add_argument('-s', "--separate_channels", type=int, default=1)
parser.add_argument('-CP', "--color_code_probas", type=int, default=1)
parser.add_argument('-ST', "--size_tile_overlay", type=int, default=10)

args = parser.parse_args()

imagelist_path = args.imagelist
predictions_path = args.predictions
background_intensities = args.background
save_folder = args.output_save_folder
minq  = args.minq
maxq  = args.maxq
gamma = args.gamma
show_labels = args.labels
generate_separate_channels = args.separate_channels
color_code_probas = args.color_code_probas
size = args.size_tile_overlay

size_basic = 9

predictions = pd.read_csv(predictions_path)
imagelist = pd.read_csv(imagelist_path)

save_folder_color = f"{save_folder}/color_overlay"
save_folder_split = f"{save_folder}/channels_split"

if color_code_probas:
    predictions = get_color_codes_pred_probas(predictions)
    
timepoints = sorted(list(set(predictions.measurement.unique()).union(imagelist.measurement.unique())))
n_timepoints = len(timepoints)

FOVs = imagelist["FOV"].unique()

for FOV in tqdm(FOVs):
    imagelist_FOV = imagelist.query(f'FOV == "{FOV}"').replace("file:", "", regex=True)
    predictions_FOV = predictions.query(f'image == "{FOV}"')
    paths_channels = imagelist_FOV[["URL_GFP", "URL_mScarlet", "URL_BFP"]]
    tagged_timepoints_normalized = joint_normalize_FOV(paths_channels, background_intensities, minq, maxq, gamma)
    
    # rare cases when there was a problem with the FOV in one of the measurements
    if len(tagged_timepoints_normalized) != n_timepoints:
        timepoints_FOV = sorted(list(imagelist_FOV.measurement.unique()))
        print(f"Warning: image {FOV} only has {len(timepoints_FOV)}/{n_timepoints} measurements available: {timepoints_FOV}")
        continue
    
    # COLOR OVERLAY
    f, axarr = plt.subplots(1, n_timepoints, figsize=(size_basic * n_timepoints, size_basic))
    # plot each timepoint in a grid
    for j in range(n_timepoints):
        axarr[j].imshow(tagged_timepoints_normalized[j])
        axarr[j].axis('off')    
    plt.subplots_adjust(wspace=0.002, hspace=0.002)
    # add text
    if show_labels:
        for timepoint in timepoints:
            predictions_timepoint = predictions_FOV[predictions_FOV.measurement == timepoint]
            j = timepoint-1
            for cell in predictions_timepoint.index:
                coords = predictions_timepoint.loc[cell, ["Location_Center_X_nucleus", "Location_Center_Y_nucleus"]]
                label = predictions_timepoint.loc[cell, "clone_predicted"]
                if color_code_probas:
                    color = predictions_timepoint.loc[cell, "color_RGB"]
                else:
                    color = "black"
                txt = axarr[j].text(coords[0], coords[1], label, fontsize=size_basic-1,ha='center', va='center', weight="bold", color=color)
                txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])   
    plt.show()
    plt.savefig(f"{save_folder_color}/{FOV}_timecourse_color.jpg", bbox_inches="tight", pil_kwargs={"quality": 92})
    #plt.savefig(f"{save_folder_color}/{FOV}_timecourse_color.png", bbox_inches="tight")
    plt.close()
    
    # EACH CHANNEL SEPARATELY
    if generate_separate_channels:
        f, axarr = plt.subplots(4, n_timepoints, figsize=(size * n_timepoints, size * 4))
        # plot each timepoint in a grid
        # 0 = color overlay, 1 = GFP (G), 2 = mScarlet (R), 3 = BFP (B)
        timepoints_channels = [tagged_timepoints_normalized] + [[tagged_timepoints_normalized[j][...,i] for j in range(n_timepoints)] for i in [1,0,2]]
        for i in range(4):          
            for j in range(n_timepoints):
                axarr[i,j].imshow(timepoints_channels[i][j], cmap="Greys_r", aspect="auto")
                axarr[i,j].axis('off')
        plt.subplots_adjust(wspace=0.002, hspace=0.002)
        # add text
        if show_labels:
            for timepoint in timepoints:
                predictions_timepoint = predictions_FOV[predictions_FOV.measurement == timepoint]
                j = timepoint-1
                for cell in predictions_timepoint.index:
                    coords = predictions_timepoint.loc[cell, ["Location_Center_X_nucleus", "Location_Center_Y_nucleus"]]
                    label = predictions_timepoint.loc[cell, "clone_predicted"]
                    if color_code_probas:
                        color = predictions_timepoint.loc[cell, "color_RGB"]
                    else:
                        color = "black"
                    for i in range(4):
                        txt = axarr[i,j].text(coords[0], coords[1], label, fontsize=size-1,ha='center', va='center', weight="bold", color=color)
                        txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])
        plt.show()
        plt.savefig(f"{save_folder_split}/{FOV}_timecourse_split.jpg", bbox_inches="tight", pil_kwargs={"quality": 92})
        #plt.savefig(f"{save_folder_split}/{FOV}_timecourse_split.png", bbox_inches="tight")
        plt.close()
