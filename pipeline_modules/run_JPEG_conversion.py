#!/usr/bin/env python3
## VERSION 4.0: added gamma correction + now uses functions file

import os
import argparse
from tqdm import tqdm
import numpy as np

from utils.image_handling import load_image_channels, get_struct_chans, get_tagged_chans, save_img

# load command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--measurement_path", required=True)
parser.add_argument('-b', "--background", nargs=5, type=int, default=[0,0,0,0,0]) # estimated background intensities of the 5 channels
parser.add_argument('-n', "--minq", type=float, default=0.05)
parser.add_argument('-x', "--maxq", type=float, default=0.9975)
parser.add_argument('-g', "--gamma", type=float, default=None)
parser.add_argument('-r', "--resolution", type=int, default=1080)    
parser.add_argument('-q', "--quality", type=int, default=92)
parser.add_argument('-sp', "--separate", action='store_true')
parser.add_argument('-s', "--suffix", default="-ch1sk1fk1fl1.tiff") # suffix of channel 1

args = parser.parse_args()
    
input_folder = os.path.join(args.measurement_path, "Images_FFC/")
output_folder_tagged = os.path.join(args.measurement_path, "Images_tagged_chans_JPEG/")
output_folder_structural = os.path.join(args.measurement_path, "Images_structural_chans_JPEG/")
output_folder_separate = os.path.join(args.measurement_path, "Images_separate_chans_JPEG/")
save_separate = args.separate
channels_background = args.background

print(channels_background)

suffix = args.suffix
JPEG_resize_px = args.resolution
JPEG_quality = args.quality
minq = args.minq
maxq = args.maxq
gamma = args.gamma

suffix_length = len(suffix)
#
# create output folders, if necessary
for folder in [output_folder_tagged, output_folder_structural, output_folder_separate]:
    try: os.mkdir(folder)
    except: pass

# get list of FOV IDs to work with
FOVs = list(sorted(set([x[:-suffix_length] for x in os.listdir(input_folder) if x.startswith("r")])))

print(input_folder)
print(len(FOVs), "FOVs")

# load individual channels
for FOV in tqdm(FOVs[:]):
    channels_paths = [input_folder + FOV + suffix.replace("ch1", f"ch{ch}") for ch in range(1,6)]
    channels = load_image_channels(channels_paths, channels_background, minq, maxq, gamma)

    tagged = get_tagged_chans(channels)
    structural = get_struct_chans(channels)

    save_img(output_folder_tagged + FOV + "_tagged_channels.jpg", tagged, (JPEG_resize_px,JPEG_resize_px), JPEG_quality)
    save_img(output_folder_structural + FOV + "_structural_channels.jpg", structural, (JPEG_resize_px, JPEG_resize_px), JPEG_quality)

    if save_separate:
        for ch in range(1,6):
            save_img(output_folder_separate + FOV + f"_ch{ch}.jpg", channels[...,ch-1], (JPEG_resize_px,JPEG_resize_px), JPEG_quality)
