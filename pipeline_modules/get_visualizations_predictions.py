#!/usr/bin/env python3

import argparse
from skimage.io import imread
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from utils.image_handling import *
from utils.mask_handling import filter_objects_size, renumber_mask
from utils.misc_utils import get_color_codes_pred_probas
from tqdm import tqdm

# generate an image, the same size as the tagged channel image (1080x1080 by default)
def visualize_labeled_tagged_hires(tagged, predictions_img, color_code_probas=False, output_path=None, quality=90):
    predictions_img = predictions_img.copy()
    if color_code_probas:
        predictions_img = get_color_codes_pred_probas(predictions_img)
    
    dpi = 80
    height, width, nbands = tagged.shape
    figsize = width / float(dpi), height / float(dpi)  # what size does the figure need to be in inches to fit the image?
    fig = plt.figure(figsize=figsize) # create a figure of the right size with one axes that takes up the full figure
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    ax.imshow(tagged, interpolation=None)
    # add text labels
    for cell in predictions_img.index:
        coords = predictions_img.loc[cell, ["Location_Center_X_nucleus", "Location_Center_Y_nucleus"]]
        label = predictions_img.loc[cell, "clone_predicted"]
        if color_code_probas:
            color = predictions_img.loc[cell, "color_RGB"]
        else:
            color = "black"
        txt = ax.text(coords[0], coords[1], label, fontsize=25,ha='center', va='center', weight='bold', color=color)
        txt.set_path_effects([patheffects.withStroke(linewidth=3.5, foreground='white')])

    # Ensure we're displaying with square pixels and the right extent.
    ax.set(xlim=[-0.5, width - 0.5], ylim=[height - 0.5, -0.5], aspect=1)

    if not output_path:
        plt.show()
    else:
        fig.savefig(output_path, dpi=dpi, pil_kwargs={'quality': quality})
        plt.close("all")

# show side by side 3 images: structural - tagged with text labels - color predictions
def visualize_predictions_comparison(structural, tagged, color_predictions, predictions_img, color_code_probas=False, ncol=3, size=8, output_path=None):
    predictions_img = predictions_img.copy()
    if color_code_probas:
        predictions_img = get_color_codes_pred_probas(predictions_img)
    nrow = 1
    f, axarr = plt.subplots(nrow, ncol, figsize=(size * ncol, size * nrow))
    axarr[0].imshow(structural)
    
    axarr[1].imshow(tagged)
    # add text labels
    for cell in predictions_img.index:
        coords = predictions_img.loc[cell, ["Location_Center_X_nucleus", "Location_Center_Y_nucleus"]]
        label = predictions_img.loc[cell, "clone_predicted"]
        if color_code_probas:
            color = predictions_img.loc[cell, "color_RGB"]
        else:
            color = "black"
        txt = axarr[1].text(coords[0], coords[1], label, fontsize=8,ha='center', va='center', weight="bold", color=color)
        txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])

    axarr[2].imshow(color_predictions)
    
    if not output_path:
        plt.show()
    else:
        f.savefig(output_path, bbox_inches="tight", pil_kwargs={"quality": 92})
        plt.close("all")

def get_label_encoding(rf_model_path, predictions):
    # get a list of all existing labels in the dataset
    if rf_model_path:
        rf = joblib.load(rf_model_path)
        labels = rf.classes_
    else:
        labels = sorted(list(predictions.clone_predicted.unique()))

    # assign each label a number
    label_encoder_dict = {label:i+1 for i,label in enumerate(labels)}
    # reverse mapping
    label_decoder_dict = {i+1:label for i,label in enumerate(labels)}

    return label_encoder_dict, label_decoder_dict

# generate color scheme for "colored segmentation" visualization
def get_updated_colormap(colormap_path):
    # each label (here, encoded as a number) corresponds to one color
    colormap = np.load(colormap_path)
    # plus there are 3 special codes/colors: to show segmentation and 1:1 mapping / filtering
    maxa = min(65535, len(colormap)-1)
    white = maxa-1
    grey = maxa-2
    red = maxa-3
    # set the corresponding colormap values to generate the desired colors
    colormap[white] = [0.99,0.99,0.99,1]
    colormap[grey] = [0.71, 0.71, 0.71, 1]
    colormap[red] = [0.804, 0.0, 0.102, 1.0]
    
    return colormap
       
# generates an image to show predictions in color + segmentation
def get_color_predictions(cell_labels, filtered_cells, detected_cells, detected_nuclei, colormap, nuclei_min_size=0, cells_min_size=0):
    
    nuclei_nodebris = filter_objects_size(detected_nuclei, nuclei_min_size, renumber=True)
    cells_nodebris = filter_objects_size(detected_cells, cells_min_size, renumber=True)
    
    # define values for segmentation visualization
    maxa = min(65535, len(colormap)-1)
    white = maxa-1
    grey = maxa-2
    red = maxa-3
    
    newmask = np.zeros(filtered_cells.shape, dtype="uint16")
    # plot all detected cells in grey
    newmask[detected_cells > 0] = grey
    # take segmentation mask, change numbers from cell_number to predicted_class -> color same predicted clones with the same color
    for cell in np.unique(filtered_cells)[1:]:
        newmask[filtered_cells == cell] = cell_labels[cell-1]
    # make boundaries between cells black
    newmask[find_boundaries(detected_cells)] = 0
    # cells that were discarded because they are too small get red boundaries
    to_color_red = detected_cells.copy()
    to_color_red[cells_nodebris!=0]=0
    newmask[find_boundaries(to_color_red)] = red
    # make all regions corresponding to detected nuclei black and nuclei border white, again nuclei too small -> red
    newmask[detected_nuclei>0] = 0
    newmask[find_boundaries(detected_nuclei)] = white
    to_color_red = (detected_nuclei>0) & (nuclei_nodebris==0)
    newmask[find_boundaries(to_color_red)] = red
    
    return colormap[newmask]

# load command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', "--imagelist", required=True)
parser.add_argument('-p', "--predictions", required=True)
parser.add_argument('-C', "--output_comparison", required=True)
parser.add_argument('-t', "--output_tagged", required=True)
parser.add_argument('-c', "--colormap", required=True)
parser.add_argument('-b', "--background", nargs=5, type=int, default=[0,0,0,0,0]) # estimated background intensities of the 5 channels
parser.add_argument('-Nm', "--nuclei_min_size", type=int, default = 0)
parser.add_argument('-Cm', "--cells_min_size", type=int, default = 0)
parser.add_argument('-n', "--minq", type=float, default=0.05)
parser.add_argument('-x', "--maxq", type=float, default=0.9975)
parser.add_argument('-g', "--gamma", type=float, default=0.43)
parser.add_argument('-r', "--rf", default=None)
parser.add_argument('-CP', "--color_code_probas", action='store_true')


args = parser.parse_args()

imagelist_path = args.imagelist
predictions_path = args.predictions
output_folder_comparisons = args.output_comparison
output_folder_labeled_tagged = args.output_tagged
colormap_path = args.colormap
background_intensities = args.background
nuclei_min_size = args.nuclei_min_size
cells_min_size = args.cells_min_size
minq  = args.minq
maxq  = args.maxq
gamma = args.gamma
rf_model_path = args.rf
color_code_probas = args.color_code_probas

# prepare data
predictions = pd.read_csv(predictions_path)
imagelist = pd.read_csv(imagelist_path)

colormap = get_updated_colormap(colormap_path)
label_encoder_dict, label_decoder_dict = get_label_encoding(rf_model_path, predictions)

images = imagelist[["FOV", "Group_Index"]].copy()
images.rename(columns={"FOV":"image"}, inplace=True)
images["filtered_cells"] = [x[5:] for x in imagelist["URL_filtered_cells"]]
images["filtered_nuclei"] = [x[5:] for x in imagelist["URL_filtered_nuclei"]]
images["detected_cells"] = [x[5:] for x in imagelist["URL_segmented_cells_mask"]]
images["detected_nuclei"] = [x[5:] for x in imagelist["URL_segmented_nuclei_mask"]]

# extract paths to the TIFF file for each channel (to generate structural/tagged overlays with gamma correction)
for j in [1,2,3,4,5]:
    images[f"ch{j}"] = imagelist["URL_GFP"].str.replace("file:","", regex=False)
    images[f"ch{j}"] = images[f"ch{j}"].str.replace(r'-ch1sk1', f'-ch{j}sk1', regex=False)

for i in tqdm(images.index):
    detected_cells_mask = imread(images.loc[i, "detected_cells"])
    detected_nuclei_mask = imread(images.loc[i, "detected_nuclei"])
    filtered_cells_mask = imread(images.loc[i, "filtered_cells"])
    filtered_nuclei_mask = imread(images.loc[i, "filtered_nuclei"])

    # extract predictions the given FOV
    predictions_FOV = predictions[predictions["image"]==images.loc[i, "image"]].sort_values("ObjectNumber_cell")
    # extract the predicted labels for each cell in FOV
    cell_labels = [label_encoder_dict[x] for x in predictions_FOV["clone_predicted"]]
    
    color_predictions = get_color_predictions(cell_labels, filtered_cells_mask, detected_cells_mask, detected_nuclei_mask, colormap, nuclei_min_size, cells_min_size)
    #plt.figure(figsize=(7,7)); plt.imshow(color_predictions); plt.show()
    
    # load and preprocess TIFF images (one per channel)
    channels = load_image_channels(images.loc[i,[f"ch{j}" for j in [1,2,3,4,5]]], background_intensities, minq, maxq, gamma)

    # create color overlays for tagged and structural channels with gamma correction
    structural = get_struct_chans(channels)
    tagged = get_tagged_chans(channels)

    visualize_predictions_comparison(structural, tagged, color_predictions, predictions_FOV, color_code_probas, output_path = f'{output_folder_comparisons}/{images.loc[i, "image"]}_predictions_comparison.jpg')
    visualize_labeled_tagged_hires(tagged, predictions_FOV, color_code_probas, output_path = f'{output_folder_labeled_tagged}/{images.loc[i, "image"]}_predictions_labeled_tagged.jpg')
