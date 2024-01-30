#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd

def extract_predictions(df, classes, remove_zeroprob=True):
    Y_pred = np.array(df[classes])
    df["clone_predicted"] = np.array(classes)[np.argmax(Y_pred, axis=1)]
    df["clone_predicted_prob"] = np.max(Y_pred, axis=1)
    
    # cells which have p=0 for all clones - remove the prediction (first class by default = wrong)
    if remove_zeroprob:
        df.loc[df["clone_predicted_prob"]==0, "clone_predicted"] = np.nan

# load command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-s', "--predictions_spatial_path", required=True)
parser.add_argument('-p', "--predictions_RF_multiple_substring", required=True) # folder and beginning of filename which all filenames contain
parser.add_argument('-o', "--output_path", required=True)

args = parser.parse_args()

predictions_spatial_path = args.predictions_spatial_path
predictions_RF_multiple_substring = args.predictions_RF_multiple_substring
output_path = args.output_path

print(predictions_spatial_path)
print(predictions_RF_multiple_substring)

# extract the (variable number of) filepaths with predictions from different models to sum
folder = "/".join(predictions_RF_multiple_substring.split("/")[:-1])
filename_pattern = predictions_RF_multiple_substring.split("/")[-1]
predictions_settings_paths = sorted([folder+"/"+x for x in os.listdir(folder) if filename_pattern in x])

# get the sum of probabilities for the different settings
predictions_modelsum = pd.read_csv(predictions_settings_paths[0])

# extract the list of class labels (clones)
cols = predictions_modelsum.columns.to_list()
labels_start_idx = cols.index('posY_FOV_abs')+1
labels_end_idx = cols.index('clone_predicted')
labels_classes = cols[labels_start_idx:labels_end_idx]

# get the sum of all settings
for i in range(1, len(predictions_settings_paths)):
    predictions_modelsum[labels_classes] += pd.read_csv(predictions_settings_paths[i])[labels_classes]

# load the spatial predictions
predictions_spatial = pd.read_csv(predictions_spatial_path)

# prepare the table to store the final predictions = (predictions_modelsum/n_settings)*predictions_spatial
predictions_final = predictions_modelsum.copy()

# update the probabilities saved as neighborhood probability matrix * sum of probability from different model subsets matrix
predictions_final[labels_classes] = predictions_modelsum[labels_classes] / 4 * predictions_spatial[labels_classes]

# get the predictions
extract_predictions(predictions_final, labels_classes)

# save as csv, limit floats to 9 decimals
predictions_final[labels_classes] = predictions_final[labels_classes].round(decimals=9)
predictions_final["clone_predicted_prob"] = predictions_final["clone_predicted_prob"].round(decimals=9)
predictions_final.to_csv(output_path, index=None)
