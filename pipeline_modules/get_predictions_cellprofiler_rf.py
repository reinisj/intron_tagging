#!/usr/bin/env python3

import os
import argparse

import pandas as pd
import numpy as np

import joblib
import json

from utils.misc_utils import get_well_ID
import subprocess

from tqdm import trange

def get_RF_predictions(cellprofiler_measurements, rf):
    # get the predictions
    X = cellprofiler_measurements[rf.feature_names_in_]
    Y_pred = rf.predict_proba(X)
    # get top class for each cell
    clone_predicted = rf.classes_[np.argmax(Y_pred, axis=1)]
    # save also the probability of this class
    clone_predicted_proba = np.max(Y_pred, axis=1)

    # save in dataframe
    Y_pred = pd.DataFrame(rf.predict_proba(X), columns=rf.classes_)
    Y_pred["clone_predicted"] = clone_predicted
    Y_pred["clone_predicted_prob"] = clone_predicted_proba

    # save predictions, report if the file already exists
    result = cellprofiler_measurements[["image", "row", "column", "FOV", "ObjectNumber_cell", "Location_Center_X_nucleus", "Location_Center_Y_nucleus", "posX_cell_abs", "posY_cell_abs", "posX_FOV_abs", "posY_FOV_abs"]].copy()
    result = result.merge(Y_pred, left_index=True, right_index=True)
    
    return result

# load command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', "--cellprofiler_measurement", required=True)
parser.add_argument('-m', "--rf_model", required=True)
parser.add_argument('-o', "--save_path", required=True)
parser.add_argument('-x', "--max_n_read", default=50000)

args = parser.parse_args()

cellprofiler_measurements_path = args.cellprofiler_measurement
rf_model_path = args.rf_model
predictions_save_path = args.save_path
nlines_split = args.max_n_read

# get how many files there are in the file
cmd = ['wc', '-l', cellprofiler_measurements_path]
output = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0].decode("utf-8")
nlines_total = int(output.split(" ")[0])
nlines_total

print("\n" + cellprofiler_measurements_path.split("/")[-1])
print(cellprofiler_measurements_path)
print(rf_model_path)
print(nlines_total, "cells")
print()

# load the trained random forest model
rf = joblib.load(rf_model_path)
# don't print technical details when running
rf.verbose = 0

# get the predictions step by step
for i in trange(0, nlines_total, nlines_split):
    if i == 0:
        measurements_batch = pd.read_csv(cellprofiler_measurements_path, skiprows=i, nrows=nlines_split)
        colnames = measurements_batch.columns
        predictions = get_RF_predictions(measurements_batch, rf)
        predictions[:-1].to_csv(predictions_save_path, index=None)
    else:
        measurements_batch = pd.read_csv(cellprofiler_measurements_path, skiprows=i, nrows=nlines_split, header=None)
        measurements_batch.columns = colnames
        predictions = get_RF_predictions(measurements_batch, rf)
        predictions.to_csv(predictions_save_path, mode='a', index=None, header=False)
