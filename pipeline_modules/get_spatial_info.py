#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from utils.misc_utils import get_well_ID
from utils.spatial_info_handling import get_distance_scores
from tqdm import tqdm

# load command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-c', "--cells_T1_path", required=True) # path to (CellProfiler features) cell positions at post-perturbation timepoint
parser.add_argument('-p', "--predictions_T0_path", required=True) # path to cell positions and predicted labels at before perturbation timepoint
parser.add_argument('-o', "--output_path", required=True)
parser.add_argument('-r', "--radius", type=int, required=True)

args = parser.parse_args()

cells_T1_path = args.cells_T1_path
predictions_T0_path = args.predictions_T0_path
output_path = args.output_path
radius = args.radius

metadata_cols = ['image', 'row', 'column', 'FOV', 'ObjectNumber_cell',
                 'Location_Center_X_nucleus', 'Location_Center_Y_nucleus',
                 'posX_cell_abs', 'posY_cell_abs', 'posX_FOV_abs', 'posY_FOV_abs']

predictions_T0 = pd.read_csv(predictions_T0_path)
cells_T1 = pd.read_csv(cells_T1_path, usecols=metadata_cols)

# extract the list of classes
cols = predictions_T0.columns.to_list()
labels_start_idx = cols.index('posY_FOV_abs')+1
labels_end_idx = cols.index('clone_predicted')
labels_classes = cols[labels_start_idx:labels_end_idx]

# prepare empty fields to later save the probabilities in
cells_T1.loc[:,labels_classes] = 0

predictions_T0["well"] = get_well_ID(predictions_T0)
cells_T1["well"] = get_well_ID(cells_T1)

wells_T0 = sorted(predictions_T0.well.unique())
wells_T1 = sorted(cells_T1.well.unique())
try:
    assert wells_T0==wells_T1
    wells = wells_T1
except AssertionError:
    print(f'WARNING: wells in timepoint0 (n={len(wells_T0)}) do not match timepoint1 (n={len(wells_T1)})')
    for w in [w for w in wells_T0 if w not in wells_T1]:
        print(f"  Well {w} is present at timepoint0 but not at timepoint1. Discarding.")
    for w in [w for w in wells_T1 if w not in wells_T0]:
        print(f"  Well {w} is present at timepoint1 but not at timepoint0. Discarding.")
    wells = sorted(list(set(wells_T0).intersection(set(wells_T1))))

# get the spatial information - this needs to be done separately for each well
for well in tqdm(wells[:]):
    # cells within the given well - timepoint 0
    cells_T0_well = predictions_T0[predictions_T0["well"] == well][['posX_cell_abs', 'posY_cell_abs', 'clone_predicted']]
    cells_T0_well = cells_T0_well.reset_index().rename(columns={"index":"index_old"})
    # cells within the well - timepoint 1
    cells_T1_well = cells_T1[cells_T1["well"] == well][['posX_cell_abs', 'posY_cell_abs']]
    
    # for each T1 cell get the T0 cells within [radius] and the distances
    neigh = NearestNeighbors(n_neighbors=2, radius=radius)
    neigh.fit(cells_T0_well[["posX_cell_abs", "posY_cell_abs"]])
    neighbor_distances_well, neighbor_idxs_well = neigh.radius_neighbors(cells_T1_well, return_distance=True)
    # convert positional indices of the neighbor cells to predicted clones
    neighbors_clones_well = [np.array([cells_T0_well.loc[cell_idx, "clone_predicted"] for cell_idx in nearest_neighbors_cell]) for nearest_neighbors_cell in neighbor_idxs_well]
    # calculate distance scores for each combination (T1 cell, clone with cells in its vicinity)
    # distance score - 1 number per clone, [0,1], score increases more cells of the same clone in vicinity + the closer the cell the higher
    cells_distance_scores = [get_distance_scores(clones, distances) for clones, distances in list(zip(neighbors_clones_well, neighbor_distances_well))]
    
    # prepare to copy
    cells_T1_well.loc[:,labels_classes] = 0
    # fill in the non-zeros where applicable
    for cell_idx, cell_distance_scores in zip(cells_T1_well.index, cells_distance_scores):
        cells_T1_well.loc[cell_idx, cell_distance_scores[0]] = cell_distance_scores[1]
    # copy back to the table with all wells [faster than doing it directly]
    cells_T1.loc[cells_T1_well.index, labels_classes] = cells_T1_well.loc[cells_T1_well.index, labels_classes]

# limit the class predictions precision so that the resulting CSV file doesn't explode and save
cells_T1[labels_classes] = cells_T1[labels_classes].round(decimals=8)
cells_T1.drop(columns="well").to_csv(output_path, index=None)
