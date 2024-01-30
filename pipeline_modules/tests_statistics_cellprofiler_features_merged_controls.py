#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import scipy.stats

from tqdm import tqdm
from utils.misc_utils import get_well_ID, get_well_from_FOVID

def separate_frames_metadata(df, frames=["GFP", "mScarlet"]):
    collector = []
    for f in frames:
        df_f = df[["clone", f"assigned_gene_{f}", f"top1_guide_{f}"]].rename(columns={f"assigned_gene_{f}":"protein", f"top1_guide_{f}":"sgRNA"})
        df_f["frame"] = f
        collector.append(df_f)
    return pd.concat(collector)[["clone", "frame", "protein", "sgRNA"]]

# load command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--measurement_name", required=True)
parser.add_argument('-f', "--features_list_path", required=True)
parser.add_argument('-pm', "--platemap_path", required=True)
parser.add_argument('-cm', "--clones_metadata_path", required=True)
parser.add_argument('-cp', "--cellprofiler_path", required=True)
parser.add_argument('-i', "--imagelist_path", required=True)
parser.add_argument('-pf', "--predictions_final_path", required=True)
parser.add_argument('-pa', "--predictions_all_path", required=True)
parser.add_argument('-o', "--test_scores_outfile", required=True)
parser.add_argument('-D', "--cp_DMSO_wells_path", required=True)

args = parser.parse_args()

measurement_name = args.measurement_name
features_list_path = args.features_list_path
platemap_path = args.platemap_path
clones_metadata_path = args.clones_metadata_path
cellprofiler_path = args.cellprofiler_path
imagelist_path = args.imagelist_path
predictions_final_path = args.predictions_final_path
predictions_all_path = args.predictions_all_path
test_scores_outfile = args.test_scores_outfile
cp_DMSO_wells_path = args.cp_DMSO_wells_path

# load and prepare data
features = pd.read_csv(features_list_path)
platemap = pd.read_csv(platemap_path)
clones_metadata = pd.read_csv(clones_metadata_path)
imagelist = pd.read_csv(imagelist_path)

predictions_all = pd.read_csv(predictions_all_path)
predictions_final = pd.read_csv(predictions_final_path)

cellprofiler_DMSO = pd.read_csv(cp_DMSO_wells_path)

# load only the speified subset of CellProfiler features we use for hit calling
metadata_features = ['image', 'row', 'column', 'FOV', 'ImageNumber_cell', 'ObjectNumber_cell']
features_to_test = features.GFP.to_list() + features.mScarlet.to_list()
cellprofiler = pd.read_csv(cellprofiler_path, usecols = metadata_features + features_to_test)

# separate GPF/mScarlet frames for clone metadata
clones_metadata = clones_metadata[["clone", "clone_number", "assigned_gene_GFP", "assigned_gene_mScarlet", "top1_guide_GFP", "top1_guide_mScarlet"]]
clones_metadata_melted = separate_frames_metadata(clones_metadata)

platemap.rename(columns={"ImagingPlateWell": "well"}, inplace=True)
plate_name = measurement_name.split("__")[0]

# add well IDs - each well corresponds to a different perturbation
imagelist["well"] = get_well_from_FOVID(imagelist.FOV)
predictions_all["well"] = get_well_ID(predictions_all)
predictions_final["well"] = get_well_ID(predictions_final)
cellprofiler["well"] = get_well_ID(cellprofiler)

# restrict platemap to only the given plate and relevant wells
platemap = platemap[(platemap.ImagingPlateName==plate_name) & (platemap.well.isin(imagelist.well.unique()))].reset_index(drop=True)

# collector variable
result = []

# test each clone, one after another
for clone_idx in tqdm(clones_metadata.index[:]):
    clone = clones_metadata.loc[clone_idx, "clone"]
    # use combined model predictions for treated wells
    predictions_final_clone = predictions_final.query(f'clone_predicted == "{clone}"')
    # load cellprofiler measurements for cells of the given clone
    cellprofiler_clone = cellprofiler.merge(predictions_final_clone[["image", "ObjectNumber_cell"]])
    # controls from the merged cellprofiler_DMSO file
    cp_controls = cellprofiler_DMSO[cellprofiler_DMSO.clone_predicted == clone][features_to_test]
    
    # empty matrix to store the calculated p-vals: rows are different compounds (=wells), columns are cellprofiler features; initialize with 1 (= not significant)
    pvals_matrix = pd.DataFrame(np.full((platemap.shape[0],len(features_to_test)), 1, dtype="float"), columns=features_to_test, index=platemap["well"]) 
    # mean matrix, stddev matrix - again same shape of the matrix
    mean_matrix = pd.DataFrame(np.full((platemap.shape[0],len(features_to_test)), np.nan, dtype="float"), columns=features_to_test, index=platemap["well"]) 
    stddev_matrix = mean_matrix.copy()
    
    # for each clone + compound + cellprofiler feature calculate the values and store them in the matrix
    for well in platemap.well[:]:
        cp_well = cellprofiler_clone[cellprofiler_clone.well == well][features_to_test]
        pvals = scipy.stats.ttest_ind(cp_well, cp_controls, equal_var=False)[1]
        means = np.array(cp_well.mean()).reshape((1,len(features_to_test)))
        stds = np.array(cp_well.std()).reshape((1,len(features_to_test)))
        pvals_matrix.loc[well] = pvals
        mean_matrix.loc[well] = means
        stddev_matrix.loc[well] = stds
        
    # calculate statistics for controls - these are the same wells for the entire plate
    stats_controls = cp_controls.mean().rename("mean_control").reset_index().merge(cp_controls.std().rename("std_control").reset_index(), on="index")
    stats_controls = stats_controls.rename(columns={"index":"variable"})
    
    # get number of detected cells in each well in the given plate
    pd.Categorical(values = cellprofiler_clone.well, categories = sorted(platemap.well.unique()), ordered = True).value_counts()
    n_cells_wells = pd.DataFrame(pd.Categorical(values = cellprofiler_clone.well, categories = sorted(platemap.well.unique()), ordered = True).value_counts().rename("n_cells_treated")).reset_index().rename(columns={"index": "well"})

    # reshape the calculated statistics
    pvals_matrix_melted = pd.melt(pvals_matrix.reset_index(), id_vars="well", value_vars=pvals_matrix.columns, value_name = "pval")
    mean_matrix_melted = pd.melt(mean_matrix.reset_index(), id_vars="well", value_vars=mean_matrix.columns, value_name = "mean_treated")
    std_matrix_melted = pd.melt(stddev_matrix.reset_index(), id_vars="well", value_vars=stddev_matrix.columns, value_name = "std_treated")

    # merge into one single table
    result_clone = mean_matrix_melted.merge(std_matrix_melted, on=["well", "variable"]).merge(pvals_matrix_melted, on=["well", "variable"])

    # add info about the compound in the respective well and number of cells
    result_clone = result_clone.merge(platemap[["CompoundName", "AssayConc_uM", "well"]], on="well")
    result_clone = result_clone.merge(n_cells_wells, on="well", how="left")
    # add info about controls
    result_clone = result_clone.merge(stats_controls, on="variable")
    result_clone["n_cells_control"] = cp_controls.shape[0]
    # calculate fold change
    result_clone["fold_change"] = result_clone["mean_treated"]/result_clone["mean_control"]
    result_clone["clone"] = clone
    
    result.append(result_clone)
    
result = pd.concat(result)

# add information about the tagged protein
result["frame"] = ""
result.loc[result.variable.str.contains("mScarlet"), "frame"] = "mScarlet"
result.loc[result.variable.str.contains("GFP"), "frame"] = "GFP"
result = result.merge(clones_metadata_melted)
result.variable = result.variable.str.replace("GFP_|mScarlet_", "", regex=True)
result.CompoundName = result.CompoundName.fillna("DMSO")

columns_reorder = ["protein", "sgRNA", "CompoundName", "variable", "pval", "fold_change", "mean_treated", "std_treated", "mean_control", "std_control", "n_cells_treated", "n_cells_control", "clone", "AssayConc_uM", "frame", "well"]
result = result[columns_reorder]
result.sort_values(["protein", "sgRNA", "CompoundName", "variable"], inplace=True)

result.to_csv(test_scores_outfile, index=None)
