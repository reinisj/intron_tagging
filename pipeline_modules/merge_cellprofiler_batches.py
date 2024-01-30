#!/usr/bin/env python3

import argparse
import os

import pandas as pd
import numpy as np

from tqdm import tqdm

from utils.misc_utils import extract_row_column_FOV
from utils.spatial_info_handling import parse_xml_index

def check_remove_NA_rows(df):
    if sum(df.isna().sum(axis=1) > 0):
        print("cells with NA values:", sum(df.isna().sum(axis=1) > 0))
        print("columns with NA values:")
        print(df.isna().sum()[df.isna().sum() > 0])
        df = df.dropna(axis=0)
    return df

# cell measurements - for each batch, load cell + nuclei + cytoplasm measurements, merge, add metadata 
def process_batch_cell_measurements(batch_folder, imagelist):
    # load measurement files for different objects types (cells, nuclei, cytoplasm) from the same batch
    cells = pd.read_csv(batch_folder + "filtered_cells.csv").add_suffix("_cell")
    nuclei = pd.read_csv(batch_folder + "filtered_nuclei.csv").add_suffix("_nucleus")
    cytoplasm = pd.read_csv(batch_folder + "cytoplasm.csv").add_suffix("_cytoplasm")

    assert len(cells) == len(nuclei) and len(nuclei) == len(cytoplasm)

    # merge (entire) cell + nuclei + cytoplasm_only measurements from the batch
    measurements = cells.merge(
        nuclei, left_on=["ImageNumber_cell", "ObjectNumber_cell"], right_on = ["ImageNumber_nucleus", "ObjectNumber_nucleus"]).merge(
        cytoplasm, left_on=["ImageNumber_cell", "ObjectNumber_cell"], right_on = ["ImageNumber_cytoplasm", "ObjectNumber_cytoplasm"])

    # merge with imagelist to add FOV info (row, column, FOV number) for each cell
    measurements = measurements.merge(imagelist[["FOV", "Group_Index", "posX_FOV_abs", "posY_FOV_abs"]], left_on = "ImageNumber_cell", right_on = "Group_Index")
    measurements.rename(columns={"FOV": "image"}, inplace=True)
    extract_row_column_FOV(measurements)

    # calculate absolute positions of cells from position of the FOV and relative position of the cell within the FOV
    # important: the coordinate system originates at the TOP left corner (i.e. to match convention for image files)
    measurements["posX_cell_abs"] = measurements["Location_Center_X_nucleus"] + measurements["posX_FOV_abs"]
    measurements["posY_cell_abs"] = measurements["Location_Center_Y_nucleus"] + measurements["posY_FOV_abs"]

    # select and reorder columns
    columns = measurements.columns
    # basic metadata: ImageNumber (corresponds to FOV), ObjectNumber (= cell number)
    columns_selected = ["image", "row", "column", "FOV", "ImageNumber_cell", "ObjectNumber_cell", "posX_cell_abs", "posY_cell_abs"]
    # spatial location information (in pixels)
    columns_selected += ["posX_FOV_abs", "posY_FOV_abs", "Location_Center_X_cell", "Location_Center_Y_cell", "Location_Center_X_nucleus", "Location_Center_Y_nucleus"]
    # AreaShape columns - only cells+nuclei, exclude 'AreaShape_Center_*' because it's the same as Location_Center
    columns_selected += [column for column in columns if column.startswith(("AreaShape")) if "cytoplasm" not in column and "Center_" not in column]
    # measure columns - later used for random forest classification
    columns_selected += [column for column in columns if column.startswith(("Correlation", "Intensity", "Granularity", "RadialDistribution", "Texture"))]
    # restrict the data only to selected columns
    measurements = measurements[columns_selected]
    # remove any possible Ncheck_remove_NA_rows values by dropping entire rows - should happen very infrequently
    measurements = check_remove_NA_rows(measurements)
    # sort
    mesurements = measurements.sort_values(["image", "ObjectNumber_cell"])
    return measurements

def process_batch_image_measurements(batch_folder, imagelist):
    image_measurements = pd.read_csv(f"{batch_folder}/Image.csv").rename(columns={"FOV": "image", "PathName_GFP":"Path_Images"})
    extract_row_column_FOV(image_measurements)

    image_measurements = image_measurements.merge(imagelist[["FOV", "posX_FOV_abs", "posY_FOV_abs"]].rename(columns={"FOV": "image"}))

    selected_columns = ["image", "ImageNumber", "posX_FOV_abs", "posY_FOV_abs", "Path_Images", "row", "column", "FOV", "Count_cytoplasm", "Count_filtered_cells", "Count_filtered_nuclei"]
    selected_columns += [col for col in image_measurements.columns if col.startswith((("Correlation", "Intensity", "Granularity", "RadialDistribution", "Texture")))]
    image_measurements = image_measurements[selected_columns]
    return image_measurements

# load arguments from command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('-i', "--images",)
parser.add_argument('-x', "--index",)
parser.add_argument('-b', "--batches",)
parser.add_argument('-o', "--output",)
parser.add_argument('-v', "--harmony_version", default="V6")
parser.add_argument('-n', "--name", default=None)

args = parser.parse_args()
imagelist_path = args.images
index_path = args.index
batches_folder_root = args.batches
merged_output_folder = args.output
opera_harmony_version = args.harmony_version
measurement_name = args.name


# infer measurement name from path, if necessary
if not measurement_name:
    measurement_name = batches_folder_root.rstrip("/").split("/")[-2]

# cell-level measurements output file path
measurements_output_path = merged_output_folder + "/" + measurement_name + "_merged_cell_measurements.csv"
# image-level measurements output file path
image_measurements_output_path = merged_output_folder + "/" + measurement_name + "_image_measurements.csv"


# load imagelist
imagelist = pd.read_csv(imagelist_path)
# get list of batches
batches = sorted([x for x in os.listdir(batches_folder_root) if "batch_" in x])

print(measurement_name)
print(imagelist_path)
print(f"{len(batches)} batches")

# extract positions of FOVs from the index xml file
FOVs_positions, image_size_x, image_size_y = parse_xml_index(index_path, opera_harmony_version=opera_harmony_version)
FOVs_positions = FOVs_positions[["image", "posX_px_zero", "posY_px_zero"]]
FOVs_positions = FOVs_positions.rename(columns={"image":"FOV", "posX_px_zero":"posX_FOV_abs", "posY_px_zero":"posY_FOV_abs"})

# add the positions to imagelist + make sure the IDs match
total_n_FOV = imagelist.shape[0]
imagelist = imagelist.merge(FOVs_positions, on="FOV")
assert imagelist.shape[0] == total_n_FOV

# process the first batch and save
batch_folder = f"{batches_folder_root}/{batches[0]}/"
measurements = process_batch_cell_measurements(batch_folder, imagelist)
measurements.to_csv(measurements_output_path, index=None)
image_measurements = process_batch_image_measurements(batch_folder, imagelist)
image_measurements.to_csv(image_measurements_output_path, index=None)

# add the rest of the batches by appending the generated file
for batch in tqdm(batches[1:]):
    batch_folder = f"{batches_folder_root}/{batch}/"
    measurements = process_batch_cell_measurements(batch_folder, imagelist)
    measurements.to_csv(measurements_output_path, mode='a', header=False, index=None)
    image_measurements = process_batch_image_measurements(batch_folder, imagelist)
    image_measurements.to_csv(image_measurements_output_path, mode='a', header=False, index=None)
