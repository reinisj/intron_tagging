#!/usr/bin/env python3

import argparse
import sys

import pandas as pd
import numpy as np

from imageio import imread,imsave
from skimage.measure import regionprops_table, regionprops
from skimage.segmentation import find_boundaries
from skimage.util import img_as_ubyte

from PIL import Image

from tqdm import tqdm,trange 

from utils.mask_handling import renumber_mask, filter_objects_size, get_mask_selected, count_neighbors

# load arguments from command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('-i', "--imagelist", required=True)    
parser.add_argument('-f', "--first", type=int, default=0)
parser.add_argument('-l', "--last", type=int)
parser.add_argument('-Nm', "--nuclei_min_size", type=int, default = 0)
parser.add_argument('-Cm', "--cells_min_size", type=int, default = 0)
parser.add_argument('-o', "--nuclei_overlap_min", type=float, default=0.66)
parser.add_argument('-L', "--NC_threshold_lower", type=float, default = 0.2)
parser.add_argument('-U', "--NC_threshold_upper", type=float, default = 0.65)
parser.add_argument('-b', "--FOV_border_px", type=int, default=2)
parser.add_argument('-s', "--noborder_strict_mode", default="strict")
parser.add_argument('-e', "--expand_px", type=int, default=5)
parser.add_argument('-c', "--colormap", default="/home/jreinis/cellprofiler_pipelines/2022-03-18-glasbey_light.npy")
parser.add_argument('-O', "--output_folder_inspect", default=None)
parser.add_argument('-Na', "--apopt_thres_nuclei", type=int, default=None)
parser.add_argument('-Ca', "--apopt_thres_cells", type=int, default=None)
parser.add_argument('-S', "--solidity_threshold", type=float, default=None)
parser.add_argument('-E', "--eccentricity_threshold", type=float, default=None)

args = parser.parse_args()

imagelist_path = args.imagelist
first = args.first
last = args.last
nuclei_min_size = args.nuclei_min_size
cells_min_size = args.cells_min_size
nuclei_overlap_min = args.nuclei_overlap_min
NC_threshold_lower = args.NC_threshold_lower
NC_threshold_upper = args.NC_threshold_upper
FOV_border_px = args.FOV_border_px
expand_px = args.expand_px
noborder_strict_mode = args.noborder_strict_mode
colormap_path = args.colormap
output_folder_inspect = args.output_folder_inspect
apopt_thres_nuclei = args.apopt_thres_nuclei
apopt_thres_cells = args.apopt_thres_cells
solidity_threshold = args.solidity_threshold
eccentricity_threshold = args.eccentricity_threshold

def map_nuclei_cells(image_nuclei, image_cells, nuclei_overlap_min=0.66):
    """ 1:1 maps nuclei to cells, returns a list of [nucleus_number, cell_number] pairs that belong together"""
    
    n_cells = np.max(image_cells)
    n_nuclei = np.max(image_nuclei)
        
    # assign all nuclei pixels to segmented cells
    nuclei_labeled_by_cells = (image_nuclei>0)*image_cells
    
    assigned_cells = []
    
    # assign each nucleus to a cell - more than 66% of its volume lies within that cell
    # iterate over nuclei: 1-based indexing because 0 is the background
    for i in range(1,n_nuclei+1):
        # get the pixels corresponding to the nucleus
        nucleus = nuclei_labeled_by_cells[image_nuclei == i]
        # count how many pixels belong to which segmented cells
        cells_assigned_pixels = np.array(np.unique(nucleus, return_counts=True)).T
        #print(i)
        #plt.imshow(image_nuclei == i); plt.show()
        #print(cells_assigned_pixels, "\n")
        # select the cell with the most pixels -> that will be the assigned cell
        top_cell, top_cell_count = cells_assigned_pixels[np.argmax(cells_assigned_pixels[:,1])]  
        # but assign only if more than 2/3 of the nucleus lays within the assigned cell borders, also discard background (top_cell==0)
        if top_cell != 0 and top_cell_count > 0.66*(len(nucleus)):
            assigned_cells.append([i,top_cell])
        
    # create an empty mapping matrix with the dimensions n_cells+1 x n_nuclei+1 (will include empty first column and first line )
    mapping_matrix = np.zeros(shape=(n_cells+1,n_nuclei+1), dtype="int")
    # populate with cells and their assigned nuclei
    for nucleus, cell in assigned_cells:
        mapping_matrix[cell, nucleus] = 1

    # select cells that were assigned to exactly one nucleus
    selected_cells = np.where(mapping_matrix.sum(axis=1)==1)[0]
    # select the corresponding nuclei
    selected_nuclei = np.argmax(mapping_matrix[selected_cells],axis=1)
    # get list of 1:1 mapped pairs
    assigned_cells_1_1 = np.array([[nucleus, cell] for [nucleus, cell] in assigned_cells if nucleus in selected_nuclei])
    
    # even if there is nothing, return a 2D array:
    if not assigned_cells_1_1.sum():        
        return np.zeros(shape=(0,2))
    return assigned_cells_1_1

def filter_mapped_cells_props(mask_nuclei, mask_cells, mapped_cells, NC_threshold_lower, NC_threshold_upper,
                              apopt_thres_nuclei = None, apopt_thres_cells = None, solidity_threshold = None, eccentricity_treshold = None, neighbors_counts = None,
                              return_props = False):
    """ Calculates basic shape/size parameters for mapped cells.
        (1) Basic filtering: calculates the nuclear:cytoplasmic area ratio and throws out everything not in specified range.
        (2) Apoptotic cells - typically small, isolated and very round. Discard cells that fulfill all of the conditions above:
            - no touching cells
            - area of nuclei and cells smaller than a given threshold (size_thresholds_apoptotic_candidates, e.g. 2750 and 5500),
            - solidity (of cells) and eccentricity (sum of nuclei and cells) above and below provided thresholds (e.g. >0.95, <1.4)
    """
    # return an empty array if there are no object to process
    if not mapped_cells.sum():
        if return_props:
            return mapped_cells.copy(), pd.DataFrame()
        return mapped_cells.copy()
   
    # variable to save pairs which passed the criteria
    mapped_filtered = []
    
    # list of regionprops measurements to collect for each cell and nucleus
    properties = ["label", "area", "eccentricity", "equivalent_diameter_area", "feret_diameter_max", "centroid", "solidity"]

    # extract the properties
    props_nuclei = pd.DataFrame(regionprops_table(mask_nuclei, properties=properties))
    props_cells = pd.DataFrame(regionprops_table(mask_cells, properties=properties))

    # merge cells and nuclei
    props_nuclei.columns = [f"{x}_n" for x in props_nuclei.columns]
    props_cells.columns = [f"{x}_c" for x in props_cells.columns]
    props_merged = pd.DataFrame(mapped_cells)
    #display(props_merged)
    props_merged.columns = ["label_n", "label_c"]
    props_merged = props_merged.merge(props_nuclei).merge(props_cells) 
    # calculated sum of eccentricity of nuclei and cells
    props_merged["eccentricity_sum"] = props_merged["eccentricity_n"] + props_merged["eccentricity_c"]
    # calculate nuclear:cytoplasmic ratio
    props_merged["NC_ratio"] = props_merged["area_n"] / props_merged["area_c"]
        
    #display(props_merged)
    
    if not neighbors_counts is None:
        props_merged = props_merged.merge(neighbors_counts, left_on="label_c", right_on="object")
        
    # filter each cell
    for i in range(props_merged.shape[0]):
        nucleus = props_merged.loc[i, "label_n"]
        cell = props_merged.loc[i, "label_c"]
        NC_ratio = props_merged.loc[i, "NC_ratio"]
        
        # basic filtering (1)
        if NC_ratio < NC_threshold_lower or NC_ratio > NC_threshold_upper:
            continue
        
        # advanced filtering (2)
        # merge with number of neighbors for each cell
        if apopt_thres_nuclei and apopt_thres_cells:
            filter_1 = props_merged.loc[i, "area_n"] < apopt_thres_nuclei and props_merged.loc[i, "area_c"] < apopt_thres_cells and not props_merged.loc[i, "n_neighbors"]
            filter_2 = props_merged.loc[i, "solidity_c"] > solidity_threshold and props_merged.loc[i, "solidity_c"] < eccentricity_treshold
            if filter_1 and filter_2:
                continue
        # add the pair if it passed the criteria   
        mapped_filtered.append([nucleus, cell])
      
    mapped_filtered = np.array(mapped_filtered)
    # if there are no cells left after filtering
    if not mapped_filtered.sum():
        mapped_filtered = np.zeros(shape=(0,2))

    #display(mapped_filtered, props_merged)
        
    if return_props:
        return np.array(mapped_filtered), props_merged
    return np.array(mapped_filtered)

def filter_cells_noborder(mask_nuclei, mask_cells, mapped_cells, border_filtering_offset=2, strict_mode=False):
    """ filter out cells that are touching or too close to border (too close defined by the border_filtering_offset parameter)
    """
    imsize=mask_nuclei.shape[0]
    mapped_noborder = []
    
    # define border pixels
    border_pixels = list(range(0,border_filtering_offset)) + list(range(imsize-border_filtering_offset,imsize))
    
    # get IDs of cells/nuclei that are in the border area
    border_horizontal_cells = mask_cells[border_pixels,:]
    border_horizontal_nuclei = mask_nuclei[border_pixels,:]
    border_vertical_cells = mask_cells[:, border_pixels]
    border_vertical_nuclei = mask_nuclei[:, border_pixels]

    border_cells = np.unique(np.concatenate((border_horizontal_cells, border_vertical_cells), axis = None))
    border_nuclei = np.unique(np.concatenate((border_horizontal_nuclei, border_vertical_nuclei), axis = None))
    
    # filter out cells/nuclei that are on the list
    if strict_mode:
        mapped_noborder = [[nucleus, cell] for [nucleus, cell] in mapped_cells if nucleus not in border_nuclei and cell not in border_cells]
    else:
        mapped_noborder = [[nucleus, cell] for [nucleus, cell] in mapped_cells if nucleus not in border_nuclei]
    
    mapped_noborder = np.array(mapped_noborder)
    if not mapped_noborder.sum():        
        return np.zeros(shape=(0,2))
    return np.array(mapped_noborder)

def get_final_masks(mask_nuclei, mask_cells, mapped_cells):
    """ Create masks of the mapped cells: nuclei, cells, cytoplasm. Make sure that the in all 3 masks the same cells have the same number.
    """
    newmask_nuclei = np.zeros(mask_nuclei.shape, "uint16")
    newmask_cells = np.zeros(mask_cells.shape, "uint16")
    
    # no more pairs remaining
    if not mapped_cells.sum():
        newmask_cytoplasm = np.zeros(mask_nuclei.shape, "uint16")
    else:
        # give the same number to both members of the pair
        for i, [nucleus, cell] in enumerate(mapped_cells):
            newmask_nuclei[mask_nuclei == nucleus] = i+1
            newmask_cells[mask_cells == cell] = i+1
            
            # cytoplasm: take image_cells_filtered and set cell pixels where there is also a nucleus to zero, same values as corresponding pair
            inds_cells_selected = mask_cells == np.array(mapped_cells)[:,1, None, None]
            inds_nuclei_selected = mask_nuclei == np.array(mapped_cells)[:,0, None, None]

            newmask_cytoplasm = newmask_cells*((np.any(inds_cells_selected,axis=0)) & (~np.any(inds_nuclei_selected,axis=0)))
            newmask_cytoplasm = newmask_cytoplasm.astype("uint16")
    
    return newmask_nuclei, newmask_cells, newmask_cytoplasm

# show all detected cells and nuclei + highlight cells after filtering in color
def get_segmentation_inspect_img(detected_nuclei, detected_cells, nuclei_nodebris, cells_nodebris, filtered_cells, colormap, filtered_cells_color=[0.549, 0.604, 0.694, 1.0]):
    
    # create a modified colormap
    colormap = colormap.copy()
    # set (mask) values for objects to show in different colors
    color = min(65535, len(colormap)-1)
    white = color-1
    grey = color-2
    red = color-3
    # set the corresponding colormap values to generate the desired colors
    colormap[color] = filtered_cells_color
    colormap[white] = [0.99,0.99,0.99,1]
    colormap[grey] = [0.71, 0.71, 0.71, 1]
    colormap[red] = [0.804, 0.0, 0.102, 1.0]
    
    newmask = np.zeros(filtered_cells.shape, dtype="uint16")
    # plot all detected cells in grey
    newmask[detected_cells > 0] = grey
    # plot filtered cells in color
    newmask[filtered_cells > 0] = color
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

imagelist = pd.read_csv(imagelist_path)

if not last or last > imagelist.shape[0]:
    last = imagelist.shape[0]
assert first <= last

imagelist_batch = imagelist.iloc[first:last]
n_images = imagelist_batch.shape[0]

colormap = np.load(colormap_path)
cells_paths = [x[5:] for x in imagelist_batch["URL_segmented_cells_mask"]]
nuclei_paths = [x[5:] for x in imagelist_batch["URL_segmented_nuclei_mask"]]
output_paths_nuclei = [x[5:] for x in imagelist_batch["URL_filtered_nuclei"]]
output_paths_cells = [x[5:] for x in imagelist_batch["URL_filtered_cells"]]
output_paths_cytoplasm = [x[5:] for x in imagelist_batch["URL_filtered_cytoplasm"]]
if output_folder_inspect:
    output_paths_inspect = [f"{output_folder_inspect}/{FOV}_segmentation.png" for FOV in imagelist_batch["FOV"]]

noborder_strict_mode = noborder_strict_mode == "strict"

print(f"{imagelist_path=} {first=} {last=} {nuclei_min_size=} {cells_min_size=} {nuclei_overlap_min=} {NC_threshold_lower=} {NC_threshold_upper=} {FOV_border_px=} {expand_px=} {noborder_strict_mode=} {colormap_path=} {output_folder_inspect=} {apopt_thres_nuclei=} {apopt_thres_cells=} {solidity_threshold=} {eccentricity_threshold=}")

cells_masks = [imread(x) for x in cells_paths]
nuclei_masks = [imread(x) for x in nuclei_paths]

print("discarding small objects (cells, nuclei)")
nuclei_nodebris = [filter_objects_size(x, nuclei_min_size, renumber=True) for x in tqdm(nuclei_masks)]
cells_nodebris = [filter_objects_size(x, cells_min_size, renumber=True) for x in tqdm(cells_masks)]

print("1:1 mapping (mapping, generating masks - nuclei, cells)")
mapped = [map_nuclei_cells(nuclei_nodebris[i], cells_nodebris[i], nuclei_overlap_min) for i in trange(n_images)]
mapped_nuclei = [get_mask_selected(nuclei_nodebris[i], mapped[i][:,0]) for i in trange(n_images)]
mapped_cells = [get_mask_selected(cells_nodebris[i], mapped[i][:,1]) for i in trange(n_images)]

print("removing cells from the border")
mapped_noborder = [filter_cells_noborder(mapped_nuclei[i], mapped_cells[i], mapped[i], FOV_border_px, strict_mode=noborder_strict_mode) for i in trange(n_images, file=sys.stdout)]

print("counting neighbors")
neighbors_counts = [count_neighbors(cells_nodebris[i], mapped[i][:,1]) for i in trange(n_images)]
print(f"filtering: {NC_threshold_lower=} {NC_threshold_lower=} {apopt_thres_nuclei=} {apopt_thres_nuclei=} {apopt_thres_cells=} {solidity_threshold=} {eccentricity_threshold=}")
mapped_filtered = [filter_mapped_cells_props(mapped_nuclei[i], mapped_cells[i], mapped_noborder[i], NC_threshold_lower, NC_threshold_upper,
                                             apopt_thres_nuclei, apopt_thres_cells, solidity_threshold, eccentricity_threshold, neighbors_counts[i]) for i in trange(n_images)]
print("generating and saving filtered masks")
filtered_masks = np.array([get_final_masks(nuclei_nodebris[i], cells_nodebris[i], mapped_filtered[i]) for i in trange(n_images, file=sys.stdout)])
filtered_nuclei, filtered_cells, filtered_cytoplasm = filtered_masks[:,0], filtered_masks[:,1], filtered_masks[:,2]
for i in trange(n_images, file=sys.stdout):
    # save in 16-bit format with Pillow because imageio is stupid and does not save it in 16bit format unless there is a value above 255
    Image.fromarray(filtered_nuclei[i]).save(output_paths_nuclei[i]) 
    Image.fromarray(filtered_cells[i]).save(output_paths_cells[i]) 
    Image.fromarray(filtered_cytoplasm[i]).save(output_paths_cytoplasm[i]) 

if output_folder_inspect:
    print("creating inspection images")
    for i in trange(n_images, file=sys.stdout):
        imsave(output_paths_inspect[i], img_as_ubyte(get_segmentation_inspect_img(nuclei_masks[i], cells_masks[i], nuclei_nodebris[i], cells_nodebris[i], filtered_cells[i], colormap)))
