import pandas as pd
import numpy as np
from skimage.segmentation import expand_labels


def renumber_mask(mask):
    mask_renumbered = np.zeros(shape=mask.shape, dtype="uint16")
    labels = np.unique(mask)
  
    # renumber the objects, keep order (e.g. 1,2,4,5 -> 1,2,3,4)
    for i, label in enumerate(labels):
        mask_renumbered[mask == label] = i
    return mask_renumbered

def filter_objects_size(mask, min_size=1500, max_size = None, renumber=False):   
    # for each object, calculate how many pixels it occupies
    area_counts = np.array(np.unique(mask, return_counts=True)).T
    # get numbers of cells to remove
    to_remove = [x[0] for x in area_counts if x[1] < min_size]
    if max_size:
        to_remove += [x[0] for x in area_counts if x[1] > max_size]
    # set the mask values for these nuclei to zero (= background)
    result = mask.copy()
    for idx in to_remove:
        result[result==idx] = 0
    if renumber:
        return renumber_mask(result)
    return result

def get_mask_selected(mask, indices):
    result = np.zeros(shape=mask.shape, dtype="uint16")
    # if there are no cells, return an empty mask
    if not np.sum(indices):
        return result
    for idx in indices:
        result[mask == idx] = idx
    #plt.imshow(colormap[result]); plt.show()
    return result


def count_neighbors(mask, objects = None, expand_px = 5):
    if objects is None:
        objects = [x for x in np.unique(mask) if x != 0]
        
    collector = []
    # for each object, get the corresponding mask, extend it by 2 px
    # whatever objects are in this extended mask are the neighbors
    for obj in objects:
        mask_object = mask == obj
        mask_object_expanded = expand_labels(mask_object, expand_px)
        neighbors = [x for x in np.unique(mask[mask_object_expanded]) if x not in [0,obj]]
        n_neighbors = len(np.unique(mask[mask_object_expanded]))-2
        #print(obj, neighbors, n_neighbors)
        collector.append(n_neighbors)
    return pd.DataFrame({"object": objects, "n_neighbors": collector})
