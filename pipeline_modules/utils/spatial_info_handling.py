import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import re
from .misc_utils import get_well_ID, move_columns_to_front

# adapted from https://github.com/VasylVaskivskyi/biostitch/blob/ab0c2194e81369b987b2119944b541ee5e7ecbf8/biostitch/image_positions.py#L67
# flips the y axis - for images, it has the opposite orientation
def zero_center_coordinates(x_pos, y_pos):
    leftmost = min(x_pos)
    top = max(y_pos)

    if leftmost < 0:
        leftmost = abs(leftmost)
        if top < 0:
            top = abs(top)
            x_pos = [i + leftmost for i in x_pos]
            y_pos = [abs(i) - top for i in y_pos]
        else:
            x_pos = [i + leftmost for i in x_pos]
            y_pos = [top - i for i in y_pos]
    else:
        if top < 0:
            top = abs(top)
            x_pos = [i - leftmost for i in x_pos]
            y_pos = [abs(i) - top for i in y_pos]
        else:
            x_pos = [i - leftmost for i in x_pos]
            y_pos = [top - i for i in y_pos]

    return x_pos, y_pos

# adds the shifted coords - done well by well in case different wells have different FOV numbers/coordinates/whatever
def add_shifted_coords_per_well(df):
    for well in df.well.unique():
        df_well = df[df.well == well].copy()
        new_coords_px = zero_center_coordinates(df_well["posX_px"], df_well["posY_px"])
        df_well["posX_px_zero"], df_well["posY_px_zero"] = new_coords_px[0], new_coords_px[1]
        yield df_well       
# wrapper
def add_shifted_coords(df): yield from add_shifted_coords_per_well(df)

# helper function to extract info from index.xml file from Opera - for files generated by Harmony software V6
def parse_xml_index_partial_harmonyV6(xml):
    # navigate to the part which has image size and resolution of each pixel, take the values from the first channel
    tag_Maps = xml.find('Maps')[2][0]
    x_resol = round(float(tag_Maps.find('ImageResolutionX').text), 23)
    y_resol = round(float(tag_Maps.find('ImageResolutionY').text), 23)
    image_size_x = int(tag_Maps.find('ImageSizeX').text)
    image_size_y = int(tag_Maps.find('ImageSizeY').text)
    
    # extract necessary information about each image in the dataset
    metadata_list = []
    tag_Images = xml.find('Images')
    for img in tag_Images:
        # (absolute) coordinates of each FOV
        x_coord = round(float(img.find('PositionX').text), 10)  # limit precision to nm
        y_coord = round(float(img.find('PositionY').text), 10)

        # field id, plane id, channel name, file name
        metadata_list.append([
            int(img.find('Row').text),
            int(img.find('Col').text),
            int(img.find('FieldID').text),
            int(img.find('PlaneID').text),
            x_coord,
            y_coord,
            round(x_coord / x_resol),
            round(y_coord / y_resol)
            ])

    # convert to dataframe
    metadata = pd.DataFrame(metadata_list)
    metadata = metadata.drop_duplicates().reset_index(drop=True)
    return metadata, image_size_x, image_size_y

# helper function to extract info from index.xml file from Opera - for files generated by Harmony software V5
def parse_xml_index_partial_harmonyV5(xml):  
    tag_Images = xml.find('Images')
    # get the resolution (i.e. size of each pixel)
    x_resol = round(float(tag_Images[0].find('ImageResolutionX').text), 23)
    y_resol = round(float(tag_Images[0].find('ImageResolutionY').text), 23)

    try:
        image_size_x = int(tag_Images[0].find('ImageSizeX').text)
        image_size_y = int(tag_Images[0].find('ImageSizeY').text)
    except:
        print("Tag 'ImageSizeX' not found. Defaulting to 1080.")
        image_size_x = 1080
        image_size_y = 1080

    # extract necessary information about each image in the dataset
    metadata_list = []
    for img in tag_Images:
        # (absolute) coordinates of each FOV
        x_coord = round(float(img.find('PositionX').text), 10)  # limit precision to nm
        y_coord = round(float(img.find('PositionY').text), 10)

        # field id, plane id, channel name, file name
        metadata_list.append([
            int(img.find('Row').text),
            int(img.find('Col').text),
            int(img.find('FieldID').text),
            int(img.find('PlaneID').text),
            #img.find('ChannelName').text,
            #img.find('URL').text,
            x_coord,
            y_coord,
            round(x_coord / x_resol),
            round(y_coord / y_resol)
            ])

    # convert to dataframe
    metadata = pd.DataFrame(metadata_list)
    metadata = metadata.drop_duplicates().reset_index(drop=True)
    return metadata, image_size_x, image_size_y

# get metadata about position of each FOV from the index.xml file generated by harmony
def parse_xml_index(xml_path, opera_harmony_version="V5"):
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_file = f.read()
        f.close()

    xml_file = re.sub(r'xmlns="http://www.perkinelmer.com/PEHH/HarmonyV\d"', '', xml_file)
    xml = ET.fromstring(xml_file)
    
    if opera_harmony_version == "V6":
        metadata, image_size_x, image_size_y =  parse_xml_index_partial_harmonyV6(xml)
    else:
        metadata, image_size_x, image_size_y =  parse_xml_index_partial_harmonyV5(xml)
    
    metadata.columns = ["row", "column", "FOV", "plane", "posX_abs", "posY_abs", "posX_px", "posY_px"]
    metadata = metadata.sort_values(["row", "column", "FOV", "plane"])
    # paths to tagged channels
    metadata["image"] = [f'r{r:02d}c{c:02d}f{f:02d}p{p:02d}' for r,c,f,p in np.array(metadata[["row", "column", "FOV", "plane"]])]
    metadata["well"] = get_well_ID(metadata)
    metadata = move_columns_to_front(metadata, ["well"])
    metadata = pd.concat(add_shifted_coords(metadata))

    return metadata, image_size_x, image_size_y
# normalizes 1D array by maximum value, i.e. sets maximum value to 1 and scales everything else accordingly
def normalize_1D(arr):
    if len(arr) == 0:
        return np.array([])
    return arr / np.max(arr)

# default reverse distance function: sigmoid-ish, rescale units a bit (divide by 10)
def get_distance_scores(clones, distances, reverse_distance_func=lambda x: 1 / ((x/10 )** 2), normalize=False):
    if normalize:
        return np.unique(clones), normalize_1D([reverse_distance_func(distances[clones == clone]).sum() for clone in np.unique(clones)])
    else:
        return np.unique(clones), normalize_1D([reverse_distance_func(distances[clones == clone]).sum() for clone in np.unique(clones)])
    
