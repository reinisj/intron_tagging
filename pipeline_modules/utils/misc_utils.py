import pandas as pd
import numpy as np
import re

def get_well_ID(df, zeropad = True):
    df2 = df.reset_index(drop=True).copy()
    if zeropad:
        return [f'{chr(df2.loc[i, "row"]+64)}{df2.loc[i, "column"]:02d}' for i in range(len(df2))]
    else:
        return [f'{chr(df2.loc[i, "row"]+64)}{df2.loc[i, "column"]}' for i in range(len(df2))]

def get_color_codes_pred_probas(predictions):
    
    # split interval [0,1] into 5 bins and assign each a color
    bins = np.arange(0,1.1,0.2)
    colors_labels = ["darkred", "red", "orange", "green", "darkgreen"]
    
    # for conversion of the color label to RGB representation 
    colors_RGB = np.array([
        [175,0,0],    # interval (0.0, 0.2], dark red
        [254,11,12],  # interval (0.2, 0.4], red
        [244,115,56], # interval (0.4, 0.6], orange
        [36,135,36],  # interval (0.6, 0.8], green
        [0,58,36]]    # interval (0.8, 1.0], dark green
    )
    colors_RGB = np.round(colors_RGB/255, 3)
    colors = {color:encoding for color,encoding in zip (colors_labels,colors_RGB)}

    predictions["color"] = pd.cut(predictions["clone_predicted_prob"], bins, labels=colors_labels)
    predictions["color_RGB"] = [colors[x] for x in predictions["color"]]
    return predictions

def move_columns_to_front(df, cols):
    return df[cols + [column for column in df.columns if column not in cols]]

def extract_row_column_FOV(df):
    df["row"] = [re.search(r"r(\d+)c(\d+)f(\d+)p", x).groups()[0] for x in df["image"]]
    df["column"] = [re.search(r"r(\d+)c(\d+)f(\d+)p", x).groups()[1] for x in df["image"]]
    df["FOV"] = [re.search(r"r(\d+)c(\d+)f(\d+)p", x).groups()[2] for x in df["image"]]


def get_well_from_FOVID(FOVs):
    rows_columns = [re.search(r"r(\d+)c(\d+)f(\d+)p", x).groups()[:2] for x in FOVs]
    return [chr(int(row)+64) + col for row, col in rows_columns]
