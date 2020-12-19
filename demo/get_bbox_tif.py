import os

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.warp
import shapely
from fiona.crs import from_epsg
from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from rasterio.mask import mask
from rasterio.plot import show
from shapely.geometry import Point, Polygon
from skimage.io import concatenate_images, imread, imshow
from tqdm import tqdm


def bbox_is_in_img(bbox, dataset_polygon):
    if Point(bbox[0], bbox[1]).within(dataset_polygon) or Point(bbox[0], bbox[3]).within(dataset_polygon) \
        or Point(bbox[2], bbox[1]).within(dataset_polygon) or Point(bbox[2], bbox[3]).within(dataset_polygon):
        return True
    else:
        return False

def cvt_row_col(dataset,df_bounds, index, type):
    if type == "max":
        row_col = list(dataset.index(df_bounds.values.tolist()[index][2],df_bounds.values.tolist()[index][1]))
        row_col = [0 if i < 0 else i for i in row_col]
        row_col[0] = dataset.shape[0] if row_col[0] > dataset.shape[0] else row_col[0]
        row_col[1] = dataset.shape[1] if row_col[1] > dataset.shape[1] else row_col[1]
        return tuple(row_col)
    else:
        row_col = list(dataset.index(df_bounds.values.tolist()[index][0],df_bounds.values.tolist()[index][3]))
        row_col = [0 if i < 0 else i for i in row_col]
        row_col[0] = dataset.shape[0] if row_col[0] > dataset.shape[0] else row_col[0]
        row_col[1] = dataset.shape[1] if row_col[1] > dataset.shape[1] else row_col[1]
        return tuple(row_col)

def get_ground_truths_bbox(img_path, shpfile_path):
    allBoundingBoxes = BoundingBoxes()
    img_name = img_path.split("/")[-1][:-4]
    dataset = rasterio.open(img_path)
    polygons = gpd.read_file(shfPath)
    #bound of raster image tiff
    dataset_coords = [(dataset.bounds[0], dataset.bounds[1]), (dataset.bounds[2], dataset.bounds[1]), (dataset.bounds[2], dataset.bounds[3]), (dataset.bounds[0], dataset.bounds[3])]
    dataset_polygon = Polygon(dataset_coords)
    # import ipdb; ipdb.set_trace()
    df_bounds = polygons.bounds
    re = [index for index, bbox in enumerate(df_bounds.values.tolist()) if bbox_is_in_img(bbox, dataset_polygon)]
    for index in tqdm(re):
        # rowmax_colmax
        row_max, col_max = cvt_row_col(dataset, df_bounds, index, "max")
        row_min, col_min = cvt_row_col(dataset, df_bounds, index, "min")
        bb = BoundingBox(
                img_name,
                "tree",
                col_min,
                row_min,
                col_max,
                row_max,
                CoordinatesType.Absolute, None,
                BBType.GroundTruth,
                format=BBFormat.XYXY)
        allBoundingBoxes.addBoundingBox(bb)
    return allBoundingBoxes


if __name__ == "__main__":
    if not os.path.exists('./demo/groundtruths'):
        os.makedirs('./demo/groundtruths')
    if not os.path.exists('./demo/detections/'):
        os.makedirs('./demo/detections/')

    if not os.path.exists('./demo/output_pred'):
        os.makedirs('./demo/output_pred')
    shfPath = "../data/W05_202003281250_RI_RSK_RSKA003603_RGB/W05_202003281250_RI_RSK_RSKA003603_RGB.shp"
    # shfPath = "./output1/with-shapely.shp"
    img_path = "../data/W05_202003281250_RI_RSK_RSKA003603_RGB/W05_202003281250_RI_RSK_RSKA003603_RGB_0.tif"
    img_name = img_path.split("/")[-1][:-4]
    dataset = rasterio.open(img_path)
    polygons = gpd.read_file(shfPath)
    print(dataset.height,dataset.width,dataset.transform,dataset.crs)
    with rasterio.open(img_path, 'r') as ds:
        arr_img = ds.read()  # read all raster values

    #bound of raster image tiff
    from shapely.geometry import Point, Polygon
    dataset_coords = [(dataset.bounds[0], dataset.bounds[1]), (dataset.bounds[2], dataset.bounds[1]), (dataset.bounds[2], dataset.bounds[3]), (dataset.bounds[0], dataset.bounds[3])]
    print(dataset_coords)
    dataset_polygon = Polygon(dataset_coords)
    # import ipdb; ipdb.set_trace()
    print(polygons.bounds)
    df_bounds = polygons.bounds
    re = [index for index, bbox in enumerate(df_bounds.values.tolist()) if bbox_is_in_img(bbox, dataset_polygon)]
    import ipdb; ipdb.set_trace()
    f= open("./demo/groundtruths/{}.txt".format(img_name),"w+")
    for index in tqdm(re[:10]):
        # rowmax_colmax
        row_max, col_max = cvt_row_col(dataset, df_bounds, index, "max")
        row_min, col_min = cvt_row_col(dataset, df_bounds, index, "min")
        f.write("{} {} {} {} {}\n".format("tree",col_min,row_min, col_max, row_max))
    f.close()
