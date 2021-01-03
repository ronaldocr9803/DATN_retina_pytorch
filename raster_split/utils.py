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
from rasterio.mask import mask
from rasterio.plot import show
from skimage.io import concatenate_images, imread, imshow
from shapely.geometry import Point, Polygon


def add_bbox_img(dataset,df_bounds, img, bboxes,start_row,start_col):
    re = []
    for bbox in bboxes:
        center_point = Point(dataset.index(df_bounds.values.tolist()[bbox][4],df_bounds.values.tolist()[bbox][5]))
        if center_point.within(Polygon([(start_row,start_col), (start_row ,start_col+256), (start_row + 256,start_col+256), (start_row + 256,start_col)])):
            re.append(bbox)
    return re


def cvt_rowmin_colmin(dataset,df_bounds, index,count_x, count_y):
    row_col = list(dataset.index(df_bounds.values.tolist()[index][0],df_bounds.values.tolist()[index][3]))
    if row_col[0] > 256 or count_y != 0: #row min
        row_col[0] = row_col[0] - 128*count_y
    if row_col[1] > 256 or count_x != 0: #col min
        row_col[1] = row_col[1] - 128*count_x
    row_col = [0 if i < 0 else i for i in row_col]
    row_col = [256 if i > 256 else i for i in row_col]
    return tuple(row_col)

def cvt_rowmax_colmax(dataset,df_bounds, index,count_x, count_y):
    row_col = list(dataset.index(df_bounds.values.tolist()[index][2],df_bounds.values.tolist()[index][1]))
    if row_col[0] > 256 or count_y != 0: #row min
        row_col[0] = row_col[0] - 128*count_y
    if row_col[1] > 256 or count_x != 0: #row min
        row_col[1] = row_col[1] - 128*count_x
    row_col = [0 if i < 0 else i for i in row_col]
    row_col = [256 if i > 256 else i for i in row_col]
    return tuple(row_col)


def get_bbox_image(img_arr,name, dataset,df_bounds, bboxes,X_points, Y_points, start_row, start_col):
    test_arr = img_arr[start_row:start_row+256,start_col:start_col+256]
    im_arr_bgr = cv2.cvtColor(test_arr, cv2.COLOR_RGB2BGR)
    lst_obj_inside_img = add_bbox_img(dataset,df_bounds, im_arr_bgr, bboxes, start_row,start_col)
    f= open("./labels/{}.txt".format(name),"w+")
    for i in lst_obj_inside_img:
        row_min, col_min = cvt_rowmin_colmin(dataset,df_bounds, i, X_points.index(start_col), Y_points.index(start_row))
        row_max, col_max = cvt_rowmax_colmax(dataset,df_bounds, i, X_points.index(start_col), Y_points.index(start_row))
        f.write("{} {} {} {}\n".format(col_min,row_min, col_max, row_max))
    f.close()
#         cv2.rectangle(im_arr_bgr,(col_min,row_min),(col_max, row_max), color=(0, 255, 0), thickness=1)

#     fig, ax = plt.subplots(1, 2, figsize=(10, 10))

#     # rgb1= np.rollaxis(rgb1, 0,3)  
#     ax[0].imshow(im_arr_bgr)
#     ax[0].set_title('image')
#     ax[1].imshow(masks[start_row:start_row + 256,start_col:start_col + 256], cmap='gray')
#     ax[1].set_title('Masks 1')

# test_arr = rgb1[0:0+256, 128:128+256]

# get_bbox_image( re, 384,256+256)
