import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
import rasterio.warp
from rasterio.mask import mask
from rasterio.plot import show
from shapely.geometry import Point, Polygon
import geopandas as gpd

from utils import get_bbox_image


def start_points(size, split_size, overlap):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(pt)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def process_one_image(img_path, shfPath):
    # img_path = "./W05_202003281250_RI_RSK_RSKA003603_RGB_2.tif"
    # shfPath = "./output1/with-shapely.shp"
    # shfPath = "./W05_202003281250_RI_RSK_RSKA003603_RGB/W05_202003281250_RI_RSK_RSKA003603_RGB.shp"
    print("Process {}".format(img_path.split("/")[-1]))
    big_image_name = img_path.split("/")[-1].split("_")[-3]
    idx_img = img_path.split("/")[-1][:-4][-1]
    # import ipdb; ipdb.set_trace()
    dataset = rasterio.open(img_path)
    with rasterio.open(img_path, 'r') as ds:
        arr = ds.read()  # read all raster values
    # read shapefile
    polygons = gpd.read_file(shfPath)
    print(dataset.height,dataset.width,dataset.transform,dataset.crs)
    
    dataset_coords = [(dataset.bounds[0], dataset.bounds[1]), (dataset.bounds[2], dataset.bounds[1]), (dataset.bounds[2], dataset.bounds[3]), (dataset.bounds[0], dataset.bounds[3])]
    dataset_polygon = Polygon(dataset_coords)

    df_bounds = polygons.bounds
    df_bounds['centerx']=(df_bounds['minx']+df_bounds['maxx'])/2
    df_bounds['centery']=(df_bounds['miny']+df_bounds['maxy'])/2
    
    lst_idx_inside_img = [index for index, bbox in enumerate(df_bounds.values.tolist()) if Point(bbox[4], bbox[5]).within(dataset_polygon)]

    #convert to 3d axis for process
    rgb1= np.rollaxis(arr, 0,3)  
    print(rgb1.shape)
    img_h, img_w, _ = rgb1.shape
    split_width = 256
    split_height = 256

    X_points = start_points(img_w, split_width, 0.5)
    Y_points = start_points(img_h, split_height, 0.5)  
    count = 0
    name = 'splitted'
    frmt = 'png'

    for i in Y_points: #index row
        for j in X_points: #index col
            img_name = "Img_{}_{}_r{:03d}_c{:03d}".format(big_image_name,idx_img,i,j)
            # import ipdb; ipdb.set_trace()
            if X_points.index(j) == len(X_points)-1: #j is the last point 
                if Y_points.index(i) == len(Y_points)-1: #i is the last point
                    split = rgb1[i:, j:] 
                    # print("Image.r%03d_c%03d"%(i,j))
                    # print(split.shape)
                    im_arr_bgr = cv2.cvtColor(split, cv2.COLOR_RGB2BGR)
                    # img_name = "Img_r%03d_c%03d"%(i,j)
                    cv2.imwrite('./images/{}.png'.format(img_name), im_arr_bgr)
                    get_bbox_image(rgb1, img_name,dataset,df_bounds, lst_idx_inside_img,X_points, Y_points, i, j)
                    count += 1
                    continue
                else:
                    split = rgb1[i:i+split_height, j:]
                    # print("Image.r%03d_c%03d"%(i,j))
                    # print(split.shape)
                    im_arr_bgr = cv2.cvtColor(split, cv2.COLOR_RGB2BGR)
                    # img_name = "Img_r%03d_c%03d"%(i,j)
                    cv2.imwrite('./images/{}.png'.format(img_name), im_arr_bgr)
                    get_bbox_image(rgb1, img_name,dataset,df_bounds, lst_idx_inside_img,X_points, Y_points, i, j)
                    count += 1
                    continue
            elif Y_points.index(i) == len(Y_points)-1:
                split = rgb1[i:, j:j+split_width] 
                # print("Image.r%03d_c%03d"%(i,j))
                # print(split.shape)
                im_arr_bgr = cv2.cvtColor(split, cv2.COLOR_RGB2BGR)
                # img_name = "Img_r%03d_c%03d"%(i,j)
                cv2.imwrite('./images/{}.png'.format(img_name), im_arr_bgr)
                get_bbox_image(rgb1, img_name,dataset,df_bounds, lst_idx_inside_img,X_points, Y_points, i, j)
                count += 1
                continue
            else:            
                split = rgb1[i:i+split_height, j:j+split_width]
                # print("Image.r%03d_c%03d"%(i,j))
                # print(split.shape)
                im_arr_bgr = cv2.cvtColor(split, cv2.COLOR_RGB2BGR)
                # img_name = "Img_r%03d_c%03d"%(i,j)
                cv2.imwrite('./images/{}.png'.format(img_name), im_arr_bgr)
                get_bbox_image(rgb1, img_name,dataset,df_bounds, lst_idx_inside_img,X_points, Y_points, i, j)
                count += 1
    print("Done image {}".format(img_path.split("/")[-1]))
    print("splitted into {} images".format(count))  
    print("-------------------------------------------------------------")


if __name__ == "__main__":
    # Read raster image (tiff)
    img_path = "./W05_202003281250_RI_RSK_RSKA003603_RGB_2.tif"
    shfPath = "./output1/with-shapely.shp"
    shfPath = "./W05_202003281250_RI_RSK_RSKA003603_RGB/W05_202003281250_RI_RSK_RSKA003603_RGB.shp"
    dataset = rasterio.open(img_path)
    with rasterio.open(img_path, 'r') as ds:
        arr = ds.read()  # read all raster values
    # read shapefile
    polygons = gpd.read_file(shfPath)
    print(dataset.height,dataset.width,dataset.transform,dataset.crs)
    
    dataset_coords = [(dataset.bounds[0], dataset.bounds[1]), (dataset.bounds[2], dataset.bounds[1]), (dataset.bounds[2], dataset.bounds[3]), (dataset.bounds[0], dataset.bounds[3])]
    dataset_polygon = Polygon(dataset_coords)

    df_bounds = polygons.bounds
    df_bounds['centerx']=(df_bounds['minx']+df_bounds['maxx'])/2
    df_bounds['centery']=(df_bounds['miny']+df_bounds['maxy'])/2
    
    lst_idx_inside_img = [index for index, bbox in enumerate(df_bounds.values.tolist()) if Point(bbox[4], bbox[5]).within(dataset_polygon)]

    #convert to 3d axis for process
    rgb1= np.rollaxis(arr, 0,3)  
    print(rgb1.shape)
    img_h, img_w, _ = rgb1.shape
    split_width = 256
    split_height = 256

    X_points = start_points(img_w, split_width, 0.5)
    Y_points = start_points(img_h, split_height, 0.5)  
    count = 0
    name = 'splitted'
    frmt = 'png'

    for i in Y_points: #index row
        for j in X_points: #index col
            if X_points.index(j) == len(X_points)-1: #j is the last point 
                if Y_points.index(i) == len(Y_points)-1: #i is the last point
                    split = rgb1[i:, j:] 
                    print("Image.r%03d_c%03d"%(i,j))
                    print(split.shape)
                    im_arr_bgr = cv2.cvtColor(split, cv2.COLOR_RGB2BGR)
                    img_name = "Img_r%03d_c%03d"%(i,j)
                    cv2.imwrite('./output2.1/Img_r%03d_c%03d.png'%(i,j), im_arr_bgr)
                    get_bbox_image(rgb1, img_name,dataset,df_bounds, lst_idx_inside_img,X_points, Y_points, i, j)
                    count += 1
                    continue
                else:
                    split = rgb1[i:i+split_height, j:]
                    print("Image.r%03d_c%03d"%(i,j))
                    print(split.shape)
                    im_arr_bgr = cv2.cvtColor(split, cv2.COLOR_RGB2BGR)
                    img_name = "Img_r%03d_c%03d"%(i,j)
                    cv2.imwrite('./output2.1/Img_r%03d_c%03d.png'%(i,j), im_arr_bgr)
                    get_bbox_image(rgb1, img_name,dataset,df_bounds, lst_idx_inside_img,X_points, Y_points, i, j)
                    count += 1
                    continue
            elif Y_points.index(i) == len(Y_points)-1:
                split = rgb1[i:, j:j+split_width] 
                print("Image.r%03d_c%03d"%(i,j))
                print(split.shape)
                im_arr_bgr = cv2.cvtColor(split, cv2.COLOR_RGB2BGR)
                img_name = "Img_r%03d_c%03d"%(i,j)
                cv2.imwrite('./output2.1/Img_r%03d_c%03d.png'%(i,j), im_arr_bgr)
                get_bbox_image(rgb1, img_name,dataset,df_bounds, lst_idx_inside_img,X_points, Y_points, i, j)
                count += 1
                continue
            else:            
                split = rgb1[i:i+split_height, j:j+split_width]
                print("Image.r%03d_c%03d"%(i,j))
                print(split.shape)
                im_arr_bgr = cv2.cvtColor(split, cv2.COLOR_RGB2BGR)
                img_name = "Img_r%03d_c%03d"%(i,j)
                cv2.imwrite('./output2.1/Img_r%03d_c%03d.png'%(i,j), im_arr_bgr)
                get_bbox_image(rgb1, img_name,dataset,df_bounds, lst_idx_inside_img,X_points, Y_points, i, j)
                count += 1
    print("splitted into {} images".format(count))  
