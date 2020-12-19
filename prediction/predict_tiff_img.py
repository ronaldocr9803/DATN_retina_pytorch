import glob
import os

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
import rasterio.warp
import torch
import torchvision.transforms as T
from natsort import natsorted
from rasterio.mask import mask
from rasterio.plot import show
from shapely.geometry import Point, Polygon
from tqdm import tqdm

# from model import fasterrcnn_resnet101_fpn
from prediction.utils import compute_windows, non_max_suppression, convert_xy_tif, init_model
# from split_image import start_points
# from utils import get_bbox_image
import pandas as pd
import warnings

with warnings.catch_warnings():
    # Suppress some of the verbose tensorboard warnings,
    # compromise to avoid numpy version errors
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

CLASS_NAMES = ["__background__", "tree"]



def get_prediction(model, img_arr, confidence):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold value for prediction score
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.
    
    """
    # img = Image.open(img_path)
    my_transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img = my_transform(img_arr)#.to(device)
    model = model.to(torch.device("cuda:0"))
    # pred = model([img])
    pred = model([img.to(torch.device("cuda:0"))])
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = pred[0]['boxes'].detach().cpu().numpy()
    # pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())

    pred_t = [pred_score.index(x) for x in pred_score if x>confidence]
    if len(pred_t) == 0:
        return None
    pred_t = pred_t[-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_score = pred_score[:pred_t+1]


    # x1_lst, y1_lst, x2_lst, y2_lst = [np.expand_dims(pred[0]['boxes'].detach().cpu().numpy()[:,i], axis=1) for i in range(pred[0]['boxes'].detach().cpu().numpy().shape[1])] 
    x1_lst, y1_lst, x2_lst, y2_lst = [np.expand_dims(pred_boxes[:,i], axis=1) for i in range(pred_boxes.shape[1])] 
    # import ipdb; ipdb.set_trace()
    image_detections = np.concatenate([
        x1_lst, y1_lst, x2_lst, y2_lst,
        np.expand_dims(pred_score, axis=1),
        np.expand_dims(pred_class, axis=1)
    ], axis=1)
    import pandas as pd
    df = pd.DataFrame(image_detections,
                      columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])
    # import ipdb; ipdb.set_trace()
    return df        #pred_boxes, pred_class, pred_score
   
def detect_object(model, img_arr, confidence=0.5, rect_th=1, text_size=0.35, text_th=1):
    """
    object_detection_api
      parameters:
        - img_path - path of the input image
        - confidence - threshold value for prediction score
        - rect_th - thickness of bounding box
        - text_size - size of the class label text
        - text_th - thichness of the text
      method:
        - prediction is obtained from get_prediction method
        - for each prediction, bounding box is drawn and text is written 
          with opencv
        - the final image is displayed
    """
    boxes, pred_cls, pred_score = get_prediction(model, img_arr, confidence)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(len(boxes))
    for i in range(len(boxes)):
      cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
      cv2.putText(img,str(round(pred_score[i],3)), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (36,255,12),thickness=text_th)
    fig = plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    # plt.show()
    plt.draw()
    plt.waitforbuttonpress(0) # this will wait for indefinite time
    plt.close(fig)


def process_one_image(img_path, shfPath):

    if not os.path.exists('./data/training_data'):
        os.makedirs('./data/training_data')

    model = init_model()
    # img_path = "./W05_202003281250_RI_RSK_RSKA003603_RGB_2.tif"
    # shfPath = "./output1/with-shapely.shp"
    # shfPath = "./W05_202003281250_RI_RSK_RSKA003603_RGB/W05_202003281250_RI_RSK_RSKA003603_RGB.shp"
    # import ipdb; ipdb.set_trace()
    big_image_name = img_path.split("/")[-1].split("_")[-3]
    idx_img = img_path.split("/")[-1][:-4][-1]
    dataset = rasterio.open(img_path)
    with rasterio.open(img_path, 'r') as ds:
        arr = ds.read()  # read all raster values
    # read shapefile
    polygons = gpd.read_file(shfPath)
    print(dataset.height,dataset.width,dataset.transform,dataset.crs)

    
    # lst_idx_inside_img = [index for index, bbox in enumerate(df_bounds.values.tolist()) if Point(bbox[4], bbox[5]).within(dataset_polygon)]

    #convert to 3d axis for process
    rgb1= np.rollaxis(arr, 0,3)  
    # print(rgb1.shape)
    # img_h, img_w, _ = rgb1.shape


    windows = compute_windows(rgb1, 256, 0.5) 
    # Save images to tmpdir
    predicted_boxes = []

    for index, window in enumerate(tqdm(windows)):
        # Crop window and predict
        crop = rgb1[windows[index].indices()]
        # import ipdb; ipdb.set_trace()
        # # Crop is RGB channel order, change to BGR
        # crop = crop[..., ::-1]
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        # pred_img1 = get_prediction(model, crop, confidence=0.7)
        boxes = get_prediction(model, crop, confidence=0.7)
        if boxes is None:
            continue
        boxes['xmin'] = pd.to_numeric(boxes['xmin'])
        boxes['ymin'] = pd.to_numeric(boxes['ymin'])
        boxes['xmax'] = pd.to_numeric(boxes['xmax'])
        boxes['ymax'] = pd.to_numeric(boxes['ymax'])

        # transform coordinates to original system
        xmin, ymin, w, h = windows[index].getRect() #(x,y,w,h)
        boxes.xmin = boxes.xmin + xmin
        boxes.xmax = boxes.xmax + xmin
        boxes.ymin = boxes.ymin + ymin
        boxes.ymax = boxes.ymax + ymin

        predicted_boxes.append(boxes)
        # if index == 3:    #break to test some first images
        #   break
        

    predicted_boxes = pd.concat(predicted_boxes)
    # Apply NMS 
    with tf.Session() as sess:
        print(
            "{} predictions in overlapping windows, applying non-max supression". \
            format(predicted_boxes.shape[0]))
        new_boxes, new_scores, new_labels = non_max_suppression(
            sess,
            predicted_boxes[["xmin", "ymin", "xmax", "ymax"]].values,
            predicted_boxes.score.values,
            predicted_boxes.label.values,
            max_output_size=predicted_boxes.shape[0],
            iou_threshold=0.5)
        # import ipdb; ipdb.set_trace()

        # Recreate box dataframe
        image_detections = np.concatenate([
            new_boxes,
            np.expand_dims(new_scores, axis=1),
            np.expand_dims(new_labels, axis=1)
        ], axis=1)

    df = pd.DataFrame(
        new_boxes,
        columns=["xmin", "ymin", "xmax", "ymax"])     #, "score", "label"])
    # df.label = df.label.str.decode("utf-8")
    df['geometry'] = df.apply(lambda x: convert_xy_tif(x, dataset), axis=1)
    df_res = gpd.GeoDataFrame(df, geometry='geometry')
    # import ipdb; ipdb.set_trace()
    df_res.to_file('./output_pred/with_shapely_pred.shp', driver='ESRI Shapefile')
    print("-----------Done--------------")
    # import ipdb; ipdb.set_trace()



if __name__ == "__main__":
    if not os.path.exists('./output_pred'):
        os.makedirs('./output_pred')
    # lst_raster = ['W03_202003311249_RI_RSK_RSKA014702_RGB']
    lst_raster = ["W05_202003281250_RI_RSK_RSKA003603_RGB"]
    for raster in lst_raster:
        # print("Process {}...".format(raster))
        lst_raster_img = natsorted([i for i in os.listdir("../data/"+raster) if i[-4:]=='.tif'])
        raster_img = lst_raster_img[-1]
        # for raster_img in lst_raster_img:
        print("Process {}...".format(raster_img))
        img_path = os.path.join("../data", raster, raster_img)
        shfPath = os.path.join("../data", raster, "{}.shp".format(raster))
        process_one_image(img_path, shfPath)
