import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse

import glob
import os
import warnings

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torchvision.transforms as T
from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import *
from natsort import natsorted
from shapely.geometry import Point, Polygon
from tqdm import tqdm

def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(image_path, model_path, class_list):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    model = torch.load(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    for img_name in os.listdir(image_path):

        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            print(image.shape, image_orig.shape, scale)
            scores, classification, transformed_anchors = model(image.cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                print(bbox, classification.shape)
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name, score)
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            cv2.imwrite("./result/{}".format(img_name), image_orig)
            # cv2.imshow

def get_prediction(model, img_arr, labels, confidence):
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
    # image = cv2.imread(os.path.join(image_path, img_name))
    # if image is None:
    #     continue
    image_orig = img_arr.copy()

    rows, cols, cns = img_arr.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side
    # import ipdb; ipdb.set_trace()
    # resize the image with the computed scale
    img_arr = cv2.resize(img_arr, (int(round(cols * scale)), int(round((rows * scale)))))
    rows, cols, cns = img_arr.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = img_arr.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))


    with torch.no_grad():

        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()

        st = time.time()
        # print(image.shape, image_orig.shape, scale)
        scores, classification, transformed_anchors = model(image.cuda().float())
        # print('Elapsed time: {}'.format(time.time() - st))
        idxs = np.where(scores.cpu() > 0.5)
        # print([np.expand_dims(transformed_anchors[:,i], axis=1) for i in range(transformed_anchors.shape[1])])
        # import ipdb; ipdb.set_trace()
        # pred_score = scores.detach().cpu().numpy())
        idxs_max = idxs[0].shape[0]
        if idxs_max == 0:
            return None
        pred_boxes = transformed_anchors[:idxs_max].detach().cpu().numpy()/scale
        pred_class = [labels[i] for i in classification[:idxs_max].detach().cpu().numpy()]
        pred_score = scores[:idxs_max].detach().cpu().numpy()
        x1_lst, y1_lst, x2_lst, y2_lst = [np.expand_dims(pred_boxes[:,i], axis=1) for i in range(pred_boxes.shape[1])] 
        image_detections = np.concatenate([
            x1_lst, y1_lst, x2_lst, y2_lst,
            np.expand_dims(pred_score, axis=1),
            np.expand_dims(pred_class, axis=1)
        ], axis=1)
    
        df = pd.DataFrame(image_detections,
                      columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])
        return df


    #     pred_t = [scores.index(x) for x in pred_score if x>confidence]
    #     for j in range(idxs[0].shape[0]):
    #         bbox = transformed_anchors[idxs[0][j], :]

    #         x1 = int(bbox[0] / scale)
    #         y1 = int(bbox[1] / scale)
    #         x2 = int(bbox[2] / scale)
    #         y2 = int(bbox[3] / scale)
    #         label_name = labels[int(classification[idxs[0][j]])]
    #         # print(bbox, classification.shape)
    #         score = scores[j]

    # pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    # pred_boxes = pred[0]['boxes'].detach().cpu().numpy()
    # # pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    # pred_score = list(pred[0]['scores'].detach().cpu().numpy())

    # pred_t = [pred_score.index(x) for x in pred_score if x>confidence]
    # if len(pred_t) == 0:
    #     return None
    # pred_t = pred_t[-1]
    # pred_boxes = pred_boxes[:pred_t+1]
    # pred_class = pred_class[:pred_t+1]
    # pred_score = pred_score[:pred_t+1]


    # # x1_lst, y1_lst, x2_lst, y2_lst = [np.expand_dims(pred[0]['boxes'].detach().cpu().numpy()[:,i], axis=1) for i in range(pred[0]['boxes'].detach().cpu().numpy().shape[1])] 
    # x1_lst, y1_lst, x2_lst, y2_lst = [np.expand_dims(pred_boxes[:,i], axis=1) for i in range(pred_boxes.shape[1])] 
    # # import ipdb; ipdb.set_trace()
    # image_detections = np.concatenate([
    #     x1_lst, y1_lst, x2_lst, y2_lst,
    #     np.expand_dims(pred_score, axis=1),
    #     np.expand_dims(pred_class, axis=1)
    # ], axis=1)
    
    # df = pd.DataFrame(image_detections,
    #                   columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])
    # # import ipdb; ipdb.set_trace()
    # return df        #pred_boxes, pred_class, pred_score
   

def process_one_image(img_path, shfPath, evaluate, model_path, class_list):

    
    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key
    #init model
    model = torch.load(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()



    big_image_name = img_path.split("/")[-1].split("_")[-3]
    idx_img = img_path.split("/")[-1][:-4][-1]
    dataset = rasterio.open(img_path)
    with rasterio.open(img_path, 'r') as ds:
        arr = ds.read()  # read all raster values
    # read shapefile
    polygons = gpd.read_file(shfPath)
    print(dataset.height,dataset.width,dataset.transform,dataset.crs)

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
        boxes = get_prediction(model, crop, labels, confidence=0.7)
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
    with tf.compat.v1.Session() as sess:
        print(
            "{} predictions in overlapping windows, applying non-max supression". \
            format(predicted_boxes.shape[0]))
        new_boxes, new_scores, new_labels = non_max_suppression(
            sess,
            predicted_boxes[["xmin", "ymin", "xmax", "ymax"]].values,
            predicted_boxes.score.values,
            predicted_boxes.label.values,
            max_output_size=predicted_boxes.shape[0],
            iou_threshold=0.1)
        # import ipdb; ipdb.set_trace()

        # Recreate box dataframe
        image_detections = np.concatenate([
            new_boxes,
            np.expand_dims(new_scores, axis=1),
            np.expand_dims(new_labels, axis=1)
        ], axis=1)

    df = pd.DataFrame(
        image_detections,
        columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])
    # import ipdb; ipdb.set_trace()
    # df.label = df.label.str.decode("utf-8")
    if evaluate:
        # calcualte precision, recall for ground truths and predict bbox
        allBoundingBoxes = get_ground_truths_bbox(img_path, shfPath)
        for ele in tqdm(df.values.tolist()):
            bb = BoundingBox(
                img_path.split("/")[-1][:-4],
                ele[5].decode(), # label
                ele[0], # x_min
                ele[1], # y_min
                ele[2], # x_max
                ele[3], # y_max
                CoordinatesType.Absolute, None,
                BBType.Detected,
                float(ele[4]), # confidence
                format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)
    
        evaluator = Evaluator()
        metricsPerClass = evaluator.GetPascalVOCMetrics(
            allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=0.5,  # IOU threshold
            method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
        print("Precision values per class:\n")
        # Loop through classes to obtain their metrics
        for mc in metricsPerClass:
            # Get metric values per each class
            c = mc['class']
            precision = mc['precision']
            recall = mc['recall']
            average_precision = mc['AP']
            total_TP = mc['total TP']
            total_FP = mc['total FP']
            total_groundTruths = mc['total positives']
            # Print AP per class
            pre = total_TP/(total_TP + total_FP)
            rec = total_TP/total_groundTruths
            print("Precision: {}: {}".format(c, total_TP/(total_TP + total_FP)))
            print('Recall: {}: {}'.format(c, total_TP/total_groundTruths))
            print("F1-score: {}: {}".format((2*pre*rec)/(pre+rec)))

    df['geometry'] = df.apply(lambda x: convert_xy_tif(x, dataset), axis=1)
    df_res = gpd.GeoDataFrame(df[["xmin", "ymin", "xmax", "ymax","geometry"]], geometry='geometry')
    # import ipdb; ipdb.set_trace()
    df_res.to_file('./demo/output_pred/with_shapely_pred.shp', driver='ESRI Shapefile')
    print("-----------Done--------------")
    # import ipdb; ipdb.set_trace()



if __name__ == '__main__':
    if not os.path.exists('./result'):
        os.makedirs('./result')

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', default = "./dataset/testfol1", help='Path to directory containing images')
    parser.add_argument('--model_path', default="./csv_retinanet_10.pt", help='Path to model')
    parser.add_argument('--class_list', default = "./dataset_not_aug/classes.csv", help='Path to CSV file listing class names (see README)')
    parser.add_argument("--image_path", type=str, help="Path to tif image")
    parser.add_argument("--shapefile_path", type=str, help="Path to shapefile coresponding the input image")
    parser.add_argument("--evaluate", type=bool, help="evaluate the testing image of your model with the shapefile path")
    parser = parser.parse_args()

    # detect_image(parser.image_dir, parser.model_path, parser.class_list)
    process_one_image(parser.image_path, parser.shapefile_path, parser.evaluate, parser.model_path, parser.class_list)
