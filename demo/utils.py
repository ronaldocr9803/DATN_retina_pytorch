import geopandas as gpd
import rasterio
import slidingwindow
import tensorflow as tf
import torch
from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import *
# from model import fasterrcnn_resnet101_fpn
from shapely.geometry import Point, Polygon, box, mapping, shape
from tqdm import tqdm


# def init_model():
#     # if model_name == "resnet50":
#     #     model = build_model_resnet50fpn(3)
#     # elif model_name == "resnet101":
#     # if torch.cuda.is_available():  
#     #     dev = "cuda:0" 
#     # else:  
#     #     dev = "cpu"  
#     # model = fasterrcnn_resnet101_fpn()
#     model = fasterrcnn_resnet101_fpn()
#     # device = torch.device("cuda:0") # device for gpu inference
#     device = torch.device("cuda:0") # device for cpu inference
#     checkpoint = torch.load("./checkpoint/chkpoint_colab_14_19-11.pt", map_location= "cuda:0") #read from last checkpoint
#     # checkpoint = torch.load("./checkpoint/chkpoint_colab_14.pt",  map_location= "cuda:0")
#     model.load_state_dict(checkpoint['state_dict'])
#     model.eval() #evaluation mode
#     return model 

def compute_windows(numpy_image, patch_size, patch_overlap):
    """Create a sliding window object from a raster tile.
    Args:
        numpy_image (array): Raster object as numpy array to cut into crops
    Returns:
        windows (list): a sliding windows object
    """

    if patch_overlap > 1:
        raise ValueError("Patch overlap {} must be between 0 - 1".format(patch_overlap))

    # Generate overlapping sliding windows
    windows = slidingwindow.generate(numpy_image,
                                     slidingwindow.DimOrder.HeightWidthChannel,
                                     patch_size, patch_overlap)

    return (windows)


def predict_tile(model,
                raster_path=None,
                numpy_image=None,
                patch_size=400,
                patch_overlap=0.05,
                iou_threshold=0.15,
                return_plot=False):
        """For images too large to input into the model, predict_tile cuts the
        image into overlapping windows, predicts trees on each window and
        reassambles into a single array.
        Args:
            raster_path: Path to image on disk
            numpy_image (array): Numpy image array in BGR channel order
                following openCV convention
            patch_size: patch size default400,
            patch_overlap: patch overlap default 0.15,
            iou_threshold: Minimum iou overlap among predictions between
                windows to be suppressed. Defaults to 0.5.
                Lower values suppress more boxes at edges.
            return_plot: Should the image be returned with the predictions drawn?
        Returns:
            boxes (array): if return_plot, an image.
                Otherwise a numpy array of predicted bounding boxes, scores and labels
        """

        if numpy_image is not None:
            pass
        else:
            # Load raster as image
            raster = Image.open(raster_path)
            numpy_image = np.array(raster)

        # Compute sliding window index
        windows = preprocess.compute_windows(numpy_image, patch_size, patch_overlap)

        # Save images to tmpdir
        predicted_boxes = []

        for index, window in enumerate(tqdm(windows)):
            # Crop window and predict
            crop = numpy_image[windows[index].indices()]

            # Crop is RGB channel order, change to BGR
            crop = crop[..., ::-1]
            boxes = self.predict_image(numpy_image=crop,
                                       return_plot=False,
                                       score_threshold=self.config["score_threshold"])

            # transform coordinates to original system
            xmin, ymin, xmax, ymax = windows[index].getRect()
            boxes.xmin = boxes.xmin + xmin
            boxes.xmax = boxes.xmax + xmin
            boxes.ymin = boxes.ymin + ymin
            boxes.ymax = boxes.ymax + ymin

            predicted_boxes.append(boxes)

        predicted_boxes = pd.concat(predicted_boxes)

        # Non-max supression for overlapping boxes among window
        if patch_overlap == 0:
            mosaic_df = predicted_boxes
        else:
            with tf.Session() as sess:
                print(
                    "{} predictions in overlapping windows, applying non-max supression".
                    format(predicted_boxes.shape[0]))
                new_boxes, new_scores, new_labels = predict.non_max_suppression(
                    sess,
                    predicted_boxes[["xmin", "ymin", "xmax", "ymax"]].values,
                    predicted_boxes.score.values,
                    predicted_boxes.label.values,
                    max_output_size=predicted_boxes.shape[0],
                    iou_threshold=iou_threshold)

                # Recreate box dataframe
                image_detections = np.concatenate([
                    new_boxes,
                    np.expand_dims(new_scores, axis=1),
                    np.expand_dims(new_labels, axis=1)
                ],
                                                  axis=1)

                mosaic_df = pd.DataFrame(
                    image_detections,
                    columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])
                mosaic_df.label = mosaic_df.label.str.decode("utf-8")

                print("{} predictions kept after non-max suppression".format(
                    mosaic_df.shape[0]))

        if return_plot:
            # Draw predictions
            for box in mosaic_df[["xmin", "ymin", "xmax", "ymax"]].values:
                draw_box(numpy_image, box, [0, 0, 255])

            # Mantain consistancy with predict_image
            return numpy_image
        else:
            return mosaic_df

def non_max_suppression(sess,
                        boxes,
                        scores,
                        labels,
                        max_output_size=200,
                        iou_threshold=0.15):
    """Provide a tensorflow session and get non-maximum suppression.
    Args:
        sess: a tensorflow session
        boxes: boxes
        scores: scores
        labels: labels
        max_output_size: passed to tf.image.non_max_suppression
        iou_threshold: passed to tf.image.non_max_suppression
    Returns:
    """
    non_max_idxs = tf.image.non_max_suppression(boxes,
                                                scores,
                                                max_output_size=max_output_size,
                                                iou_threshold=iou_threshold)
    new_boxes = tf.cast(tf.gather(boxes, non_max_idxs), tf.int32)
    new_scores = tf.gather(scores, non_max_idxs)
    new_labels = tf.gather(labels, non_max_idxs)
    return sess.run([new_boxes, new_scores, new_labels])
    
def convert_xy_tif(x, dataset):
    minx_lst, maxy_lst = dataset.xy(x.ymin, x.xmin)
    maxx_lst, miny_lst = dataset.xy(x.ymax, x.xmax)
    return box(minx_lst, miny_lst, maxx_lst, maxy_lst)

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
    polygons = gpd.read_file(shpfile_path)
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
                format=BBFormat.XYX2Y2)
        allBoundingBoxes.addBoundingBox(bb)
    return allBoundingBoxes

if __name__ == "__main__":
    # numpy_image = np
    windows = compute_windows(numpy_image, patch_size, patch_overlap)
