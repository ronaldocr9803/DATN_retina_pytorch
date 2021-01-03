
import glob
import logging
import os
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm
import pandas as pd
import argparse

random.seed(42)
img_path_lst = glob.glob(os.path.join(".","dataset_not_aug","images","*"))
num_train_sample = int(round(len(img_path_lst) * 0.8))
img_train_path_lst = random.sample(img_path_lst, num_train_sample) 
img_val_path_lst = [x for x in img_path_lst if x not in img_train_path_lst]


def convert_annotations_to_df(annotation_dir, image_dir, image_set="train"):
    xml_list = []
    for image_path in tqdm(image_dir):
        # filename = 
        xml_file = annotation_dir + image_path.split("/")[-1][:-4] + ".xml"
    # for xml_file in glob.glob(annotation_dir + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            bbx = member.find("bndbox")
            xmin = int(bbx.find("xmin").text)
            ymin = int(bbx.find("ymin").text)
            xmax = int(bbx.find("xmax").text)
            ymax = int(bbx.find("ymax").text)
            label = member.find("name").text

            value = (
                image_path,
                xmin,
                ymin,
                xmax,
                ymax,
                "tree"
            )
            xml_list.append(value)

    column_name = [
        "filename",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "class"
    ]

    xml_df = pd.DataFrame(xml_list, columns=column_name)
    # xml_df["filename"] = [
    #     os.path.join(image_dir, xml_df["filename"][i]) for i in range(len(xml_df))
    # ]

    # if image_set == "train":
    #     # label encoder encodes the labels from 0
    #     # we need to add +1 so that labels are encode from 1 as our
    #     # model reserves 0 for background class.
    #     xml_df["labels"] = encoder.fit_transform(xml_df["class"]) + 1
    # elif image_set == "val" or image_set == "test":
    #     xml_df["labels"] = encoder.transform(xml_df["class"]) + 1
    return xml_df

def get_pascal(annot_dir, image_dir, image_set="train", **kwargs):
    n = f"./dataset_not_aug/pascal_{image_set}.csv"
    df = convert_annotations_to_df(annot_dir, image_dir, image_set)
    df.to_csv(n, index=False)
    print("Done {}".format(n))
    # return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert xml file to csv file for training model')

    parser.add_argument('--img_folder', default='csv', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--mode', help='Path to COCO directory')

    get_pascal("./dataset_not_aug/annotations/", img_train_path_lst, image_set = "train")

    get_pascal("./dataset_not_aug/annotations/", img_val_path_lst, image_set = "val")
