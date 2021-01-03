# pytorch-retinanet

Vietnamese below

How to run project 

1. Download dataset: shorturl.at/anuv2
2. unzip dataset above to dataset_not_aug folder
3.

![img3](https://github.com/yhenon/pytorch-retinanet/blob/master/images/3.jpg)
![img5](https://github.com/yhenon/pytorch-retinanet/blob/master/images/5.jpg)

Pytorch  implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

This implementation is primarily designed to be easy to read and simple to modify.

## Results
Currently, this repo achieves 33.5% mAP at 600px resolution with a Resnet-50 backbone. The published result is 34.0% mAP. The difference is likely due to the use of Adam optimizer instead of SGD with weight decay.

## Installation

1) Clone this repo

2) Install the required packages:

```
apt-get install tk-dev python-tk
```

3) Install the python packages:
	
```
pip install pandas
pip install pycocotools
pip install opencv-python
pip install requests

```

## Training

The network can be trained using the `train.py` script. 

```
sh train.sh
```

## Demo

This part will run an entire satelite image in (have size 3874 x 3100)

```
sh demo.sh
```
Some config to run demo:

    --image_path: path to save your satelite image (.tif format)    
    --shapefile_path: path to save your shpfile for evaluate with model
    --model_path: path to load your model (.pt format save in checkpoint folder)
    --evaluate: True if you have shpfile path to evaluate result with your model, False if otherwise. In case False, --shapefile_path is optional


## Model

The retinanet model uses a resnet backbone. You can set the depth of the resnet model using the --depth argument. Depth must be one of 18, 34, 50, 101 or 152. Note that deeper models are more accurate but are slower and use more memory.
