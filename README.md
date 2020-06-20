# YOLO based object detection

This repository contains the files necessary to run YOLO based object detection right through your webcam. The weights and
config files for the yolov3 were obtained from [here](https://pjreddie.com/darknet/yolo/)

# Steps:

## Reading the image: 

There are two possibilities to get the input image, either through your webcam or from local disk.

## Constructing the model:

To construct the model from the .cfg file, the OpenCV dnn module was used. The noteable functions are

* readNet: constructs the model from .cfg and weight of the model
* blobFromImage: carries out mean normalization and scaling over image
* setInput: feeds the blob to the network 
* forward: Performs the forward pass on network, It is important to specify the name of Output Layer or else the output from 
last layer will be given. We dont want the outputs of intermediate layers to be ignored.

## Running the inferernce:

For each detection, from each output layer we filter out the predictions with low confidence. For remaining candidates we store the confidence, 
class_id and bounding box coordinates.

## Non-maximal suppression:

There will be still some duplicate detections (boxes) even after filtering detections with low confidence. So NMS is applied here to get only 1 box per detection.
In the end you can get output something like this.

![alt text](./images/detections.PNG?raw=true)