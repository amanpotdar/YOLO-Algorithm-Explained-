# YOLO-Algorithm-Explained-
A summary for the YOLO  algorithm


YOLO (You Only Look Once)


YOLO Object Detection with OpenCV


Machine Learning and Artificial Intelligence are certainly buzzwords nowadays. If you have been keeping up with the advancements in the area of object detection, you might have got used to hearing this word YOLO. It has certainly become a buzzword. So what exactly is “YOLO”? YOLO (you only Look once) is a method or a way to do object detection. It is the algorithm or we can say a strategy behind how the code is going to detect objects present in the image. The official implementation of “YOLO” is available through DarkNet.

Code:-
import cv2

import argparse

import numpy as np

ap=argparse.ArgumentParser()

ap.add_argument(‘-i’,’ — image’,requimagered=True, help=’specify the path to input image’)

ap.add_argument(‘-c’,’ — config’,required=True,help=’specify the path to YOLO config file’)

ap.add_argument(‘-w’,’ — weights’,required=True,help=’path to the YOLO pre-trained weights’)

ap.add_argument(‘-cl’,’ — classes’,required=True,help=’path to text file containing class names’)

args=ap.parse_args()


As you can see in the figure that the YOLO detection technique is able to differentiate between separate horses
Note:- Make sure pip is linked to Python 3.x ( pip -V will show this info)If needed use pip3. Use sudo apt-get install python3-pip to get pip3 if not already installed.OpenCV-PythonYou need to compile OpenCV from the source from the master branch on Github to get the Python bindings.

The particular dataset is trained on COCO dataset from Microsoft. It is capable of detecting many common objects. The input image can be of the user’s choice.

Run the script by typing$ python yolo_opencv.py — image dog.jpg — config yolov3.cfg — weights yolov3.weights — classes yolov3.txt

Generally in CNN networks we only use a single output layer at the end. In YOLO v3 architecture we use multiple layers giving out predictions. The output layer is not connected to the next layer. We generally use draw_bounding_box() function draws a rectangle over the given predicted region and write classes name over the box.

The major concept of YOLO is to build a CNN network to predict a (7, 7, 30) tensor. It uses a CNN network to reduce the spatial dimension to 7×7 with 1024 output channels at each location. YOLO performs a linear regression using two fully connected layers to make 7×7×2 boundary box predictions (the middle picture below). To make a final prediction, we keep those with high box confidence scores (greater than 0.25) as our final predictions (the right picture).


The class confidence score for each prediction box is computed as:


It measures the confidence in intine classification and the localization(where an object is located).

We may mix up those scoring and probability terms easily. Here are the mathematical definitions for your future reference.



Network Design
Advantages of YOLO algorithm:-
Better Speed (can go up to 45 frames per second)
Its Network understands generalized object representation.
Open Source
If you compare YOLO algorithm to fast RCNN or other algorithms, we can see that other algorithms which perform detection on various region proposals and thus end up performing prediction multiple times for various images. We can say that YOLO’s architecture is more like the Fast Convolutional Neural network or the FCNN.. Such architecture splits the input image in mxm grid and generates 2 bounding boxes and class probabilities for those bounding boxes.


YOLO also reasons globally about the image when making predictions.

Yolo sees the entire image when training and also during test time. Therefore it implicitly encodes contextual information about classes as well as their appearance.

YOLO usually outperforms top detection methods like DPM and R-CNN and other algorithms by a wide margin.

Accuracy
Here are there accuracy improvements after applying the techniques discussed so far:


Source
Accuracy comparison for different detectors:


Source
YOLO’s Limitations:-
YOLO algorithm can struggle with small objects that appear in groups. As the model learns to predict bounding boxes from data, it can struggle to generalize objects in new ratios or configurations.

Certainly, the YOLO algorithm will be of immense use to mankind in the near future.

References:-
jonathan_hui/real-time-object-detection-with-yolo
The Original Paper on YOLO
https://pjreddie.com/yolo/ (Image Credits)
DarkNet Implementation
www.arunponnusamy.com

