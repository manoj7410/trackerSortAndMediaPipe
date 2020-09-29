# ORIGINAL SORT CODE
https://github.com/abewley/sort

# ORIGINAL MediaPipe Code
https://github.com/google/automl-video-ondevice


# Object Detection along with Object Tracking 
The code has been referenced from AutoML video Edge Library of Google. It provides the functionality of
Object detection along with Sort/CamShift/MediaPipe trackers. Usage are given below:

# For Coral Device
-------------------

## Prerequisites

Make sure you've setup your coral device:
https://coral.ai/docs/setup

Install the TFLite runtime on your device:
https://www.tensorflow.org/lite/guide/python

```
sudo apt-get update
sudo apt-get install git
sudo apt-get install python3-opencv
pip3 install numpy
```

## Get the Code

`git clone https://github.com/manoj7410/trackerSortAndMediaPipe.git`

After that is done downloading, move into the directory.  
`cd trackerSortAndMediaPipe/detectionTracker/examples/`

## Running an Example with sort tracker

`python3 video_file_demo.py --use_tracker sort`

## OR to run the demo with Camera
`python3 camera_demo.py --use_tracker sort`


## Running an Example with MediaPipe tracker

`python3 video_file_demo.py --use_tracker mediapipe`


## Running an Example with CamShift tracker

`python3 video_file_demo.py --use_tracker camshift`



# For Linux Desktop
-------------------

If you are looking to do inferencing with no additional hardware, using only CPU
then you may use the vanilla Tensorflow (.pb) and TFLite (.tflite) models.

## Prerequisites

```
sudo apt-get update
sudo apt-get install python3.7
sudo apt-get install python3-pip
pip3 install opencv-contrib-python --user
pip3 install numpy
```

Note: opencv-contrib-python is only necessary for the examples, but can be
excluded if only the library is being used.

If you plan on running TFLite models on the desktop, install the TFLite
interpreter: https://www.tensorflow.org/lite/guide/python

If you plan on running Tensorflow models on desktop:  
`pip3 install tensorflow==1.14`

## Get the Code

`git clone https://github.com/manoj7410/trackerSortAndMediaPipe.git`

After that is done downloading, move into the directory.  
`cd trackerSortAndMediaPipe/detectionTracker/examples/`

## Running an Example

For TFLite:  
`python3 video_file_demo.py`

For Tensorflow:  
`python3 video_file_demo.py`

