# ORIGINAL SORT CODE
https://github.com/abewley/sort

# ORIGINAL MediaPipe Code
https://github.com/google/automl-video-ondevice


# AutoML Video Edge Library

AutoML Video Edge Library is an open source engine used for inferencing models
trained using AutoML Video. It supports running Tensorflow, TF-TRT, TFLite, and
EdgeTPU-optimized TFLite models.

Originally Developed For:

* [Linux Desktop](#for-linux-desktop)
* [Coral Device](#for-coral-device)

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
`cd trackerSortAndMediaPipe/detectionTracker/automl-video-ondevice/examples/`

## Running an Example

For TFLite:  
`python3 video_file_demo.py --model=data/traffic_model.tflite`

For Tensorflow:  
`python3 video_file_demo.py --model=data/traffic_model.pb`

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
`cd trackerSortAndMediaPipe/detectionTracker/automl-video-ondevice/examples/`

## Running an Example with sort tracker

`python3 examples/video_file_demo.py --model=data/traffic_model_edgetpu.tflite --use_tracker sort`

## Running an Example with MediaPipe tracker

`python3 examples/video_file_demo.py --model=data/traffic_model_edgetpu.tflite --use_tracker BASIC`

