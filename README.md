# Edge TPU simple camera examples

This repo contains a collection of examples that use camera streams
together with the [TensorFlow Lite API](https://tensorflow.org/lite) with a
Coral device such as the
[USB Accelerator](https://coral.withgoogle.com/products/accelerator) or
[Dev Board](https://coral.withgoogle.com/products/dev-board) and aligns an Object tracker 
with the detected objects. 

## Installation

1.  First, be sure you have completed the [setup instructions for your Coral
    device](https://coral.ai/docs/setup/). If it's been a while, repeat to be sure
    you have the latest software.

    Importantly, you should have the latest TensorFlow Lite runtime installed
    (as per the [Python quickstart](
    https://www.tensorflow.org/lite/guide/python)).

2.  Clone this Git repo onto your computer:

    ```
    mkdir google-coral && cd google-coral

    git clone https://github.com/manoj7410/trackerSortAndMediaPipe.git
    ```

3.  Download the models:

    ```
    cd gstreamerWithSortTracker

    sh download_models.sh
    ```

    These canned models will be downloaded and extracted to a new folder
    ```all_models```.


Further requirements may be needed by the different camera libraries, check the
README file for the respective subfolder.

## Contents

  * __Gstreamer__ Python examples using gstreamer to obtain camera streem. These
    examples work on Linux using a webcam, Raspberry Pi with
    the Raspicam and on the Coral DevBoard using the Coral camera. For the
    former two you will also need a Coral USB Accelerator to run the models.
  
## Canned models

For the demos in this repository you can change the model and the labels
file by using the flags flags ```--model``` and
```--labels```. Be sure to use the models labeled _edgetpu, as those are
compiled for the accelerator -  otherwise the model will run on the CPU and
be much slower.


For detection you need to select one of the SSD detection models
and its corresponding labels file:

```
mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite, coco_labels.txt
mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite, coco_labels.txt
mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite, coco_labels.txt
```


