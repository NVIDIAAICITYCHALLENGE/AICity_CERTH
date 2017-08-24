# CERTH - ITI
# Intelligent traffic city management from surveillance systems
### Introduction

Vehicle detection in this work is deployed by following two approaches: 

* CERTH-RCNN 
* DeepHOG.

An ensemble framework is also investigated as a complementary approach so as to study the impact that the fusion of the the two. Object localization in both cases is performed by using the Region Proposal Network (RPN) that CERTHCNN uses.

CERTH-RCNN uses a modification of the original Faster R-CNN. We used the Faster-RCNN-Resnet101 model that is pretrained on the COCO dataset and tuned it on NVIDIA AI city dataset, so as to be able to detect vehicles in video frames.

For DeepHOG vehicle detection, we used Histograms of Oriented Gradients(HOG) as a local appearance features to represent pedestrians and vehicle objects and encode them into a Fisher vector. HOG features were computed on the already predicted boxes of the RCNN.

Here is an example from the **_aic480_** dataset:

![Detection 480](https://github.com/NVIDIAAICITYCHALLENGE/AICity_CERTH/blob/master/samples/Picture2.png)

And another one from the **_aic540_** dataset:

![Detection 540](https://github.com/NVIDIAAICITYCHALLENGE/AICity_CERTH/blob/master/samples/Picture1.png)

We also perform vehicle tracking on sequences taken from traffic surveillance cameras.

![Tracking](https://github.com/NVIDIAAICITYCHALLENGE/AICity_CERTH/blob/master/samples/Picture3.png)

### How to use the code

There are

