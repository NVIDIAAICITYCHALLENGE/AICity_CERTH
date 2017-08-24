# CERTH - ITI
# Intelligent traffic city management from surveillance systems
### Introduction

Vehicle detection in this work is deployed by following two approaches: 

* **CERTH-RCNN** 
* **DeepHOG**

An **ensemble** framework is also investigated as a complementary approach so as to study the impact that the fusion of the the two. Object localization in both cases is performed by using the Region Proposal Network (RPN) that **CERTH-RCNN** uses.

**CERTH-RCNN** uses a modification of the original Faster R-CNN. We used the *Faster-RCNN-Resnet101* model that is pretrained on the *COCO* dataset and tuned it on *NVIDIA AI city* dataset, so as to be able to detect vehicles in video frames.

For **DeepHOG** vehicle detection, we used *Histograms of Oriented Gradients(HOG)* as a local appearance features to represent pedestrians and vehicle objects and encode them into a Fisher vector. *HOG* features were computed on the already predicted boxes of the RCNN.

Here is an example from the **_aic480_** dataset:

![Detection 480](https://github.com/NVIDIAAICITYCHALLENGE/AICity_CERTH/blob/master/samples/Picture2.png)

And another one from the **_aic540_** dataset:

![Detection 540](https://github.com/NVIDIAAICITYCHALLENGE/AICity_CERTH/blob/master/samples/Picture1.png)

We also perform **vehicle tracking** on sequences taken from traffic surveillance cameras.

![Tracking](https://github.com/NVIDIAAICITYCHALLENGE/AICity_CERTH/blob/master/samples/Picture3.png)

### How to use the code

There are three folders, one for each inplementation and there are README files in each one to help you get started.
You should start by setting up the required environment.

The code is written for Python 2.7 and the project makes heavy use of the TensorFlow's Object Detection API that was recently released by Google. You can find it here:  https://github.com/tensorflow/models/tree/master/object_detection.

Once you have those installed read the instructions inside the CERTH_RCNN folder to get started on predicted bounding boxes for your images using our provided models. You can even train your own model by following the numerous tutorials included in the Object Detection API repo.

After you have succesfully extracted vehicle detections from the RCNN you can also try our DeepHOG and Ensemble models and the CERTH_KCF vehicle tracking script.

### Contact info:
Please send me an e-mail with your comments and/or questions in `giannakeris@iti.gr`

### Disclaimer:
This piece of software is still a work in progress and may contain project-specific pieces of code that may not work well on your developed models.
