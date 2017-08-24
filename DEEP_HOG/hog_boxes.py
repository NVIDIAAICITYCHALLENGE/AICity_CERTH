from __future__ import division
#------------------------------------------------------------------------------------------------
#Developed by Information Technologies Institute of the Center for Research and Technology Hellas
#This code was created by the team to participate in the NVIDIA AI City Challenge / Aug 2017
#Contact: giannakeris@iti.gr
#------------------------------------------------------------------------------------------------

#This script computes HOG feature vectors on the 
#detected bounding boxes of the RCNN extracted on a previous step.
#Requires the RCNN's predictions txt output files (one per image) where each line has the format: 
#[<class> <xmin> <ymin> <xmax> <ymax> <score>] (without the [,],<,> symbols).
#The output is a txt file per image where each line is the HOG feature vector 
#representation of the corresponding box.
#This representation can be later used in order to compute a Fisher vector 
#and classify the box with either the DeepHOG or the Ensemble model.

#Usage: Please edit the required paths 
#       in the section immediatly after the imports bellow.

import cv2
import numpy
import pandas
import os
from tqdm import *

########################################## EDIT HERE ######################################################
IMAGE_FOLDER = '/enter/images/path/'            #the folder where images are stored
TEST_FOLDER = '/enter/detected/boxes/path/'     #the folder where txt files with detected boxes are stored
OUTPUT_PATH = '/enter/output/folder/'           #desired output path
###########################################################################################################

nbins = 9
derivAperture = 1
winSigma = 4.0
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 1
nlevels = 64
signedGradient = 0

imageNames = os.listdir(IMAGE_FOLDER)
annotationNames = os.listdir(TEST_FOLDER)

for name in tqdm(imageNames):
    image = cv2.imread(IMAGE_FOLDER+name)

    names = ('xmin', 'ymin', 'xmax', 'ymax')
    labels = pandas.read_csv(annotationNames+os.path.splitext(name)[0]+'.txt', header=None, sep=' ', names = names, usecols=[1,2,3,4])
    num_of_boxes = labels.shape[0]

    box_list = labels.values.tolist()

    feature_array = numpy.zeros(shape=(num_of_boxes,4*9*4+1), dtype=numpy.float)

    for idx, box in enumerate(box_list):
        xmin,ymin,xmax,ymax = box
        xmin=int(xmin)
        ymin=int(ymin)
        xmax=int(xmax)
        ymax=int(ymax)
        h = ymax-ymin
        w = xmax-xmin
        h0 = h
        w0 = w
        if(h%3 != 0):
            h = 3*(h//3)
            ymax = ymin + h
        if(w%3 != 0):
            w = 3*(w//3)
            xmax = xmin + w
        winSize = (w,h)
        blockSize = (2*(w//3), 2*(h//3))
        blockStride = (w//3, h//3)
        cellSize = (w//3, h//3)
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradient)
        feature_array[idx] = numpy.concatenate([numpy.transpose(hog.compute(image[ymin:ymax, xmin:xmax])), numpy.atleast_2d(w0/h0)], axis=1)
        
    feature_df = pandas.DataFrame(feature_array)
    feature_df.to_csv(OUTPUT_PATH+os.path.splitext(name)[0]+'.hog', header=False, index=False)