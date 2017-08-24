#------------------------------------------------------------------------------------------------
#Developed by Information Technologies Institute of the Center for Research and Technology Hellas
#This code was created by the team to participate in the NVIDIA AI City Challenge / Aug 2017
#Contact: giannakeris@iti.gr
#------------------------------------------------------------------------------------------------

#This script performs vehicle detection in a set of images 
#and exports the labels, coordinates and classification 
#score for each bounding box in a .txt file (one per image).
#It requires as input a frozen inference graph file exported 
#from a tensorflow model and the appropriate label map file.

#Usage: Please edit the required paths 
#       in the section immediatly after the imports bellow.

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from tqdm import tqdm
import pandas
from collections import defaultdict
from io import StringIO
from PIL import Image
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

############ EDIT HERE ############

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/edit/this/path/graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/edit/this/path/label_map.pbtxt'

# Path to folder that test images are stored.
PATH_TO_TEST_IMAGES_DIR = '/edit/this/path/images/'

PATH_TO_OUTPUT_FOLDER = '/edit/this/path/output/'

###################################

NUM_CLASSES = 14

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
      

TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, each) for each in os.listdir(PATH_TO_TEST_IMAGES_DIR) if each.endswith('.jpeg')]

def box_transf(boxes, w, h):
    res = []
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        ymin = ymin * h
        xmin = xmin * w
        ymax = ymax * h
        xmax = xmax * w
        res += [[xmin, ymin, xmax, ymax]]
    return res

    
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for image_path in tqdm(TEST_IMAGE_PATHS):
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            
            h, w, d = image_np.shape
            
            image_boxes = np.around(np.array(box_transf(boxes[0], w, h)))
            image_scores = scores[0]
            image_classes = np.array([str(category_index[classes[0][int(c)]]['name']) for c in classes[0]])
            result1 = np.transpose(np.atleast_2d(image_classes))
            result2 = np.concatenate([image_boxes, np.transpose(np.atleast_2d(image_scores))], axis=1)
            result1 = pandas.DataFrame(result1)
            result2 = pandas.DataFrame(result2)
            result = pandas.concat([result1, result2], axis=1)
            result.columns = ['class', 'xmin', 'ymin', 'xmax', 'ymax', 'score']
            #threshold the detected boxes by score
            result_th = result[result['score']>=threshold]
            #export the txt file in the folder of your choice
            result_th.to_csv(PATH_TO_OUTPUT_FOLDER+os.path.splitext(os.path.split(image_path)[1])[0]+'.txt', header=False, index=False, sep=' ')
