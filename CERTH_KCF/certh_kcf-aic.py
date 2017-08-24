#------------------------------------------------------------------------------------------------
#Developed by Information Technologies Institute of the Center for Research and Technology Hellas
#Contact: giannakeris@iti.gr
#------------------------------------------------------------------------------------------------


print('This is the CERTH vehicle tracker v0.2 for Nvidia Smart Cities Challenge')
print('Importing libraries...')
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
import imageio
import time
import KCF
import cv2
import sys
import argparse
from utils import label_map_util
from utils import visualization_utils as vis_util
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Vehicle detection and tracking, CERTH_KCF')
parser.add_argument('-sf', '--sequence_folder',
            action="store", dest="sf",
            help="folder where the sequence is stored", type=str, required=True)
            
parser.add_argument('-sn', '--sequence_name',
            action="store", dest="sn",
            help="name of the sequence", type=str, required=True)
            
parser.add_argument('-m', '--model',
            action="store", dest="model",
            help="path to frozen model file", type=str, required=True)
            
parser.add_argument('-l', '--label_file',
            action="store", dest="lbl",
            help="the label map file", type=str, required=True)
            
parser.add_argument('-o', '--output_file',
            action="store", dest="op",
            help="absolute path of output, extention must be .mp4", type=str, required=True)
            
parser.add_argument('-fps', '--video_fps',
            action="store", dest="fps",
            help="choose fps of output video", type=int, default=30)
            
parser.add_argument('-nf', '--number_frames',
            action="store", dest="nf",
            help="choose how many number of frames to process", type=int, default=1000)

parser.add_argument('-der', '--detection_rate',
            action="store", dest="der",
            help="the algorithm will perform detection every this number of frames", type=int, default=3)

parser.add_argument('-dlo', '--dlost_thres',
            action="store", dest="dlo",
            help="deletes track if detection is lost for this number of consecutive frames", type=int, default=6) 
			
parser.add_argument('-ret', '--reset_thres',
            action="store", dest="ret",
            help="clear the screen from tracked boxes every this number of frames", type=int, default=15)
			
parser.add_argument('-cod', '--confidence_drop',
            action="store", dest="cod",
            help="bellow this confidence score boxes will be dropped", type=float, default=0.5) 
			
parser.add_argument('-sth', '--size_thres',
            action="store", dest="sth",
            help="above this size (percentage of screen) boxes will be dropped", type=float, default=0.33) 
			
parser.add_argument('-si', '--save_iou',
            action="store", dest="si",
            help="if the algorithm finds an iou between current box and tracked boxes above that threshold it will not create a new id", type=float, default=0.33)

parser.add_argument('-ri', '--rectify_iou',
            action="store", dest="ri",
            help="if the max iou between the tracked box and a detection box is bellow that it will rectify the tracked box", type=float, default=0.7)
			
parser.add_argument('-mi', '--merge_iou',
            action="store", dest="mi",
            help="if the algorithm finds tracked boxes with iou score above it will delete the newest track (merging)", type=float, default=0.8)
						
args = parser.parse_args()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = args.model     #'/data/testing/480/output_inference_graph_24243.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = args.lbl       #'/data/testing/480/nvidia_label_map.pbtxt'

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
      
def get_iou(bbb1, bbb2):
    
    bb1 = [bbb1[0], bbb1[0]+bbb1[2], bbb1[1], bbb1[1]+bbb1[3]]
    bb2 = [bbb2[0], bbb2[0]+bbb2[2], bbb2[1], bbb2[1]+bbb2[3]]
    
    assert bb1[0] < bb1[1]
    assert bb1[2] < bb1[3]
    assert bb2[0] < bb2[1]
    assert bb2[2] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[2], bb2[2])
    x_right = min(bb1[1], bb2[1])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[1] - bb1[0]) * (bb1[3] - bb1[2])
    bb2_area = (bb2[1] - bb2[0]) * (bb2[3] - bb2[2])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def box_trans(box):
    box = [box[0] * height, box[1] * width, box[2] * height, box[3] * width]
    box = list(map(int, box))
    box = [box[1], box[0], abs(box[3]-box[1]), abs(box[2]-box[0])]
    return box
    
def rev_box_trans(box_list):
    res = []
    for box in box_list:
        temp = [box[1], box[0], box[3]+box[1], box[2]+box[0]]
        res = res + [temp]
    return res
    
PATH_TO_TEST_VIDEO = args.sf
VIDEO_NAME = args.sn
write_video=True
vid = imageio.get_reader(PATH_TO_TEST_VIDEO+VIDEO_NAME, 'ffmpeg')
fps = vid.get_meta_data()['fps']
nframes = vid.get_meta_data()['nframes']
w, h =  vid.get_meta_data()['size']
writer = imageio.get_writer(args.op, fps=args.fps, format='FFMPEG')
num = args.nf # first frames to read
frames_to_read = [i for i in range(num)]
all_boxes = []
all_scores = []
all_classes = []
tracks = []
tracked_boxes = []
frameList = []
start = time.time()
detection_rate = args.der #detection every this number of frames
dlost_thres = args.dlo #delete track if detection lost for this number of frames
reset_thres = args.ret #clear the screen from tracked boxes every this number of frames
confidence_drop = args.cod #bellow this confidence score boxes will be dropped
upper_size_thres = args.sth #above this size (percentage of screen) boxes will be dropped
save_iou = args.si #if you find an iou between current box and tracked boxes above that threshold don't create a new id
rectify_iou = args.ri #if the max iou between the tracked box and a detection box is bellow that rectify the tracked box
merge_iou = args.mi #if you find tracked boxes with iou score above that delete the newest track (merge)

print('Calculating boxes. Please wait...')
with tf.device('/gpu:0'):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=tf.ConfigProto(log_device_placement=True)) as sess:
            for idf in tqdm(frames_to_read):
                #frame = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                #frame_np = load_image_into_numpy_array(frame)
                get = vid.get_data(idf)
                frame_np = np.array(get)
                write_on = frame_np
                height = h
                width = w
                # Actual detection.
                if(idf%detection_rate == 0):
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    frame_np_expanded = np.expand_dims(frame_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: frame_np_expanded})
                    to_del=()
            
                    #delete unwanted boxes (too large, non car labels, etc)
                    for i in range(boxes.shape[1]):
                        if((np.abs(boxes[0][i][0]-boxes[0][i][2])*np.abs(boxes[0][i][1]-boxes[0][i][3]) >= upper_size_thres) or 
                           (scores[0][i] < confidence_drop)):
                            to_del = to_del + (i,)
                            num_detections[0] = num_detections[0] - 1
                    boxes = np.delete(boxes, to_del, axis=1)
                    scores = np.delete(scores, to_del, axis=1)
                    classes = np.delete(classes, to_del, axis=1)
                    boxes_t=[] #initiallize transformed bozes storage
            
                    # Tracking
                    if(idf > 0 and idf%reset_thres==0): #reset tracked boxes
                        for idx, track in enumerate(tracks):
                            if(tracks[idx]):
                                boundingbox = track.update(frame_np)
                                tracked_boxes[idx] = [boundingbox]
                            else:
                                tracked_boxes[idx] = []
                
                    for bid, box in enumerate(np.squeeze(boxes, axis=0)): #transform detected boxes and search through
                        box = box_trans(box)
                        boxes_t = boxes_t + [box]
                        #save new tracks
                    
                        save_flag = True
                        for sbox in sum(tracked_boxes, []): #check all tracked boxes
                            if(sbox): #if the sbox exists
                                if(get_iou(sbox, box) > save_iou): #if you find an iou above that threshold don't save a new box
                                    save_flag = False    
                        if (save_flag == True): #initiallize a new tracker (you found a box with no iou above the threshold)
                            tracker = KCF.kcftracker(False, True, True, False)  # hog, fixed_window, multiscale, lab
                            tracks = tracks + [tracker]
                            tracked_boxes = tracked_boxes + [[box]]
                            tracker.init(box, frame_np)

            
                frameList = frameList + [[]]            
                for idx, track in enumerate(tracks):
                    if (tracks[idx]):
                            boundingbox = track.update(frame_np)
                            if(idf%detection_rate == 0): #every 3rd frame
                                max_iou = 0
                                for idb, ibox in enumerate(boxes_t):
                                    iou = get_iou(ibox, boundingbox)
                                    if(iou >= max_iou):
                                        max_iou_id = idb
                                        max_iou = iou
                                if(max_iou > rectify_iou):
                                    #keep
                                    tracked_boxes[idx] = tracked_boxes[idx] + [boundingbox]
                                    #frameList append
                                    frameList[idf] = frameList[idf] + [[idx, 
                                                                       np.squeeze(classes, axis=0)[max_iou_id], 
                                                                       np.squeeze(scores, axis=0)[max_iou_id], 
                                                                       boundingbox, 
                                                                       boxes_t[max_iou_id],
                                                                       1]]
                                elif(max_iou <= rectify_iou and max_iou > 0.1):
                                    #rect
                                    track.init(boxes_t[max_iou_id], frame_np)
                                    boundingbox = track.update(frame_np)
                                    tracked_boxes[idx] = tracked_boxes[idx] + [boundingbox]
                                    frameList[idf] = frameList[idf] + [[idx, 
                                                                       np.squeeze(classes, axis=0)[max_iou_id], 
                                                                       np.squeeze(scores, axis=0)[max_iou_id], 
                                                                       boundingbox, 
                                                                       boxes_t[max_iou_id],
                                                                       1]]
                                else:
                                    tracked_boxes[idx] = tracked_boxes[idx] + [boundingbox]
                                    counter = 0
                                    if(idf >= detection_rate+dlost_thres-1):
                                        for i in range(idf, idf-dlost_thres, -1):
                                            try:
                                                the_score = frameList[i][[row[0] for row in frameList[i]].index(idx)][5]
                                            except ValueError:
                                                the_score = -1
                                            if(the_score == 0):
                                                counter = counter + 1
                                    if(counter < dlost_thres-1):
                                        #detection lost
                                        tracker_bug=0
                                        try:
                                            frameList[idf] = frameList[idf] + [[idx,
                                                                                frameList[idf-1][[row[0] for row in frameList[idf-1]].index(idx)][1], 
                                                                                frameList[idf-1][[row[0] for row in frameList[idf-1]].index(idx)][2], 
                                                                                boundingbox, 
                                                                                boundingbox,
                                                                                0]]
                                        except ValueError:
                                            tracker_bug=1
                                            
                                        if(tracker_bug==1):
                                            frameList[idf] = frameList[idf] + [[idx,
                                                                                np.squeeze(classes, axis=0)[max_iou_id], 
                                                                                np.squeeze(scores, axis=0)[max_iou_id], 
                                                                                boxes_t[max_iou_id], 
                                                                                boxes_t[max_iou_id],
                                                                                0]]
                                    else:
                                        tracks[idx] = []
                                        tracked_boxes[idx] = []
                            else:
                                #no detection result
                                frameList[idf] = frameList[idf] + [[idx, 
                                                                    frameList[idf-1][[row[0] for row in frameList[idf-1]].index(idx)][1], 
                                                                    frameList[idf-1][[row[0] for row in frameList[idf-1]].index(idx)][2], 
                                                                    boundingbox, 
                                                                    boundingbox,
                                                                    frameList[idf-1][[row[0] for row in frameList[idf-1]].index(idx)][5]]]

                #merge
                m_to_del = []
                for box1 in frameList[idf]:
                    for box2 in frameList[idf]:
                        if(box1[0] is not box2[0]):
                            iou_m = get_iou(box1[3], box2[3])
                            if(iou_m >= merge_iou):
                                #delete
                                tracks[max([box1[0], box2[0]])] = []
                                tracked_boxes[max([box1[0], box2[0]])] = []
                                m_to_del = m_to_del + [max([box1[0], box2[0]])]
            
                new_m = []
                for i in m_to_del:
                    if i not in new_m:
                        new_m.append(i)
                for m in new_m:
                    frameList[idf].pop([row[0] for row in frameList[idf]].index(m))
                    
                if(write_video==True):
                    #visualize detection
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        write_on,
                        np.array(rev_box_trans([row[3] for row in frameList[idf]])),
                        np.array([row[1] for row in frameList[idf]]).astype(np.int32),
                        np.array([row[2] for row in frameList[idf]]),
                        category_index,
                        use_normalized_coordinates=False,
                        max_boxes_to_draw=500,
                        line_thickness=1,
                        min_score_thresh=0)
                        
                    for i in frameList[idf]:
                        cv2.putText(write_on, 
                            (str)(i[0]),
                            ((int)(i[3][0]+i[3][2]/2), (int)(i[3][1]+i[3][3]/2)), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (255,255,51), 2)
                    writer.append_data(np.array(write_on))

                    
    end = time.time()
    writer.close()
    writer.close()
    writer.close()
    print('Completed')
    print('Runs on ', len(frames_to_read)/(end - start), ' fps')