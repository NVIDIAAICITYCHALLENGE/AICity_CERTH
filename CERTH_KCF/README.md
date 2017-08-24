The CERTH_KCF tracker gets a video sequence and an inference tf graph of the trained model and performs object detection and tracking exporting the results to an output video.

To run the tracker you need to install this python KCF implementation https://github.com/uoip/KCFcpp-py-wrapper

Extra packages needed:
-imageio
-opencv-python
-tqdm
-ffmpeg

Place the KCF folder inside tensorflow/models/object_detection and add it's path to PYTHOPATH

Type python certh_kcf.py --help to get info about the parameters