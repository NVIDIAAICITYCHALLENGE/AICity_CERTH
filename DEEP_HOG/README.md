- [x] Requirements:

- scipy
- sklearn
- tqdm
- pandas
- keras (tf backend)
- opencv-python


- [x] Instructions:

At first make sure you have a txt file for every image containing detections from the **CERTH_RCNN** in the following format:
```
<class> <xmin> <ymin> <xmax> <ymax> <score>
```

Open `hog_boxes.py` and edit the required paths (there is a help section in there to guide you).

Run ```python hog_boxes.py``` and wait while the scripts exports HOG features for every detected box and saving them in a .hog file (one per image). You now have a HOG feature vector for every detected box in your set.

Open `deephog.py` and edit the required paths and variables.

Run python ```python deephog.py``` and wait while the script calculates a Fisher Vector representation, then classifies the box based on our trained DeepHOG pipeline (read the paper for more info) and finally outputs the result .txt files in the same format as the **CERTH_RCNN**.

Note that the script executes both the **DeepHOG** and the **Ensemble** models in one go. You can disable one of them by commenting the appropriate lines.
