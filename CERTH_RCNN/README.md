-You need to have tensorflow-gpu installed

-Install other required python packages: Pandas, tqdm

-Clone the entire tensorflow models repo inside the tensorflow folder: https://github.com/tensorflow/models

-Install the object_detection API following the instructions in https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md

-Place test_nn.py to object_detection folder, open it and edit the required paths.

-By now you should have the following structure:
        +/path/to/tesorflow/
            +models
                +object_detection
                    -test_nn.py
                +slim

-Run python test_nn.py

-Once the code finishes you will have a txt file for every image containing bounding box coordinates, classes and scores for each box!
