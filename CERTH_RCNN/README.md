## Models

We have created two models. One for each training set [aic1080 & aic480].

We use the 1080 model to detect objects on the 540 set as well.

The folders contain the .config file that was used during training. You don't actually need this to test the model, nevertheless we provide it for anyone interested.
There are also label map files that map each class with a unique id. This is required to run the tests.
You will also need the inference graph file for each model which encodes the frozen weights that the model has converged to and is used by the test script. You can download from here:

- 480: https://www.dropbox.com/s/u6b56bplmqo4t8v/output_inference_graph_480.pb?dl=0
- 1080: https://www.dropbox.com/s/hns3zopbxmgww8m/output_inference_graph_1080.pb?dl=0

## Instructions

- You need to have tensorflow-gpu installed
- Install other required python packages: Pandas, tqdm. You can install those by running:
```
pip install pandas
pip install tqdm
```
- Clone the entire tensorflow models repo inside the tensorflow folder: https://github.com/tensorflow/models
- Install the object_detection API following the instructions in https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md
- Place test_nn.py to object_detection folder, open it and edit the required paths following the instructions.
- By now you should have the following structure:
```
+/path/to/tesorflow/
  +models
    +object_detection
      -test_nn.py
    +slim
```
- From models/object_detection run:
```
python test_nn.py
```
- Once the code finishes you will have a txt file for every image containing bounding box coordinates, classes and scores for each box! :blue_car::bus::truck::vertical_traffic_light:
