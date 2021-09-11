# REAL TIME TRANSLATION OF AMERICAN SIGN LANGUAGE TO TEXT

## 1. STEPS TO SETUP TENSORFLOW OBJECT DETECTION API
**(All resources available on GCS)**
## Installation

First clone the master branch of the Tensorflow Models repository:

```bash
git clone https://github.com/tensorflow/models.git
```
Python Package Installation:

```bash
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python3 -m pip install .
```
To test the installation run:
```
python3 object_detection/builders/model_builder_tf2_test.py
```

## Creating Dataset

 Open the capture_images.ipynb python notebook file and run.

Split the dataset into Train and Test data in separate folders. Suggestion: Take 80% for train and 20% for test.

Label the images:
```python
# Clone the repo
https://github.com/tzutalin/labelImg
# Run the python file
python3 labelImg.py
```
Generate a labelmap.pbtxt file:
```
item { 
    name:'hello'
    id:1
}
item { 
    name:'yes'
    id:2
}
item { 
    name:'no'
    id:3
}
item { 
    name:'thanks'
    id:4
}
item { 
    name:'I love you'
    id:5
}

```




Modify the config file for pre trained model for number of batches and steps to run (here used, faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config):
```
cd models/research/object_detection/configs/tf2/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config
```
Convert xml files to csv:
```
# Run the python file
python3 xml_to_csv.py
```
Convert csv file to tfrecord:
```
# Run the python file to generate tfrecord for train data
python3 /path/generate_tfrecords.py --path_to_images /path/Train --path_to_annot /path/annotations.csv --path_to_label_map /path/
labelmap.pbtxt --path_to_save_tfrecords /path/train.record

# Run the python file to generate tfrecord for test data
python3 /path/generate_tfrecords.py --path_to_images /path/Test --path_to_annot /path/annotations.csv --path_to_label_map /path/
labelmap.pbtxt --path_to_save_tfrecords /path/test.record
```


## Google Cloud AI Platform: Training and Evaluation with TensorFlow 2
**Create a config.yaml file**
```
training_inputs:
    scaleTier: CUSTOM
    masterType: n1-highcpu-16
    workerType: nvidia-tesla-v100
    parameterServerType: standard
    workerCount: 4
    parameterServerCount: 2
    runtimeVersion: 2.4
    pythonVersion: 3.7
```

**Training with multiple GPUs**

A user can start a training job on Cloud AI Platform using the following command:
```
# From the tensorflow/models/research/ directory
cp object_detection/packages/tf2/setup.py .
gcloud ai-platform jobs submit training object_detection_rcnn_`date +%m_%d_%Y_%H_%M_%S` \
    --runtime-version 2.1 \
    --python-version 3.7 \
    --job-dir=gs://${MODEL_DIR} \
    --package-path ./object_detection \
    --module-name object_detection.model_main_tf2 \
    --region us-central1 \
    --scale-tier CUSTOM \
    --master-machine-type n1-highcpu-16 \
    --master-accelerator count=4,type=nvidia-tesla-t4 \
    -- \
    --model_dir=gs://${MODEL_DIR} \
    --config=gs://${CONFIG_PATH} \
    --pipeline_config_path=gs://${PIPELINE_CONFIG_PATH}
    
#Where gs://${MODEL_DIR} specifies the directory on Google Cloud Storage where the training
#checkpoints and events will be written to and gs://${PIPELINE_CONFIG_PATH} points to the
#pipeline configuration stored on Google Cloud Storage and gs://${CONFIG_PATH} points to 
#config.yaml file.
```


**Evaluating with GPU**

Evaluation jobs run on a single machine. Run the following command to start the evaluation job:
```
# From the tensorflow/models/research/ directory
cp object_detection/packages/tf2/setup.py .
gcloud ai-platform jobs submit training object_detection_eval_`date +%m_%d_%Y_%H_%M_%S` \
    --runtime-version 2.1 \
    --python-version 3.6 \
    --job-dir=gs://${MODEL_DIR} \
    --package-path ./object_detection \
    --module-name object_detection.model_main_tf2 \
    --region us-central1 \
    --scale-tier BASIC_GPU \
    -- \
    --model_dir=gs://${MODEL_DIR} \
    --pipeline_config_path=gs://${PIPELINE_CONFIG_PATH} \
    --checkpoint_dir=gs://${MODEL_DIR}
    
#where gs://${MODEL_DIR} points to the directory on Google Cloud Storage where training
#checkpoints are saved and gs://{PIPELINE_CONFIG_PATH} points to where the model
#configuration file stored on Google Cloud Storage. Evaluation events are written to
#gs://${MODEL_DIR}/eval    
```

**Running Tensorboard**

Progress for training and eval jobs can be inspected using Tensorboard. If using the recommended directory structure, Tensorboard can be run using the following command:
```
tensorboard --logdir=${MODEL_DIR}

#where ${MODEL_DIR} points to the directory that contains the train and eval directories.
#Please note it may take Tensorboard a couple minutes to populate with data.
```


## 2. STEPS TO RUN VIDEO CONFERENCING BETWEEN MULTIPLE CLIENTS AND SERVER
Video conferencing between multiple clients and server where the video of the client is captured. Object detection model is applied on this image frame to detect the hand sign and translated to text. The text is attacted to the image frame and broadcasted to the other clients.

**Run the server python code**
```
python3 server.py
```

**Run the client python code**
```
python3 client.py

# Run client on multiple devices/terminal for multiple client video conferencing.
```





