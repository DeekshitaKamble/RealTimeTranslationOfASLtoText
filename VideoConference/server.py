import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import cv2 
import numpy as np
import time
from socket import socket, AF_INET, SOCK_STREAM
from threading import Thread
import struct
import pyshine as ps
import zlib
import pdb


# Load pipeline config and build a detection model
CONFIG_PATH="/Users/deekshitakamble/Documents/Comp512_project/ObjectDetection1/Experiment_1/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config"
CHECKPOINT_PATH="/Users/deekshitakamble/Documents/Comp512_project/ObjectDetection1/Experiment_1/inference/data/checkpoints"
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-2')).expect_partial()

HOST = "10.0.0.234"
PORT_VIDEO = 6600
lnF = 640*480*3
CHUNK = 1024
BufferSize = 4096
addressesText = {}
addresses = {}
threads = {}

MAX_SEND = (5000 * CHUNK)

LOGS = True
font = cv2.FONT_HERSHEY_SIMPLEX


# Helper function for object detection
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap('/Users/deekshitakamble/Documents/Comp512_project/ObjectDetection1/Experiment_1/data/labelmap.pbtxt')


# Make a connection from the client to the server
def ConnectionsVideo():
    while True:
        try:
            clientVideo, addr = serverVideo.accept()
            print("{} is connected. Socket fd {}".format(addr, clientVideo))
            addresses[clientVideo] = addr
            if len(addresses) > 1:
                for sockets in addresses:
                    if sockets not in threads:
                        threads[sockets] = True
                        sockets.send(("start").encode())
                        Thread(target=ClientConnectionVideo, args=(sockets, )).start()
            else:
                continue
        except:
            continue


# The compressed data received from client is decompressed and converted to image frame
def extractFrame(data, length):
    if len(data) != length:
        print("Corrupted Data")
        return np.zeros(480, 640, 3)

    frame = zlib.decompress(data)
    frame = np.array(list(frame))
    frame = np.array(frame, dtype=np.uint8).reshape(480, 640, 3)
    return frame


# Helper function
def listToString(sentence): 
    str1 = " " 
    return (str1.join(sentence))


# Integrate the object detection model to translate the image hand signs to text
def getText(image_np, sentence):
	input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
	detections = detect_fn(input_tensor)
    
	num_detections = int(detections.pop('num_detections'))
	detections = {key: value[0, :num_detections].numpy()
					for key, value in detections.items()}
	detections['num_detections'] = num_detections

    # detection_classes should be ints.
	detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

	label_id_offset = 1
	image_np_with_detections = image_np.copy()

	viz_utils.visualize_boxes_and_labels_on_image_array(
				image_np_with_detections,
				detections['detection_boxes'],
				detections['detection_classes']+label_id_offset,
				detections['detection_scores'],
				category_index,
				use_normalized_coordinates=True,
				max_boxes_to_draw=5,
				min_score_thresh=.5,
				agnostic_mode=False)
    
    # Extract main word and append to sentence
	word = category_index[detections['detection_classes'][np.argmax(detections['detection_scores'])]+1]['name']
	if (sentence[len(sentence)-1] != word):
		sentence.append(word)
	time.sleep(2)
	sentence = sentence[-5:]
	print(sentence)
	sentences = listToString(sentence)
	return sentences


# Add text to the image data frame
def appendText(text, frame):
    print("APPENDING TEXT")
    frame = ps.putBText(frame, text, 10, 10, vspace=10, hspace=1, font_scale=0.7, background_RGB=(255, 0, 0), text_RGB=(255, 250, 250))
    return frame


# Receive data from the client
def recvData(sock, dataSz):
    data = b''
    while len(data) != dataSz:
        dataRead = dataSz - len(data)
        if dataRead > MAX_SEND:
            data += sock.recv(MAX_SEND)
        else:
            data += sock.recv(dataRead)
    return data


# Broadcast the image frame to all other clients
def broadcastVideo(senderSock, data):
    frame = np.array(data, dtype=np.uint8).reshape(1, lnF)
    data = bytearray(frame)
    data = zlib.compress(data, 9)
    print("Length of Data sent ::: {}".format(len(data)))
    length = struct.pack('!I', len(data))
    for sock in addresses:
        if sock != senderSock:
            sock.sendall(length)

    dataToSend = b''
    while len(data) > 0:
        if MAX_SEND <= len(data):
            dataToSend = data[:MAX_SEND]
            data = data[MAX_SEND:]
            for sock in addresses:
                if sock != senderSock:
                    sock.sendall(dataToSend)
        else:
            for sock in addresses:
                if sock != senderSock:
                    sock.sendall(data)
                    data = b''


# Receive data from client and add text to the image frame
def ClientConnectionVideo(clientVideo):
    sentence = [' ']
    while True:
        try:
            dataSz = recvData(clientVideo, 4)
            length, = struct.unpack('!I', dataSz)
            if LOGS: print("Length of Data To Receive ::: {}".format(length))
            data = recvData(clientVideo, length)
            if LOGS: print("Length of Data Received ::: {}".format(len(data)))

            frame = extractFrame(data, length)
            text = getText(frame, sentence)
            frame = appendText(text, frame)
            
            broadcastVideo(clientVideo, frame)
        except:
            continue


# Main fucntion
if __name__ == "__main__":
    serverVideo = socket(family=AF_INET, type=SOCK_STREAM)
    try:
        serverVideo.bind((HOST, PORT_VIDEO))
    except OSError:
        print("Server Busy")
        exit()

    serverVideo.listen(2)
    print("Waiting for connection..")
    AcceptThreadVideo = Thread(target=ConnectionsVideo)
    AcceptThreadVideo.start()
    AcceptThreadVideo.join()
    serverVideo.close()


