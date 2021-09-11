import cv2
from socket import socket, AF_INET, SOCK_STREAM
from imutils.video import WebcamVideoStream
from array import array
from threading import Thread
import numpy as np
import zlib
import struct
import pickle
import pdb
import sys

HOST = "10.0.0.234"
PORT_VIDEO = 6600

CHUNK=1024
lnF = 640*480*3
CHANNELS=1
RATE=44100

MAX_SEND = (5000 * CHUNK)
LOGS = True


# Convert frame to Byte array
def frameToByteArray(frame):
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 480))
    frame = np.array(frame, dtype = np.uint8).reshape(1, lnF)
    byte_array = bytearray(frame)
    return byte_array

# Send length of the data
def sendDataLen(dataSz):
    length = struct.pack("!I", dataSz)
    if LOGS: print("Length of Data ::: {}".format(length))
    sock.sendall(length)


# Send data to the server
def sendData(data):
    bytesToBeSend = b''
    while len(data) > 0:
        if MAX_SEND <= len(data):
            bytesToBeSend = data[:MAX_SEND]
            data = data[MAX_SEND:]
            if LOGS: print("Length of Data Sent1 ::: {} ".format(len(bytesToBeSend)))
            sock.sendall(bytesToBeSend)
        else:
            bytesToBeSend = data
            if LOGS: print("Length of Data Sent2 ::: {}".format(len(bytesToBeSend)))
            sock.sendall(bytesToBeSend)
            data = b''
    if LOGS: print("##### Data Sent!! #####")


# Send image data frame captured at the client interface to the server
def SendFrame():
    cap = cv2.VideoCapture(0)
    while True:
        try:
            img, frame = cap.read()
            frame_ba = frameToByteArray(frame)
            data = zlib.compress(frame_ba, 9)
            sendDataLen(len(data))
            sendData(data)
        except:
            continue

# Receive image data frame from the server
def RecieveFrame():
    while True:
        try:
            lengthbuf = recvData(4)
            length, = struct.unpack('!I', lengthbuf)
            if LOGS: print("Data to recieve ::: {}".format(length))
            data = recvData(length)
            if LOGS: print("Data received ::: {}".format(len(data)))
            printFrame(data, length)
        except:
            continue


# Display image frame at the client interface
def printFrame(data, length):
    if len(data) != length:
        print("Data corrupted")
        return

    frame = zlib.decompress(data)
    frame = np.array(list(frame))
    frame = np.array(frame, dtype=np.uint8).reshape(480, 640, 3)
    cv2.imshow("Client", frame)
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()


# Receiver data from the server
def recvData(dataSz):
    data = b''
    while len(data) != dataSz:
        dataRead = dataSz - len(data)
        if dataRead > MAX_SEND:
            data += sock.recv(MAX_SEND)
        else:
            data += sock.recv(dataRead)
    return data


# Main function
if __name__ == "__main__":
    sock = socket(family=AF_INET, type=SOCK_STREAM)
    sock.connect((HOST, PORT_VIDEO))


    initiation = sock.recv(5).decode()

    if initiation == "start":
        if len(sys.argv) == 1:
            SendFrameThread = Thread(target=SendFrame).start()
            RecieveFrame()
        elif len(sys.argv) == 2:
            if (sys.argv[1] == "send"):
                SendFrameThread = Thread(target=SendFrame).start()

            if (sys.argv[1] == "recv"):
                RecieveFrame()

