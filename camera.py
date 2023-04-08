import cv2
from imutils.video import VideoStream
import imutils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os
from flask import Response

class VideoCamera(object):
    def __init__(self):
        self.faceNet = cv2.dnn.readNetFromCaffe('model.prototxt', 'model.caffemodel')
        self.maskNet = load_model('mobilenet_v2.model')
        self.vs = VideoStream(src=0).start()

    def __del__(self):
        self.vs.stop()

    def detect_and_predict_mask(self, frame):
        # grab the dimensions of the frame and then construct a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()

        # initialize our list of faces, their corresponding locations and list of predictions

        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                # we need the X,Y coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                # ensure the bounding boxes fall within the dimensions of the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            faces = np.array(faces, dtype='float32')
            preds = self.maskNet.predict(faces, batch_size=12)

        return (locs, preds)

    def get_frame(self):
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = self.vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and predict if they are wearing masks or not
        (locs, preds) = self.detect_and_predict_mask(frame)

        # loop over the detected face locations and their corresponding locations

        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # resize the frame to have a maximum width of 600 pixels
        frame = imutils.resize(frame, width=600)

        # encode the frame in JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)

        # convert the frame from JPEG to bytes
        data = jpeg.tobytes()

        return data
