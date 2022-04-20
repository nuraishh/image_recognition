import logging
import os
from datetime import datetime
import cv2

import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input

vgg_model = vgg16.VGG16(weights='imagenet')


def predict_frame(image):
    # reverse color channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reshape image to (1, 224, 224, 3)
    image_batch = np.expand_dims(image, axis=0)

    # apply pre-processing
    image = preprocess_input(image_batch)

    #predictions
    predictions = vgg_model.predict(image)
    label_vgg = keras.applications.imagenet_utils.decode_predictions(
    predictions)

    return label_vgg



def key_action():
    # https://www.ascii-code.com/
    k = cv2.waitKey(1)
    if k == 113: # q button
        return 'q'
    if k == 32: # space bar
        return 'space'
    if k == 112: # p key
        return 'p'
    return None


def init_cam(width, height):
    """
    setups and creates a connection to the webcam
    """

    logging.info('start web cam')
    cap = cv2.VideoCapture(0)

    # Check success
    if not cap.isOpened():
        raise ConnectionError("Could not open video device")

    # Set properties. Each returns === True on success (i.e. correct resolution)
    assert cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    assert cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def add_text(text, frame):
    # Put some rectangular box on the image
    #cv2.putText()
    return NotImplementedError
