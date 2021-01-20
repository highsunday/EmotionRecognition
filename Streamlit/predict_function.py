import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def openFaceDetector():
    confidence_threshold=0.3
    prototxt_path="model/face_detector/deploy.prototxt"
    model_path="model/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return face_detector

def openCustom_model():
    save_model_path="model/save_model/run_2020_12_03-19_37_22.h5"
    model = load_model(save_model_path) 
    return model

def predict_image(decoder,model,image,confidence_threshold):
    label=["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    decoder.setInput(blob)
    detections = decoder.forward()
    crop_img_list=[]
    label_list=[]
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
            
            crop_img = image[startY:endY, startX:endX]
            crop_img_list.append(cv2.resize(crop_img, (200, 300), interpolation=cv2.INTER_AREA))
            crop_img = cv2.resize(crop_img, (48, 48), interpolation=cv2.INTER_AREA)
            crop_img = crop_img.reshape(1, 48, 48, 3)
            res=predict_emotion(model,crop_img)
            y_classes = res.argmax(axis=-1)
            #print(res)
            predict=label[y_classes[0]]
            label_list.append(predict)
            cv2.putText(image, str(i)+":"+predict, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
    return image,crop_img_list,label_list

def predict_emotion(model,crop_img):
    return model.predict(crop_img)
