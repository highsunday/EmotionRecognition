import streamlit as st
import cv2
from PIL import Image
import numpy as np
import io
import os
import base64
import predict_function
from const import CLASSES, COLORS
from settings import DEFAULT_CONFIDENCE_THRESHOLD, MODEL, PROTOTXT
from minio import Minio
import pandas as pd
import time
from object_detection.utils import label_map_util

html = """
    <style>
    .big-font {
        font-size:30px !important;
        color: green;
    }
    </style>
    
    <style>
    .big-font2 {
        font-size:35px !important;
       
    }
    </style>
    
  <style>
    .reportview-container {
      flex-direction: row-reverse;
    }

    header > .toolbar {
      flex-direction: row-reverse;
      left: 1rem;
      right: auto;
    }

    .sidebar .sidebar-collapse-control,
    .sidebar.--collapsed .sidebar-collapse-control {
      left: auto;
      right: 0.5rem;
    }

    .sidebar .sidebar-content {
      transition: margin-right .3s, box-shadow .3s;
      width: 805px;
    }

    .sidebar.--collapsed .sidebar-content {
      margin-left: auto;
      margin-right: -21rem;
    }

    @media (max-width: 991.98px) {
      .sidebar .sidebar-content {
        margin-left: auto;
      }
    }
  </style>
"""
st.markdown(html, unsafe_allow_html=True)

def get_image_download_link(img):
    """Generates a link allowing the PIL image to be downloaded
    in:  PIL image
    out: href string
    """
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}"><input type="button" value="Download"></a>'
    return href


@st.cache
def process_image(image):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()
    return detections


st.title("表情辨識")
st.write("""
 這個app可辨識人的七種表情\n
 包含生氣、噁心、害怕、快樂、難過、驚訝與自然\n
""")

DEMO_IMAGE='images/sampleImage.jpg'
DEMO_IMAGE='images/test7.jpg'
img_size=48
confidence_threshold=0.3


face_detector=predict_function.openFaceDetector()
model=predict_function.openCustom_model()

img_file_buffer = st.file_uploader("", type=["png", "jpg", "jpeg"])
label_list=None
if img_file_buffer is not None:
    image_orignal=Image.open(img_file_buffer)
    image=np.array(Image.open(img_file_buffer).convert("RGB"))
    image,crop_img_list,label_list=predict_function.predict_image(face_detector,model,image,confidence_threshold)
    st.image(
    image, caption=f"Upload image", use_column_width=True,
    )
else:
    demo_image = DEMO_IMAGE
    image_orignal=Image.open(demo_image)
    image=np.array(Image.open(demo_image).convert("RGB"))
    image,crop_img_list,label_list=predict_function.predict_image(face_detector,model,image,confidence_threshold)
    st.image(
    image, caption=f"Sample image", use_column_width=True,
    )
result = Image.fromarray(image)
st.markdown(get_image_download_link(result), unsafe_allow_html=True)
face_num=len(crop_img_list)
st.sidebar.markdown('<p class="big-font2">'+'預測結果:'+'</p>', unsafe_allow_html=True)
face_id_list=[]
for i in range(face_num):
    face_id_list.append("id:"+str(i))
option = st.sidebar.selectbox('',face_id_list)
option=option.split(':')[1]

st.sidebar.markdown('<p class="big-font">'+label_list[int(option[-1])]+'</p>', unsafe_allow_html=True)
st.sidebar.image(crop_img_list[int(option[-1])], use_column_width=True)
st.sidebar.write()
    



