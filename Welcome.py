# Importing necessary libraries
import streamlit as st
import cv2
import json
import numpy as np
import threading 
import time
import pandas as pd
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit.components.v1 as com
from mediapipe_functions import *
import tempfile
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title="Welcome to iSLR",
    page_icon="👋",
)

# st.markdown("""
# <style>
# .css-9s5bis.edgvbvh3
# {
#     visibility:hidden
# }
# .css-h5rgaw.egzxvld1
# {
#     visibility:hidden
# }
# </style>
# """, unsafe_allow_html=True)

st.sidebar.success("Select a demo above.")

st.title("ASL Sign Language Recognition App")

f=st.file_uploader("Please upload a video of a demo of ASL sign", type=['mp4'])

if f is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(f.read())
    
    st.sidebar.video(tfile.name)
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    
    # for local video file
    # cap = cv2.VideoCapture('videos\goodbye.mp4')
    
    final_landmarks=[]
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image, results = mediapipe_detection(frame, holistic)
            draw(image, results)
            landmarks = extract_coordinates(results)
            final_landmarks.extend(landmarks)
    
    df1 = pd.DataFrame(final_landmarks,columns=['x','y','z'])
    
    # Necessary functions 
    ROWS_PER_FRAME = 543
    def load_relevant_data_subset(df):
        data_columns = ['x', 'y', 'z']
        data = df
        n_frames = int(len(data) / ROWS_PER_FRAME)
        data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
        return data.astype(np.float32)
    
    # Loading data
    test_df = load_relevant_data_subset(df1)
    test_df = tf.convert_to_tensor(test_df)
    
    # Inference
    interpreter = tf.lite.Interpreter("models/model.tflite")
    prediction_fn = interpreter.get_signature_runner("serving_default")
    output = prediction_fn(inputs=test_df)
    sign = np.argmax(output["outputs"])
    
    sign_json=pd.read_json("sign_to_prediction_index_map.json",typ='series')
    sign_df=pd.DataFrame(sign_json)
    sign_df.iloc[sign]
    top_indices = np.argsort(output['outputs'])[::-1][:5]
    top_values = output['outputs'][top_indices]
    
    output_df = sign_df.iloc[top_indices]
    output_df['Value'] = top_values
    output_df.rename(columns = {0:'Index'}, inplace = True)
    st.write(output_df)
# callbacks : on_change, on_click
# com.iframe("https://embed.lottiefiles.com/animation/132349")
with open('assets/animations/14592-loader-cat.json') as source:
    animation=json.load(source)
st_lottie(animation, width=300)
