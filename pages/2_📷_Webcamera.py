import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import time
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe_functions import *
from utils import *
import tensorflow as tf

st.title("Webcamera")
st.write("Steps to use: \n1. Click on Start button.\n2. To stop the video when done, press Stop. \n\n The output will be displayed in about 40 secs.")

class VideoProcessor:
    def __init__(self) -> None:
        self.threshold1 = 100
        self.threshold2 = 200
        self.my_list = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.my_list.append(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Create the video processor instance
video_processor = VideoProcessor()

ctx = webrtc_streamer(key="sample",  video_processor_factory=lambda: video_processor)

time.sleep(10)
st.write(len(ctx.video_processor.my_list))

# Access the frames list after the webrtc_streamer function has finished running
frames_list = ctx.video_processor.my_list

# # Display the last frame
# if frames_list:
#     st.image(frames_list[-1], channels="BGR")
st.write("Running...")

# Continuing with the code for inference pipeline
final_landmarks = extract_landmarks(frames_list)
df1 = pd.DataFrame(final_landmarks,columns=['x','y','z'])
ROWS_PER_FRAME = 543

# Loading data
st.write(len(frames_list))
test_df = load_relevant_data_subset(df1, ROWS_PER_FRAME=ROWS_PER_FRAME)
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
