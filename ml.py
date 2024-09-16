import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
from io import BytesIO
import pygame
import time
import base64
import speech_recognition as sr
import threading
from tensorflow.keras.models import load_model
import pyttsx3
import easyocr  

# Initialize pygame mixer for playing audio
pygame.mixer.init()

# Initialize the OCR reader
reader = easyocr.Reader(['en', 'hi'])  # You can add more languages

# Load the YOLO model
net = cv2.dnn.readNet('https://www.kaggle.com/datasets/valentynsichkar/yolo-coco-data?select=yolov3.weights', 'https://www.kaggle.com/datasets/valentynsichkar/yolo-coco-data?select=yolov3.cfg')
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]

# Define the classes for YOLO
with open('https://www.kaggle.com/datasets/valentynsichkar/yolo-coco-data?select=coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Global variables for controlling the app
stop_signal = False
command = ""
engine = pyttsx3.init()

# Function to calculate distance (dummy function, modify according to camera specifics)
def calculate_distance(width, known_width=30, focal_length=700):
    distance = (known_width * focal_length) / width
    return distance

# Function to detect objects
def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    results = []
    if len(indexes) > 0:
        indexes = indexes.flatten()
        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            distance = calculate_distance(w)  # Calculate distance based on the width of the object
            results.append({
                "label": label,
                "confidence": confidences[i],
                "box": (x, y, w, h),
                "distance": distance
            })

    return results

# Function to detect text using OCR
def detect_text(frame, language):
    # Convert the frame to grayscale (optional)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform OCR on the image
    result = reader.readtext(gray)

    if len(result) > 0:
        # Loop through detected text
        for (bbox, text, prob) in result:
            if prob > 0.5:  # Confidence threshold
                # Draw bounding box around text
                cv2.rectangle(frame, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (0, 255, 0), 2)
                cv2.putText(frame, text, (int(bbox[0][0]), int(bbox[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Speak the detected text
                speak(text, lang=language[1])

    return frame

# Function to handle frame processing
def process_frame(frame, language):
    # Object detection
    detections = detect_objects(frame)
    
    for detection in detections:
        x, y, w, h = detection["box"]
        label = detection["label"]
        distance = detection["distance"]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f'{label} ({distance / 100:.2f} meters)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Speak object detection result and distance in meters
        speak(f'{label} detected at {distance / 100:.2f} meters', lang=language[1])

    # OCR detection
    frame = detect_text(frame, language)

    return frame

# Function to speak text using gTTS and pygame
def speak(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    pygame.mixer.music.load(audio_data, "mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

# Streamlit design
def main():
    global stop_signal, command

    st.set_page_config(page_title="Smart Guide with Text Detection", layout="wide")

    # Add a background image
    def get_base64_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    img_base64 = get_base64_image("pic/ist.jpg")

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            font-family: 'Arial', sans-serif;
        }}
        .card-box {{
            background-color: rgb(255 255 255 / 20%);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 0px 15px rgb(0 0 0 / 50%)
            max-width: 700px;
            margin: 50px auto;
            text-align: center;
        }}
        .stButton > button {{
            width: 100%;
            padding: 12px;
            border-radius: 10px;
            background-color: #007bff;
            color: white;
            border: none;
            font-size: 18px;
        }}
        .stButton > button:hover {{
            background-color: #0056b3;
        }}
        h1 {{
            color: #fff;
            font-size: 2.5rem;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #fff;
            font-size: 2rem;
            margin-bottom: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.image("pic/Smart.png", width=150) 
    #st.title("Smart Guide")

    st.markdown(
        """
        <div class="card-box">
            <h2>Real time object and text detection</h2>
           
        </div>
        """,
        unsafe_allow_html=True
    )

    language = st.selectbox(
        "Select Language", 
        [
            ("English", "en"), 
            ("Bengali", "bn"), 
            ("Hindi", "hi"),
            ("Tamil", "ta"),
            ("Telugu", "te"),
            ("Spanish", "es"),
            ("French", "fr"),
            ("German", "de"),
            ("Chinese", "zh"),
            ("Japanese", "ja"),
            ("Korean", "ko")
        ]
    )

    video_source = st.selectbox("Select Video Source", ["Webcam", "Upload Video", "Upload Image"])

    if video_source == "Webcam":
        video_stream = st.checkbox("Start Video Stream")
        if video_stream:
            run_webcam(language)

    elif video_source == "Upload Video":
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mkv", "mov"])
        if uploaded_video:
            process_uploaded_video(uploaded_video, language)

    elif video_source == "Upload Image":
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            process_uploaded_image(uploaded_image, language)

# Function to run the webcam
def run_webcam(language):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame, language)
        stframe.image(processed_frame, channels="BGR")

    cap.release()

# Function to process uploaded video
def process_uploaded_video(video, language):
    video_path = video.name
    with open(video_path, "wb") as f:
        f.write(video.getbuffer())
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame, language)
        stframe.image(processed_frame, channels="BGR")

    cap.release()

# Function to process uploaded image
def process_uploaded_image(image, language):
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    processed_frame = process_frame(frame, language)
    st.image(processed_frame, channels="BGR")

if __name__ == '__main__':
    main()
