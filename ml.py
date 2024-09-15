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
from datetime import datetime


pygame.mixer.init()
reader = easyocr.Reader(['en', 'hi'])

# Load the YOLO model
net = cv2.dnn.readNet('ojt//yolov3.weights', 'ojt//yolov3.cfg')
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]

# Define the classes for YOLO
with open('ojt//coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Global variables for controlling the app
stop_signal = False
start_signal = False
engine = pyttsx3.init()

# Function to calculate distance (in meters)
def calculate_distance(width, known_width=30, focal_length=700):
    distance_cm = (known_width * focal_length) / width
    distance_m = distance_cm / 100  
    return distance_m

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
            distance = calculate_distance(w)
            results.append({
                "label": label,
                "confidence": confidences[i],
                "box": (x, y, w, h),
                "distance": distance
            })

    return results

# Function to detect text using OCR
def detect_text(frame, language):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)

    if len(result) > 0:
        for (bbox, text, prob) in result:
            if prob > 0.5:
                cv2.rectangle(frame, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (0, 255, 0), 2)
                cv2.putText(frame, text, (int(bbox[0][0]), int(bbox[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                speak(text, lang=language[1])

    return frame

# Function to process the frame
def process_frame(frame, language):
    detections = detect_objects(frame)

    for detection in detections:
        x, y, w, h = detection["box"]
        label = detection["label"]
        distance = detection["distance"]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f'{label} ({distance:.2f} meters)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        speak(f'{label} detected at {distance:.2f} meters', lang=language[1])

    frame = detect_text(frame, language)
    return frame

# Function to speak text
def speak(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    pygame.mixer.music.load(audio_data, "mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

# Voice recognition function with date and time validation
def recognize_voice():
    global start_signal, stop_signal
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    while True:
        with mic as source:
            audio = recognizer.listen(source)

     

    try:
        command = recognizer.recognize_google(audio).lower()
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")

        if "start" in command:
            start_signal = True
            stop_signal = False
            speak(f"Starting the Smart Guide at {current_time}", 'en')

        elif "stop" in command:
            stop_signal = True
            start_signal = False
            speak(f"Stopping the Smart Guide at {current_time}", 'en')

        # New commands for time and date
        elif "time" in command or "tell me the time" in command:
            speak(f"The current time is {current_time}", 'en')

        elif "date" in command or "tell me the date" in command:
            speak(f"Today's date is {current_date}", 'en')

    except sr.UnknownValueError:
        pass


# Streamlit design
def main():
    global stop_signal, start_signal

    st.set_page_config(page_title="Smart Guide", layout="wide")

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
            background-color: rgba(255, 255, 255, 0.9);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.3);
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
            color: #333;
            font-size: 2.5rem;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #555;
            font-size: 2rem;
            margin-bottom: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Smart Guide")

    st.markdown(
        """
        <div class="card-box">
            <h2>Experience Object and Text Detection in Real-Time</h2>
           
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
        st.write("Use voice command 'start' to begin the video stream and 'stop' to end it.")

    voice_thread = threading.Thread(target=recognize_voice)
    voice_thread.daemon = True
    voice_thread.start()

    video_frame = st.empty()

    if video_source == "Webcam":
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_signal:
                break

            if start_signal:
                frame = process_frame(frame, language)

            video_frame.image(frame, channels="BGR")

        cap.release()

    if video_source == "Upload Video":
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mkv"])
        if uploaded_video:
            temp_file = "temp_video.mp4"
            with open(temp_file, "wb") as f:
                f.write(uploaded_video.getbuffer())

            cap = cv2.VideoCapture(temp_file)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop_signal:
                    break

                if start_signal:
                    frame = process_frame(frame, language)

                video_frame.image(frame, channels="BGR")

            cap.release()

    if video_source == "Upload Image":
        uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
        if uploaded_image:
            img_array = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_array, 1)
            frame = process_frame(frame, language)
            video_frame.image(frame, channels="BGR")

if __name__ == "__main__":
    main()
