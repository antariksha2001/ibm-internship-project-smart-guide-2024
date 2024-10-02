
<body>
    <div class="container">
        <h1>Smart Guide for a Blind Person</h1>
        
  <div class="section">
            <h2>Project Overview</h2>
            <p>
         
The <strong>Smart Guide for a Blind Person</strong> is an AI-driven assistive system designed to help visually impaired individuals with real-time navigation and object recognition. The project integrates state-of-the-art technologies like YOLO (You Only Look Once) for object detection and EasyOCR for text recognition. The solution provides real-time audio feedback in multiple languages, allowing users to safely navigate their surroundings. This project aims to address the limitations of traditional navigation aids like white canes by offering an intelligent, real-time guiding system.

  </p>
        </div>

   <div class="section">
            <h2>Key Features</h2>
            <ul>
               
  <li><strong> 1. Real-Time Object Detection</strong>:
The system detects objects in real-time using the YOLOv3 model and provides details about their distance. Objects in the user’s path are identified, and an audio alert is generated to inform the user.</li>

<li><strong>2. Real-Time Text Detection</strong>:
Using EasyOCR, the system can detect and read text from the user's environment. This feature is particularly useful for reading signs, documents, and other important visual information.
</li>
<li><strong>3. Multilingual Audio Feedback</strong>:
The detected objects and texts are converted to speech using gTTS (Google Text-to-Speech) for cloud-based audio synthesis or pyttsx3 for offline use. The system supports multiple languages, providing greater accessibility for users.
</li>
<li><strong>4. Distance Calculation</strong>:
For each detected object, the system calculates the distance from the user and announces it to help avoid obstacles and navigate safely.
</li>
<li><strong>5. Simple and Accessible Interface</strong>:
Developed using Streamlit, the interface is designed to be simple and intuitive, providing options for:
</li>
<li>Starting and stopping object and text detection.</li>
<li>Switching between different languages for audio feedback.</li>
<li>Real-time video feed from the camera for object and text detection.
            </ul>
        </div>

 
<h2>Technology Stack</h2>
<strong>Object Detection</strong>
YOLO (You Only Look Once): A state-of-the-art real-time object detection system that can process images quickly and efficiently.


<strong>Text Detection</strong>
EasyOCR: An open-source optical character recognition library that extracts text from images.


<strong>Audio Feedback</strong>
<ul><li><b>gTTS:</b> Google Text-to-Speech for converting text into audio.</li>
<li><b>pyttsx3:</b> An offline text-to-speech library for Python.</li></ul>
<strong>Framework</strong>
Streamlit: A web application framework for machine learning and data science projects, used to build the user interface.
Programming Language
Python 3.11: The core language used for developing the project.
Additional Libraries
OpenCV: For processing images from the webcam.
Pygame: For playing back audio feedback.
Tesseract-OCR: An alternative to EasyOCR for text recognition, supporting additional languages.
Hardware and Software Requirements
Hardware
Webcam: To capture video feed for object and text detection.
CPU/GPU: A computer with a good CPU/GPU is recommended for real-time processing.
Software
Python 3.11
VS Code: Integrated development environment for Python.
