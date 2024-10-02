 <h1>Smart Guide for a Blind Person</h1>
        
       
   <h2>Project Overview</h2>
            <p>
                <strong>Smart Guide for a Blind Person</strong> is an assistive technology designed to enhance the independence and safety of visually impaired individuals. The system integrates technologies such as <strong>object detection, OCR, currency recognition, GPS navigation</strong>, and <strong>speech synthesis</strong>, providing real-time feedback through voice guidance. It uses <em>YOLOv3-tiny, OpenCV, EasyOCR, Tesseract</em>, and <em>PyTorch</em>, with a web and mobile app interface.
            </p>
        </div>

       
            <h2>Key Features</h2>
            <ul>
                <li><strong>Object Detection</strong>: Detects objects in real time using the YOLO model.</li>
                <li><strong>Currency Detection</strong>: Recognizes various Indian currency notes.</li>
                <li><strong>Text Recognition (OCR)</strong>: Reads text in multiple languages (Tamil, Telugu, Hindi, English).</li>
                <li><strong>GPS Navigation</strong>: Provides spoken directions via Google Maps API.</li>
                <li><strong>Traffic Light Detection</strong>: Informs when it is safe to cross streets.</li>
                <li><strong>Surface Hazard Detection</strong>: Detects unsafe surfaces like wet or muddy areas.</li>
                <li><strong>Heat/Temperature Detection</strong>: Warns about hot or hazardous objects.</li>
                <li><strong>Voice Commands</strong>: Allows control via voice interaction.</li>
                <li><strong>SOS System</strong>: Sends emergency alerts with the user’s location.</li>
            </ul>
        </div>

        <div class="section">
            <h2>Technologies Used</h2>
            <ul>
                <li><strong>Backend:</strong> Python, Flask</li>
                <li><strong>Object Detection:</strong> YOLOv3-tiny, OpenCV</li>
                <li><strong>OCR:</strong> EasyOCR, Tesseract</li>
                <li><strong>Speech Synthesis:</strong> pyttsx3, gTTS</li>
                <li><strong>GPS Navigation:</strong> Google Maps API</li>
                <li><strong>Front-end:</strong> HTML, CSS, JavaScript</li>
                <li><strong>Mobile Interface:</strong> Streamlit-based and Flask-based apps</li>
            </ul>
        </div>

        <div class="section">
            <h2>Installation</h2>
            <p>To run this project locally, follow these steps:</p>
            <pre>
1. Clone the repository:
   git clone https://github.com/yourusername/smart-guide-for-blind-person.git
   cd smart-guide-for-blind-person

2. Install the dependencies:
   pip install -r requirements.txt

3. Download Pre-trained YOLOv3-tiny Weights and place in the `models/` directory.

4. Run the application:
   python app.py
            </pre>
        </div>

        <div class="section">
            <h2>Usage</h2>
            <p>Follow these steps to use the key features:</p>
            <ul>
                <li><strong>Object Detection</strong>: Select object detection and choose camera input or upload a video/image.</li>
                <li><strong>Currency Detection</strong>: Choose currency detection and point the camera at the note.</li>
                <li><strong>OCR</strong>: Select OCR, choose a language, and point the camera at the text.</li>
                <li><strong>GPS Navigation</strong>: Enter destination and follow spoken directions.</li>
                <li><strong>SOS System</strong>: Activate the SOS feature to send emergency location data.</li>
            </ul>
        </div>

        <div class="section">
            <h2>Project Structure</h2>
            <pre>
smart-guide-for-blind-person/
├── app.py                     # Main Flask app
├── static/                    # CSS, JS files
├── templates/                 # HTML files
├── models/                    # YOLOv3-tiny weights
├── currency_detection/        # Currency detection logic
├── ocr/                       # OCR logic
├── gps_navigation/            # GPS navigation logic
├── speech_synthesis/          # Real-time voice feedback
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
            </pre>
        </div>

        <div class="section">
            <h2>Hardware Requirements</h2>
            <ul>
                <li>Raspberry Pi (optional, for portability)</li>
                <li>Camera module or webcam</li>
                <li>Smartphone (for mobile app)</li>
                <li>Internet connection (for cloud-based features)</li>
            </ul>
        </div>

        <div class="section">
            <h2>Challenges and Limitations</h2>
            <ul>
                <li><strong>Voice Command Recognition</strong>: May face challenges in noisy environments.</li>
                <li><strong>Object Detection Accuracy</strong>: Can vary in crowded or busy areas.</li>
                <li><strong>Hardware Dependencies</strong>: Constant internet connection needed for some features.</li>
            </ul>
        </div>

        <div class="section">
            <h2>Future Enhancements</h2>
            <ul>
                <li>Integrating more languages for OCR.</li>
                <li>Improving object detection accuracy.</li>
                <li>Enhanced AI assistant for more complex queries.</li>
                <li>Battery optimization for mobile devices.</li>
            </ul>
        </div>

        <footer>
            <p>Smart Guide for a Blind Person &copy; 2024 | Developed by Your Name</p>
        </footer>

