🎵 MoodMate: Emotion-Based Music Recommender

MoodMate is an interactive web application that uses deep learning to detect a user's emotion in real-time and recommends a personalized music playlist to match their mood. It features a modern, stylish UI and can analyze emotions from both static images and a live webcam feed.


✨ Key Features

Real-Time Emotion Detection: 🧑‍💻 Uses a live webcam feed to classify your emotion (Happy, Sad, Angry, etc.) instantly.

Image-Based Analysis: 🖼️ Upload any image with a face to get an emotion analysis and a curated playlist.

Confidence Score: 📊 Clearly displays the model's confidence (e.g., "HAPPY (95%)") so you can trust the prediction.

Interactive Playlists: 🎶 Generates a custom playlist based on your detected mood, complete with artist and song title.

YouTube Integration: ▶️ Each song in the playlist includes a "Listen ▷" link to search for the song on YouTube.

Stylish UI: ✨ A modern, "glassmorphism" interface built with Streamlit.

🛠️ Technology Stack

Model: VGG16 (Fine-tuned on the FER-2013 dataset)

Deep Learning: TensorFlow / Keras

Web App: Streamlit

Real-Time Video: streamlit-webrtc

Computer Vision: OpenCV (Haar Cascade for face detection)

Data Handling: Pandas & NumPy

UI/UX: Custom CSS (Glassmorphism)

🚀 Installation

Follow these steps to set up and run the project on your local machine.

1. Clone the Repository
   
   git clone https://github.com/YourUsername/Emotion-Detection-and-Music-Recommended-system.git
   cd Emotion-Detection-and-Music-Recommended-system
2. Create and Activate a Virtual Environment
   
   This is crucial for managing your project's dependencies.
   On Windows (cmd.exe):
   
   python -m venv venv
   venv\Scripts\activate
   On macOS/Linux:
   
   python3 -m venv venv
   source venv/bin/activate
3. Install Required Libraries
   
   All dependencies are listed in the requirements.txt file.
   pip install -r requirements.txt
4. Download Required Files
   For the application to run, you need two key files:

* Emotion Model: You need the trained model file. This project assumes it is located at models/final_tuned_vgg16_model.h5.

* Face Detector: The webcam feature requires the OpenCV face detector.

Download this file: haarcascade_frontalface_default.xml

Save it in the main folder of your project (the same place as ui_app.py).
▶️ How to Run the App

Once all dependencies are installed and your venv is active, run the following command from your project's root directory:
streamlit run ui_app.py

Your web browser will automatically open, and you can start using the application!
