import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import time

# --- App Configuration ---
# Use the "wide" layout to give our app more space
st.set_page_config(
    page_title="MoodMate | Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# --- 1. STYLISH UI: Custom CSS for Glassmorphism & Background ---

def load_css():
    st.markdown(
        """
        <style>
        /* Import a cool font */
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

        html, body, [data-testid="stApp"] {
            font-family: 'Montserrat', sans-serif;
            /* Dark abstract background */
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
                              url("https://images.unsplash.com/photo-1506269996136-39e3831A335c?auto=format&fit=crop&q=80");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        /* Make all text white for contrast */
        h1, h2, h3, h4, h5, h6, [data-testid="stMarkdownContainer"] p, .st-emotion-cache-1jicfl2 {
            color: #FFFFFF !important;
        }

        /* Style for the main "glass" containers */
        .glass-container {
            background-color: rgba(40, 40, 40, 0.6); /* Semi-transparent black */
            backdrop-filter: blur(10px); /* The "frosted glass" effect */
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        /* Style for the emotion result box */
        .emotion-box {
            background-color: rgba(0, 150, 0, 0.7); /* Greenish glass */
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 10px 20px;
            border: 1px solid rgba(0, 255, 0, 0.2);
            text-align: center;
        }
        
        .emotion-box h2 {
            color: #FFFFFF;
            font-weight: 700;
            margin: 0;
        }

        /* Style the tabs */
        [data-testid="stTabs"] {
            background: none;
        }
        [data-testid="stTabs"] button {
            color: #ADADAD; /* Greyed out */
            font-size: 1.1rem;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #FFFFFF; /* White and bold when selected */
            font-weight: 700;
            border-bottom: 3px solid #00A36C;
        }

        /* Style the file uploader button */
        [data-testid="stFileUploader"] label {
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px dashed rgba(255, 255, 255, 0.4);
            color: #FFFFFF;
            border-radius: 10px;
        }
        [data-testid="stFileUploader"] label:hover {
            border-color: #00A36C;
            color: #00A36C;
        }
        
        /* Style the webcam 'Start' button */
        .stButton button {
            background-color: #00A36C;
            color: white;
            font-weight: 600;
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
        }
        .stButton button:hover {
            background-color: #008256;
            color: white;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

# --- File Paths and Constants ---
MODEL_PATH = 'models/final_tuned_vgg16_model.h5'
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
EMOTION_MAP = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

# --- 🎵 Song Database (Built-in) ---
SONG_DATABASE = {
    'happy': [
        {'artist_name': 'Pharrell Williams', 'title': 'Happy'},
        {'artist_name': 'Katrina & The Waves', 'title': 'Walking On Sunshine'},
        {'artist_name': 'Queen', 'title': "Don't Stop Me Now"}
    ],
    'sad': [
        {'artist_name': 'Adele', 'title': 'Someone Like You'},
        {'artist_name': 'Johnny Cash', 'title': 'Hurt'},
        {'artist_name': 'R.E.M.', 'title': 'Everybody Hurts'}
    ],
    'angry': [
        {'artist_name': 'Linkin Park', 'title': 'In the End'},
        {'artist_name': 'Green Day', 'title': 'American Idiot'},
        {'artist_name': 'Nine Inch Nails', 'title': 'Head Like a Hole'}
    ],
    'neutral': [
        {'artist_name': 'Enya', 'title': 'Orinoco Flow'},
        {'artist_name': 'Bon Iver', 'title': 'Skinny Love'},
        {'artist_name': 'Norah Jones', 'title': 'Come Away With Me'}
    ],
    'surprise': [
        {'artist_name': 'Queen', 'title': 'Bohemian Rhapsody'},
        {'artist_name': 'Talking Heads', 'title': 'Once in a Lifetime'},
        {'artist_name': 'David Bowie', 'title': 'Changes'}
    ],
    'fear': [
        {'artist_name': 'Pink Floyd', 'title': 'Run Like Hell'},
        {'artist_name': 'Michael Jackson', 'title': 'Thriller'},
        {'artist_name': 'Nine Inch Nails', 'title': 'Closer'}
    ],
    'disgust': [
        {'artist_name': 'Nirvana', 'title': 'Smells Like Teen Spirit'},
        {'artist_name': 'CeeLo Green', 'title': 'F**k You'},
        {'artist_name': 'The Offspring', 'title': 'Self Esteem'}
    ]
}

# --- Caching Models ---
@st.cache_resource
def load_emotion_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"🔴 Error loading emotion model: {e}")
        return None

@st.cache_resource
def load_face_detector():
    try:
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if face_cascade.empty():
            st.error(f"🔴 Error loading Haar Cascade file. Make sure '{HAAR_CASCADE_PATH}' is in the folder.")
            return None
        return face_cascade
    except Exception as e:
        st.error(f"🔴 Error loading Haar Cascade: {e}")
        return None

# --- 2. INTERACTIVE FEATURE: YouTube Links ---
@st.cache_data
def recommend_songs(emotion, num_recommendations=3):
    """Recommends songs and adds YouTube search links."""
    song_list = SONG_DATABASE.get(emotion, SONG_DATABASE['neutral'])
    playlist_df = pd.DataFrame(song_list).sample(n=num_recommendations)
    
    def create_youtube_link(row):
        query = f"{row['artist_name']} {row['title']}".replace(" ", "+")
        return f"<a href='https://www.youtube.com/results?search_query={query}' target='_blank' style='color: #00A36C; text-decoration: none;'>Listen ▷</a>"
        
    playlist_df['Listen'] = playlist_df.apply(create_youtube_link, axis=1)
    return playlist_df

# --- Real-Time Webcam Class ---
class EmotionVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = load_face_detector()
        self.emotion_model = load_emotion_model()
        self.current_emotion = "neutral" 

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        if self.face_cascade is None or self.emotion_model is None:
            return frame

        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            roi_norm = roi_gray / 255.0
            roi_3ch = np.repeat(np.expand_dims(roi_norm, axis=-1), 3, axis=-1)
            roi_final = np.expand_dims(roi_3ch, axis=0)
            
            prediction = self.emotion_model.predict(roi_final, verbose=0)
            emotion_index = np.argmax(prediction)
            self.current_emotion = EMOTION_MAP.get(emotion_index, "Unknown")
            
            cv2.putText(img, self.current_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            self.current_emotion = "neutral"

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI ---

# Load the custom CSS
load_css()

st.title("🎵 MoodMate | Emotion-Based Music Recommender")
st.markdown("Your personal DJ that curates a playlist based on your facial expression.")

tab1, tab2 = st.tabs(["📁 Upload an Image", "📷 Live Webcam Detection"])

# --- Tab 1: Upload an Image ---
with tab1:
    # Use st.markdown to apply the custom glass-container class
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    st.header("Get a Playlist from an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        image_np = np.array(image)

        with st.spinner('Analyzing emotion and curating your playlist...'):
            emotion_model = load_emotion_model()
            if emotion_model is not None:
                if len(image_np.shape) > 2 and image_np.shape[2] in [3, 4]:
                    gray_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray_img = image_np
                
                img_resized = cv2.resize(gray_img, (48, 48))
                img_normalized = img_resized / 255.0
                img_3_channel = np.repeat(np.expand_dims(img_normalized, axis=-1), 3, axis=-1)
                img_final = np.expand_dims(img_3_channel, axis=0)
                
                prediction = emotion_model.predict(img_final, verbose=0)
                emotion_index = np.argmax(prediction)
                detected_emotion = EMOTION_MAP.get(emotion_index, "Unknown")
                
                # --- 3. STYLISH UI: Custom styled emotion box ---
                st.markdown(f'<div class="emotion-box"><h2>Emotion Detected: {detected_emotion.upper()}</h2></div>', unsafe_allow_html=True)
                
                if detected_emotion == 'happy':
                    st.balloons()

                playlist = recommend_songs(detected_emotion)
                
                if not playlist.empty:
                    st.header("🎶 Here's Your Personalized Playlist:")
                    # --- 4. INTERACTIVE FEATURE: Render table with clickable HTML links ---
                    st.markdown(
                        playlist[['artist_name', 'title', 'Listen']].to_html(escape=False, index=False),
                        unsafe_allow_html=True
                    )
    
    st.markdown('</div>', unsafe_allow_html=True) # Close the glass-container

# --- Tab 2: Live Webcam Detection ---
with tab2:
    st.header("Live Emotion Playlist Generator")
    st.info("Click 'Start' to activate your webcam. Your playlist will update in real-time below!")

    col1, col2 = st.columns([2, 1]) # Give webcam more space

    with col1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="webcam",
            video_processor_factory=EmotionVideoTransformer,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False}
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.header("🎶 Your Live Playlist")
        playlist_placeholder = st.empty()
        
        if 'last_emotion' not in st.session_state:
            st.session_state.last_emotion = 'neutral'

        while ctx.state.playing:
            if ctx.video_transformer:
                current_emotion = ctx.video_transformer.current_emotion
                
                if current_emotion != st.session_state.last_emotion:
                    st.session_state.last_emotion = current_emotion
                    
                    if current_emotion == 'happy':
                        st.balloons()
                    
                    playlist = recommend_songs(current_emotion)
                    
                    with playlist_placeholder.container():
                        st.markdown(f'<div class="emotion-box" style="background-color: rgba(0, 100, 150, 0.7);"><h2>{current_emotion.upper()}</h2></div>', unsafe_allow_html=True)
                        st.markdown(
                            playlist[['artist_name', 'title', 'Listen']].to_html(escape=False, index=False),
                            unsafe_allow_html=True
                        )
                
            time.sleep(1) # Refresh every second
        
        st.markdown('</div>', unsafe_allow_html=True)