import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# Suppress warnings from sklearn
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Configuration ---\
# --- Paths ---\
# Input files required for this script
MUSIC_DATA_PATH = 'cleaned_music_sentiment_dataset.csv'

# --- Output Files ---\
# These are the final files the backend_app.py will use
OUTPUT_CSV_PATH = 'new_dataset_with_emotions.csv'
OUTPUT_SIMILARITY_MATRIX_PATH = 'new_dataset_similarity_matrix.npy'
OUTPUT_INDICES_PATH = 'new_dataset_song_indices.pkl'
OUTPUT_PLOT_PATH = 'new_dataset_emotion_distribution.png'

# --- 2. Main Execution Block ---\
if __name__ == '__main__':
    # --- Phase 1: Load and Prepare New Music Data ---\
    print("\n--- Phase 1: Loading and Preparing Music Data ---")
    df_music = pd.read_csv(MUSIC_DATA_PATH)
    
    # Drop rows where key music/emotion data is missing
    key_cols = ['User_Text', 'Sentiment_Label', 'Song_Name', 'Artist', 
                'Genre', 'Tempo (BPM)', 'Mood', 'Energy', 'Danceability']
    df_music.dropna(subset=key_cols, inplace=True)
    
    # Ensure there are no duplicate songs
    df_music.drop_duplicates(subset=['Song_Name', 'Artist'], inplace=True, keep='first')
    df_music.reset_index(drop=True, inplace=True)

    print(f"Loaded and filtered dataset with {len(df_music)} unique songs.")

    # --- Phase 2: Map Emotion Labels ---
    print("\n--- Phase 2: Mapping Emotion Labels ---")
    # This map is useful for encoding the label, but the string label is also needed
    emotion_map = {'Sad': 0, 'Happy': 1, 'Relaxed': 2, 'Motivated': 3, 'Calm': 4}
    
    # We rename 'predicted_emotion' to 'emotion_label_encoded' for clarity.
    # This is an ENCODING of the *existing label*, not a *prediction*.
    df_music['emotion_label_encoded'] = df_music['Sentiment_Label'].map(emotion_map)
    
    # Save this cleaned-up and encoded dataset
    df_music.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved cleaned dataset with encoded labels to '{OUTPUT_CSV_PATH}'")

    # --- Phase 3: Visualize Emotion Distribution ---
    print("\n--- Phase 3: Visualizing Emotion Distribution ---")
    plt.figure(figsize=(10, 6))
    df_music['Sentiment_Label'].value_counts().plot(kind='bar', color='green')
    plt.title('Emotion Distribution from Dataset Labels')
    plt.xlabel('Emotion (from Sentiment_Label)')
    plt.ylabel('Number of Songs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH)
    print(f"Saved emotion distribution plot to '{OUTPUT_PLOT_PATH}'")

    # --- Phase 4: Build Content-Based Recommendation Engine ---
    print("\n--- Phase 4: Building Content-Based Recommendation Engine ---")
    print("This is the new, improved method.")

    # 1. Process Numerical Features
    print("Processing numerical and ordinal features...")
    scaler = MinMaxScaler()
    numerical_features = pd.DataFrame(
        scaler.fit_transform(df_music[['Tempo (BPM)']]), 
        columns=['Tempo_Scaled']
    )
    
    # 2. Process Ordinal Features (Low, Medium, High)
    ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2}
    ordinal_features = pd.DataFrame({
        'Energy_Scaled': df_music['Energy'].map(ordinal_map).fillna(1), # Fill NaNs with 'Medium'
        'Danceability_Scaled': df_music['Danceability'].map(ordinal_map).fillna(1)
    })
    # Scale ordinal features as well
    ordinal_features = pd.DataFrame(
        scaler.fit_transform(ordinal_features),
        columns=ordinal_features.columns
    )

    # 3. Process Categorical Features (One-Hot Encoding)
    print("Processing categorical features (Genre, Mood, Sentiment)...")
    genre_features = pd.get_dummies(df_music['Genre'], prefix='genre')
    mood_features = pd.get_dummies(df_music['Mood'], prefix='mood')
    sentiment_features = pd.get_dummies(df_music['Sentiment_Label'], prefix='sentiment')

    # 4. Combine all features into one matrix
    print("Combining all features into final matrix...")
    features_df = pd.concat([
        numerical_features,
        ordinal_features,
        genre_features,
        mood_features,
        sentiment_features
    ], axis=1)

    print(f"Created feature matrix with shape: {features_df.shape}")

    # 5. Calculate Cosine Similarity
    # This matrix is built from *music features*, not User_Text
    print("Calculating cosine similarity matrix based on content features...")
    cosine_sim = cosine_similarity(features_df, features_df)
    
    # 6. Save the similarity matrix
    np.save(OUTPUT_SIMILARITY_MATRIX_PATH, cosine_sim)
    print(f"Saved new similarity matrix to '{OUTPUT_SIMILARITY_MATRIX_PATH}'")
    
    # 7. Create and save the series for song title-to-index mapping (this is the same)
    indices = pd.Series(df_music.index, index=df_music['Song_Name']).drop_duplicates()
    with open(OUTPUT_INDICES_PATH, 'wb') as f:
        pickle.dump(indices, f)
    print(f"Saved song indices to '{OUTPUT_INDICES_PATH}'")
    
    print("\n--- All processing complete! ---")
    print("The new files are now ready for your backend application.")
