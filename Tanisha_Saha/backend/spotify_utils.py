import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import random

def create_spotify_client(client_id, client_secret):
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

def get_track_details(sp, uri):
    track = sp.track(uri)
    info = {
        "name": track["name"],
        "artist": track["artists"][0]["name"],
        "album": track["album"]["name"],
        "url": track["external_urls"]["spotify"]
    }
    return info

def load_mood_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "uri" in df.columns and "mood" in df.columns:
        mapping = dict(zip(df["uri"], df["mood"]))
        return mapping
    else:
        raise ValueError("CSV must have columns: 'uri' and 'mood'")

def get_random_songs_by_mood(mood_map, emotion, count=5):
    """Get random song URIs for a specific emotion"""
    songs_with_emotion = [uri for uri, mood in mood_map.items() if mood == emotion]
    if len(songs_with_emotion) < count:
        return songs_with_emotion  # Return all available songs if less than requested
    return random.sample(songs_with_emotion, count)
