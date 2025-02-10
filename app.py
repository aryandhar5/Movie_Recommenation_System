import streamlit as st
import pandas as pd
import sys
import sys
import os
import os
import sys

# Force install scikit-learn if missing
try:
    import sklearn
except ModuleNotFoundError:
    os.system('pip install --no-cache-dir scikit-learn')
    
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))  # Ensure 'src' is found
from src.recommender import MovieRecommender



# Load the recommender model with TMDB & YouTube API keys
TMDB_API_KEY = "eb2c05fba9acb42e7de3a72487b21633"  # Replace with your actual API key
YOUTUBE_API_KEY = "AIzaSyDWVwqZ5N5_FBik3gY46xMR6kCfTczq_jM"  # Replace with your actual API key
recommender = MovieRecommender("data/processed_movies.csv", TMDB_API_KEY, YOUTUBE_API_KEY)

# Streamlit UI
st.title("🎬 Movie Recommender System")
st.subheader("A Content-Based Movie Recommender with Posters & Trailers")

# Dropdown for movie selection
movie_list = recommender.movies["title"].values
selected_movie = st.selectbox("Choose a movie:", movie_list)

# Recommendation Button
if st.button("Recommend"):
    recommended_movies, posters, trailers = recommender.recommend(selected_movie)
    
    if recommended_movies:
        st.subheader("Recommended Movies:")
        cols = st.columns(5)  # Display 5 posters in a row
        for idx, col in enumerate(cols):
            with col:
                st.image(posters[idx], caption=recommended_movies[idx], width=150)
                st.markdown(f"[🎥 Watch Trailer]({trailers[idx]})", unsafe_allow_html=True)
    else:
        st.write("No recommendations found.")

# Allow Streamlit to run on all networks (important for Google Cloud)

