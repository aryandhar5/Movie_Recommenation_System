import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    """A content-based movie recommender system with movie posters and YouTube trailers."""

    def __init__(self, data_path, tmdb_api_key, youtube_api_key):
        self.data_path = data_path
        self.tmdb_api_key = tmdb_api_key
        self.youtube_api_key = youtube_api_key
        self.movies = pd.read_csv(self.data_path)
        self.vectorizer = CountVectorizer(max_features=5000, stop_words="english")
        self.similarity = None
        self.train()

    def train(self):
        """Trains the recommendation model using CountVectorizer and cosine similarity."""
        vectors = self.vectorizer.fit_transform(self.movies["tags"]).toarray()
        self.similarity = cosine_similarity(vectors)

    def fetch_poster(self, movie_title):
        """Fetch movie poster URL using TMDB API."""
        base_url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": self.tmdb_api_key, "query": movie_title}
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data["results"]:
                poster_path = data["results"][0]["poster_path"]
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return "https://via.placeholder.com/500x750?text=No+Image"

    def fetch_trailer(self, movie_title):
        """Fetch YouTube trailer URL using YouTube API."""
        search_url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": f"{movie_title} official trailer",
            "key": self.youtube_api_key,
            "maxResults": 1,
            "type": "video"
        }
        
        response = requests.get(search_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if "items" in data and len(data["items"]) > 0:
                video_id = data["items"][0]["id"]["videoId"]
                return f"https://www.youtube.com/watch?v={video_id}"
        return "No Trailer Found"

    def recommend(self, movie_title):
        """Recommends top 5 similar movies, posters, and trailers."""
        if movie_title not in self.movies["title"].values:
            return [], [], []

        movie_index = self.movies[self.movies["title"] == movie_title].index[0]
        distances = self.similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = [self.movies.iloc[i[0]]["title"] for i in movie_list]
        posters = [self.fetch_poster(movie) for movie in recommended_movies]
        trailers = [self.fetch_trailer(movie) for movie in recommended_movies]

        return recommended_movies, posters, trailers

if __name__ == "__main__":
    TMDB_API_KEY = "eb2c05fba9acb42e7de3a72487b21633"  # Replace with your actual API key
    YOUTUBE_API_KEY = "AIzaSyDWVwqZ5N5_FBik3gY46xMR6kCfTczq_jM"  # Replace with your actual API key
    recommender = MovieRecommender("data/processed_movies.csv", TMDB_API_KEY, YOUTUBE_API_KEY)
    movies, posters, trailers = recommender.recommend("Batman Begins")
    print(movies)
    print(posters)
    print(trailers)