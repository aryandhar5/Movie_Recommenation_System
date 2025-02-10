import pandas as pd
import ast
from nltk.stem.porter import PorterStemmer

class MoviePreprocessor:
    """Handles data loading and preprocessing for content-based filtering."""

    def __init__(self, movies_path, credits_path, ratings_path):
        self.movies_path = movies_path
        self.credits_path = credits_path
        self.ratings_path = ratings_path
        self.ps = PorterStemmer()

    def load_data(self):
        """Loads and merges movie and credits datasets."""
        movies = pd.read_csv(self.movies_path)
        credits = pd.read_csv(self.credits_path)
        dataset = movies.merge(credits, on="title")
        dataset = dataset[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
        dataset.dropna(inplace=True)
        return dataset

    @staticmethod
    def convert(obj):
        """Extracts relevant information from JSON-like strings."""
        return [i["name"] for i in ast.literal_eval(obj)]

    def process_data(self, dataset):
        """Processes genres, keywords, cast, crew, and overview into tags."""
        dataset["genres"] = dataset["genres"].apply(self.convert)
        dataset["keywords"] = dataset["keywords"].apply(self.convert)
        dataset["cast"] = dataset["cast"].apply(lambda x: [i["name"] for i in ast.literal_eval(x)[:3]])
        dataset["crew"] = dataset["crew"].apply(lambda x: [i["name"] for i in ast.literal_eval(x) if i["job"] == "Director"])
        dataset["overview"] = dataset["overview"].apply(lambda x: x.split() if isinstance(x, str) else [])

        for col in ["genres", "keywords", "cast", "crew"]:
            dataset[col] = dataset[col].apply(lambda x: [i.replace(" ", "") for i in x])

        dataset["tags"] = dataset["overview"] + dataset["genres"] + dataset["keywords"] + dataset["cast"] + dataset["crew"]
        dataset["tags"] = dataset["tags"].apply(lambda x: " ".join(x).lower())

        dataset["tags"] = dataset["tags"].apply(lambda x: " ".join(self.ps.stem(word) for word in x.split()))

        return dataset[["movie_id", "title", "tags"]]

if __name__ == "__main__":
    preprocessor = MoviePreprocessor("data/tmdb_5000_movies.csv", "data/tmdb_5000_credits.csv", "data/ratings.csv")
    data = preprocessor.load_data()
    processed_data = preprocessor.process_data(data)
    processed_data.to_csv("data/processed_movies.csv", index=False)
