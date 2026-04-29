import kagglehub
import os
import numpy as np
import pandas as pd
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download latest version
path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
df = pd.read_csv(os.path.join(path, "imdb_top_1000.csv"))

for col in ['Overview', 'Genre', 'Director', 'Star1', 'Star2']:
    df[col] = df[col].fillna('')
# Weighted feature string — genre and director repeated for emphasis
df['features'] = (
    df['Overview'] + ' ' +
    (df['Genre'] + ' ') * 9 +
    (df['Director'] + ' ') * 5 +
    df['Star1'] + ' ' + df['Star2']
)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

title = input("Enter a movie title: ")
def recommender(title, df, cosine_sim, n=3):
    # Check if exact match exists
    matching_movies = df[df['Series_Title'] == title]
    
    if matching_movies.empty:
        # Try case-insensitive search
        matching_movies = df[df['Series_Title'].str.lower() == title.lower()]
        
    if matching_movies.empty:
        print(f"Movie '{title}' not found.")
        print("\nAvailable movies containing similar keywords:")
        similar = df[df['Series_Title'].str.contains(title, case=False, na=False)]['Series_Title'].head(5)
        if len(similar) > 0:
            for i, movie in enumerate(similar, 1):
                print(f"  {i}. {movie}")
        return None
    #print(matching_movies) Returns the row for the matching movie from the csv
    
    idx = matching_movies.index[0]
    #print(idx) # Returns the index of the matching movie in the dataframe
    # Get similarity scores for that movie vs all others
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort by score, descending — skip index 0 (the movie itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    # Return top-n movie titles
    movie_indices = [i[0] for i in sim_scores]
    return df['Series_Title'].iloc[movie_indices].tolist()

result = recommender(title, df, cosine_sim)
if result:
    print(f"\nMovies similar to '{title}':")
    for i, movie in enumerate(result, 1):
        print(f"  {i}. {movie}")