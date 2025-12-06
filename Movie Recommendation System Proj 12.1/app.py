import os
import time
import streamlit as st
import pickle
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "f73fb13ec097ce6fd0fc412cbd21b201")


def requests_session_with_retries(total_retries=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)):
    session = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST"])
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

session = requests_session_with_retries()

FALLBACK_POSTER = "https://via.placeholder.com/500x750?text=No+Poster"

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    """
    Fetch poster URL from TMDB, with retries, timeout and fallback.
    Cached by Streamlit to avoid repeated API calls.
    """
    if not movie_id:
        return FALLBACK_POSTER

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY}
    try:
        resp = session.get(url, params=params, timeout=5)  # short timeout
        resp.raise_for_status()
        data = resp.json()
        poster_path = data.get("poster_path")
        if poster_path:
            # use a reasonable size instead of 'original' to save bandwidth
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            return FALLBACK_POSTER
    except requests.exceptions.HTTPError as e:
        # HTTP errors (4xx,5xx)
        st.warning(f"TMDB HTTP error for id {movie_id}: {e}")
    except requests.exceptions.Timeout:
        st.warning(f"Timeout when fetching poster for movie id {movie_id}")
    except requests.exceptions.ConnectionError as e:
        st.warning(f"Connection error when fetching poster for movie id {movie_id}: {e}")
    except Exception as e:
        st.warning(f"Unexpected error when fetching poster for movie id {movie_id}: {e}")
    return FALLBACK_POSTER

def recommend(movie, movies_df, similarity_matrix, k=5):
    movie_index = movies_df[movies_df['title'] == movie].index[0]
    distances = similarity_matrix[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:k+1]
    recommended_movies = []
    recommended_movies_posters = []

    for i in movies_list:
        movie_id = movies_df.iloc[i[0]].movie_id
        recommended_movies.append(movies_df.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_movies_posters

# Load pickles (you can also cache these)
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

st.title("Movie Recommender System")
selected_movie_name = st.selectbox('Select your favorite movie:', movies['title'].values)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name, movies, similarity)
    cols = st.columns(len(names))
    for col, name, poster in zip(cols, names, posters):
        with col:
            st.text(name)
            st.image(poster, use_container_width=True)
