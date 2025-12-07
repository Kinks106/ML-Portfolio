import pickle
import streamlit as st
import numpy as np

st.header("Book Recommendation System using Collaborative Filtering")

# load pickles
model = pickle.load(open('books_recommender.pkl', 'rb'))
books_name = pickle.load(open('books_name.pkl', 'rb'))
final_ratings = pickle.load(open('final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('book_pivot.pkl', 'rb'))

def fetch_posters_by_indices(indices):

    poster_urls = []
    for idx in indices:
        # get title from pivot index
        try:
            title = book_pivot.index[int(idx)]
        except Exception:
            poster_urls.append(None)
            continue

        matches = final_ratings[final_ratings['title'] == title]
        if not matches.empty:
            poster_urls.append(matches.iloc[0].get('image_url', None))
        else:
            poster_urls.append(None)
    return poster_urls

def recommend_books(selected_title, n_recs=6):
    if selected_title not in book_pivot.index:
        return [], []

    book_idx = np.where(book_pivot.index == selected_title)[0][0]

    distances, suggestions = model.kneighbors(
        book_pivot.iloc[book_idx, :].values.reshape(1, -1),
        n_neighbors=n_recs
    )
    suggestion_indices = suggestions.flatten() 
    titles = []
    filtered_indices = []
    for idx in suggestion_indices:
        title = book_pivot.index[int(idx)]
        if title == selected_title:
            continue
        titles.append(title)
        filtered_indices.append(idx)

    max_show = 5
    titles = titles[:max_show]
    filtered_indices = filtered_indices[:max_show]

    poster_urls = fetch_posters_by_indices(filtered_indices)

    return titles, poster_urls


selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    books_name
)

if st.button("Recommend Books"):
    recommended_books, poster_url = recommend_books(selected_books, n_recs=6)

    if not recommended_books:
        st.warning("No recommendations found for this book.")
    else:
        n = len(recommended_books)
        cols = st.columns(n)
        for i, col in enumerate(cols):
            with col:
                st.text(recommended_books[i])
                if poster_url[i]:
                    st.image(poster_url[i])
                else:
                    st.write("No image available")
