import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os

# Page configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .header {
        color: #FF4B4B;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .movie-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        transition: transform 0.3s ease;
    }
    .movie-card:hover {
        transform: translateY(-5px);
    }
    .movie-title {
        color: white;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
        margin: 10px 0;
    }
    .similarity-score {
        color: #00FF9D;
        font-weight: bold;
        text-align: center;
        font-size: 16px;
    }
    .genre-badge {
        background-color: #4B56D2;
        color: white;
        padding: 3px 8px;
        border-radius: 15px;
        font-size: 12px;
        margin-right: 5px;
        margin-bottom: 5px;
        display: inline-block;
    }
    .divider {
        height: 3px;
        background: linear-gradient(90deg, #FF4B4B, #00FF9D);
        margin: 25px 0;
        border-radius: 3px;
    }
    .poster {
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Load movie data (with handling for large files)
def load_movies():
    # Check if we have a processed version already
    if os.path.exists('movies_processed.pkl'):
        return pickle.load(open('movies_processed.pkl', 'rb'))
    
    try:
        # Try loading original data
        movies = pickle.load(open('movie.pkl', 'rb'))
        
        # Limit to 1000 movies if dataset is large
        if len(movies) > 1000:
            if 'popularity' in movies.columns:
                movies = movies.sort_values('popularity', ascending=False).head(1000)
            else:
                movies = movies.head(1000)
        
        # Save processed version
        pickle.dump(movies, open('movies_processed.pkl', 'wb'))
        return movies
    
    except Exception as e:
        st.error(f"Error loading movie data: {e}")
        return None

# Compute movie similarity
def compute_similarity(movies):
    # Check if we have computed similarity already
    if os.path.exists('similarity_processed.pkl'):
        return pickle.load(open('similarity_processed.pkl', 'rb'))
    
    try:
        # Create tags if not present
        if 'tags' not in movies.columns:
            # Create feature tags for similarity calculation
            movies['tags'] = ''
            for feature in ['overview', 'genres', 'keywords', 'cast', 'crew']:
                if feature in movies.columns:
                    if isinstance(movies[feature].iloc[0], list):
                        movies['tags'] += movies[feature].apply(lambda x: ' '.join(x) if x else '')
                    else:
                        movies['tags'] += movies[feature].astype(str) + ' '
                        
            # Clean tags
            movies['tags'] = movies['tags'].apply(lambda x: x.lower())
        
        # Calculate similarity
        cv = CountVectorizer(max_features=2000, stop_words='english')
        vectors = cv.fit_transform(movies['tags']).toarray()
        similarity = cosine_similarity(vectors)
        
        # Save for future use
        pickle.dump(similarity, open('similarity_processed.pkl', 'wb'))
        return similarity
    
    except Exception as e:
        st.error(f"Error computing similarity: {e}")
        return None

# Function to fetch movie poster
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        response = requests.get(url)
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
        return None
    except:
        return None

# Get movie recommendations
def get_recommendations(movie, movies_df, similarity_matrix, count=6):
    if movie not in movies_df['title'].values:
        return []
    
    # Get movie index
    idx = movies_df[movies_df['title'] == movie].index[0]
    
    # Calculate similarity scores
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Skip first movie (itself) and get top recommendations
    top_movies = similarity_scores[1:count+1]
    
    # Get movie details
    recommendations = []
    for i, score in top_movies:
        movie_info = {
            'id': movies_df.iloc[i]['movie_id'],
            'title': movies_df.iloc[i]['title'],
            'similarity': round(score * 100, 1),
            'poster': fetch_poster(movies_df.iloc[i]['movie_id'])
        }
        
        # Add genres if available
        if 'genres' in movies_df.columns:
            if isinstance(movies_df.iloc[i]['genres'], list):
                movie_info['genres'] = movies_df.iloc[i]['genres']
            else:
                movie_info['genres'] = []
        else:
            movie_info['genres'] = []
        
        # Add overview if available
        if 'overview' in movies_df.columns:
            if isinstance(movies_df.iloc[i]['overview'], list):
                movie_info['overview'] = " ".join(movies_df.iloc[i]['overview'])
            else:
                movie_info['overview'] = str(movies_df.iloc[i]['overview'])
        else:
            movie_info['overview'] = ""
        
        recommendations.append(movie_info)
    
    return recommendations

# Main app function
def main():
    # App header
    st.markdown('<div class="header">🎬 Movie Recommender System</div>', unsafe_allow_html=True)
    
    # Load movie data
    with st.spinner("Loading movie data..."):
        movies = load_movies()
    
    if movies is None:
        st.error("Failed to load movie data. Please make sure 'movie.pkl' exists in the directory.")
        return
    
    # Calculate similarity
    with st.spinner("Computing movie similarities..."):
        similarity = compute_similarity(movies)
    
    if similarity is None:
        st.error("Failed to compute movie similarities.")
        return
    
    # Movie selection dropdown
    movie_list = sorted(movies['title'].values)
    selected_movie = st.selectbox("Select a movie you like:", movie_list)
    
    # Get recommendations button
    if st.button("Show Recommendations", type="primary"):
        with st.spinner("Finding similar movies..."):
            # Get recommendations
            recommendations = get_recommendations(selected_movie, movies, similarity)
            
            if not recommendations:
                st.error(f"Couldn't find recommendations for '{selected_movie}'")
                return
            
            # Display selected movie details
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            try:
                # Get selected movie info
                movie_row = movies[movies['title'] == selected_movie].iloc[0]
                movie_id = movie_row['movie_id']
                poster = fetch_poster(movie_id)
                
                # Show movie details
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if poster:
                        st.image(poster, width=220, caption="", use_container_width=False, clamp=True, 
                               output_format="JPEG")
                    else:
                        st.image("https://via.placeholder.com/220x330?text=No+Poster", width=220)
                
                with col2:
                    st.markdown(f"<h1>{selected_movie}</h1>", unsafe_allow_html=True)
                    
                    # Show genres
                    if 'genres' in movies.columns and isinstance(movie_row['genres'], list):
                        st.markdown("**Genres:**")
                        genre_html = ""
                        for genre in movie_row['genres']:
                            genre_html += f"<span class='genre-badge'>{genre}</span>"
                        st.markdown(genre_html, unsafe_allow_html=True)
                    
                    # Show overview
                    if 'overview' in movies.columns:
                        st.markdown("<br>**Overview:**", unsafe_allow_html=True)
                        if isinstance(movie_row['overview'], list):
                            overview = " ".join(movie_row['overview'])
                        else:
                            overview = str(movie_row['overview'])
                        st.write(overview)
            
            except Exception as e:
                st.warning(f"Could not display details for {selected_movie}: {e}")
            
            # Display recommendations
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.subheader("Recommended Movies")
            
            # Create columns for recommendations
            cols = st.columns(3)
            
            # Display each recommendation
            for i, movie in enumerate(recommendations):
                with cols[i % 3]:
                    st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
                    
                    # Display poster
                    if movie['poster']:
                        st.image(movie['poster'], use_container_width=True, output_format="JPEG", clamp=True)
                    else:
                        st.image("https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)
                    
                    # Display title and similarity
                    st.markdown(f"<div class='movie-title'>{movie['title']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='similarity-score'>Similarity: {movie['similarity']}%</div>", unsafe_allow_html=True)
                    
                    # Display genres
                    if movie['genres']:
                        genres_html = "<div style='text-align:center; margin-top:10px;'>"
                        for genre in movie['genres'][:3]:  # Show up to 3 genres
                            genres_html += f"<span class='genre-badge'>{genre}</span>"
                        genres_html += "</div>"
                        st.markdown(genres_html, unsafe_allow_html=True)
                    
                    # Add overview in expander
                    if movie['overview']:
                        with st.expander("Show overview"):
                            st.write(movie['overview'])
                    
                    st.markdown("</div>", unsafe_allow_html=True)
    
    # About section
    with st.expander("About this recommender system"):
        st.write("""
        This movie recommender system uses content-based filtering to suggest movies similar to your selection.
        
        **How it works:**
        - Analyzes movie features (plot, genres, cast, keywords)
        - Creates a mathematical representation of each movie
        - Uses cosine similarity to find movies with similar characteristics
        - Shows the most similar movies based on content
        
        **Data:** TMDB 5000 Movies Dataset
        """)

# Run the app
if __name__ == "__main__":
    main() 