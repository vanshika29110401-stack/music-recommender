
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="🎧 Music Recommender", layout="wide")

# Load Data
pred_matrix = np.load("pred_matrix.npy")
user_song_matrix = pd.read_csv("user_song_matrix.csv", index_col=0)
df_songs = pd.read_csv("enhanced_indian_songs.csv")

# Detect song title column automatically
possible_title_cols = ['Song Name', 'song_name', 'title', 'track', 'name']
song_title_col = None
for col in possible_title_cols:
    if col in df_songs.columns:
        song_title_col = col
        break
if song_title_col is None:
    raise KeyError("No column found for song titles. Expected one of: " + ", ".join(possible_title_cols))

# Optional columns
if 'genre_clean' not in df_songs.columns:
    df_songs['genre_clean'] = df_songs['genre'] if 'genre' in df_songs.columns else 'Unknown'
if 'album_cover_url' not in df_songs.columns:
    df_songs['album_cover_url'] = "https://via.placeholder.com/150"
if 'audio_preview_url' not in df_songs.columns:
    df_songs['audio_preview_url'] = ""

def recommend_songs(user_id, user_song_matrix, pred_matrix, df, N=10):
    if user_id in user_song_matrix.index:
        user_idx = user_song_matrix.index.tolist().index(user_id)
        user_preds = pred_matrix[user_idx]
        rated_songs = user_song_matrix.loc[user_id]
        rated_songs = rated_songs[rated_songs > 0].index.tolist()
        song_indices = np.argsort(user_preds)[::-1]

        # Ensure song_id column exists or use index if song_id is not available
        if 'song_id' in df.columns:
            rec_indices = [i for i in song_indices if df['song_id'].iloc[i] not in rated_songs][:N]
        else:
            rec_indices = [i for i in song_indices if df.index[i] not in rated_songs][:N]

        recommendations = df.iloc[rec_indices].copy()
        recommendations['predicted_rating'] = user_preds[rec_indices]
    else:
        # New user – top popular songs
        recommendations = df.sample(n=N, random_state=42).copy()
        recommendations['predicted_rating'] = np.random.rand(N) # Assign random ratings for new users
    return recommendations

# Sidebar Login
st.sidebar.header("Login")
email = st.sidebar.text_input("Enter your Email")

if not email:
    st.warning("⚠️ Please enter your email to continue.")
    st.stop()

valid_users = user_song_matrix.index.tolist()
existing_user = email in valid_users

if existing_user:
    st.sidebar.success(f"Logged in as: {email}")
else:
    st.sidebar.info("New user detected – default recommendations will be shown.")

# Sidebar filters
num_recs = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# Green style for multiselects (optional, move to separate CSS if preferred)
st.markdown("""
<style>.css-1d391kg .css-1n76uvr {border-color: green !important;}</style>
""", unsafe_allow_html=True)
genres_filter = st.sidebar.multiselect("Filter by Genre", options=df_songs['genre_clean'].unique())
moods_filter = st.sidebar.multiselect("Filter by Mood", options=df_songs['mood'].unique())

# Generate recommendations dynamically
recs = recommend_songs(email, user_song_matrix, pred_matrix, df_songs, N=num_recs)

# Apply filters dynamically
if genres_filter:
    recs = recs[recs['genre_clean'].isin(genres_filter)]
if moods_filter:
    recs = recs[recs['mood'].isin(moods_filter)]

# Display Recommendations
st.title("🎵 Personalized Music Recommendations")
st.subheader(f"Top {len(recs)} Recommendations for {email}")

if isinstance(recs, pd.DataFrame) and not recs.empty:
    for idx, row in recs.iterrows():
        col1, col2 = st.columns([1,4])
        with col1:
            st.image(row['album_cover_url'], width=80)
        with col2:
            st.markdown(f"**{row[song_title_col]}** by {row['Artists']}")
            st.markdown(f"*Genre:* {row['genre_clean']} | *Mood:* {row['mood']} | *Predicted Rating:* {row['predicted_rating']}")
            if row['audio_preview_url']:
                st.audio(row['audio_preview_url'])
elif isinstance(recs, pd.DataFrame) and recs.empty:
    st.info("No recommendations found matching your criteria.")
else:
    st.error("Error generating recommendations.")
st.write("Debug Info: Type of recs:", type(recs))
