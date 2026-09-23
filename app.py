import streamlit as st
import os
from data_loader import load_data
from hybrid_model import HybridModel
from nlp_model import NLPSentimentModel
from database import get_or_create_user, get_user_ratings, get_all_db_ratings, merge_ratings, add_rating

st.set_page_config(page_title="Cinemate AI", layout="wide", initial_sidebar_state="expanded")

# Injecting Custom CSS for "UI/UX Pro Max" Aesthetics
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* Gradient Text for Main Title */
    .gradient-text {
        font-weight: 900;
        font-size: 3.5rem;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 10px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Glassmorphism Movie Cards */
    .movie-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 65, 108, 0.4);
        background: rgba(255, 255, 255, 0.06);
    }

    .movie-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    
    .movie-genres {
        font-size: 0.9rem;
        color: #FF416C;
        font-weight: 600;
        margin-bottom: 16px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .movie-score {
        display: inline-block;
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 6px 14px;
        border-radius: 30px;
        color: #fff;
        font-weight: 800;
        font-size: 1rem;
        box-shadow: 0 4px 10px rgba(56, 239, 125, 0.3);
    }
    
    .movie-score-secondary {
        display: inline-block;
        background: linear-gradient(135deg, #4A00E0, #8E2DE2);
        padding: 6px 14px;
        border-radius: 30px;
        color: #fff;
        font-weight: 800;
        font-size: 1rem;
        box-shadow: 0 4px 10px rgba(142, 45, 226, 0.3);
    }

    .explanation {
        font-size: 0.95rem;
        color: #CBD5E0;
        border-left: 4px solid #FF416C;
        padding-left: 12px;
        margin-top: 16px;
        font-style: italic;
        line-height: 1.5;
        background: rgba(255, 255, 255, 0.02);
        padding-top: 8px;
        padding-bottom: 8px;
        border-radius: 0 8px 8px 0;
    }
    
    /* Customizing Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px 8px 0px 0px;
        gap: 8px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def init_system():
    movies, ratings = load_data()
    model = HybridModel()
    
    if os.path.exists('models/cf_model.pkl') and os.path.exists('models/cb_model.pkl'):
        model.load('models/cf_model.pkl', 'models/cb_model.pkl', ratings)
    else:
        model.train(movies, ratings)
        if not os.path.exists('models'):
            os.makedirs('models')
        model.save('models/cf_model.pkl', 'models/cb_model.pkl')
        
    movie_dict = {m['movieId']: m for m in movies}
    all_movie_ids = list(movie_dict.keys())
    
    # Merge any DB ratings into the global dataset so global CF works better over time
    db_ratings = get_all_db_ratings()
    ratings = merge_ratings(ratings, db_ratings)
    user_ids = sorted(list(set(r['userId'] for r in ratings)))
    
    # Load NLP Model
    nlp_model = NLPSentimentModel()
    if os.path.exists('models/nlp_model.pkl'):
        nlp_model.load('models/nlp_model.pkl')
    
    return model, movie_dict, all_movie_ids, user_ids, ratings, nlp_model

# --- Header Section ---
st.markdown('<div class="gradient-text">Cinemate AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Next-Generation Hybrid Recommendation Engine</div>', unsafe_allow_html=True)

with st.spinner('Waking up the AI...'):
    model, movie_dict, all_movie_ids, user_ids, ratings, nlp_model = init_system()

# --- Sidebar Configuration (Auth) ---
if not st.user.is_logged_in:
    st.sidebar.markdown("### 🔐 Login")
    st.sidebar.write("Login with Google to get personalized movie recommendations!")
    if st.sidebar.button("Login with Google", type="primary"):
        st.login("google")
    st.sidebar.markdown("---")
    st.sidebar.info("You must log in to use the application.")
    st.stop()

user_email = st.user.email
st.sidebar.markdown(f"### 👋 Welcome back!")
st.sidebar.write(f"Logged in as **{user_email}**")
if st.sidebar.button("Logout"):
    st.logout()

# Link the Google email to our internal DB user ID
selected_user = get_or_create_user(user_email)

st.sidebar.markdown("---")
# User History Preview
st.sidebar.markdown(f"**Your Favorites**")
# Merge the ML dataset ratings (if this user was mapped to a default ID) with their DB ratings
db_ratings = get_user_ratings(selected_user)
user_ratings = merge_ratings([r for r in ratings if r['userId'] == selected_user], db_ratings)
user_ratings.sort(key=lambda x: x['timestamp'], reverse=True)
top_recent = [r for r in user_ratings if r['rating'] >= 4.0][:5]

if top_recent:
    for r in top_recent:
        if r['movieId'] in movie_dict:
            m = movie_dict[r['movieId']]
            st.sidebar.markdown(f"⭐ **{r['rating']}** | {m['title']}")
else:
    st.sidebar.write("No highly rated movies found. Rate some movies to get started!")

# --- Main Content Tabs ---
tab_discover, tab_search, tab_nlp = st.tabs(["✨ Discover", "🔍 Search Movies", "✍️ Sentiment Analyzer"])

# --- TAB 1: DISCOVER ---
with tab_discover:
    st.markdown("### 🎬 Top Picks For You")
    
    if st.button("Generate Magic Recommendations", type="primary"):
        with st.spinner("Analyzing billions of parameters..."):
            recs = model.recommend(selected_user, all_movie_ids, top_n=10)
            
            for i, (m_id, score) in enumerate(recs):
                m = movie_dict[m_id]
                
                # Logic for explanation
                cb_score = model.cb_model.predict(
                    model.user_history_cache[selected_user]['movies'],
                    model.user_history_cache[selected_user]['ratings'],
                    m_id
                ) if selected_user in model.user_history_cache else 3.0
                cf_score = model.cf_model.predict(selected_user, m_id)
                
                if cb_score > cf_score:
                    explanation = f"Because you loved similar movies (Content match: {cb_score:.1f}/5)"
                else:
                    explanation = f"Trending among users with your taste (Collaborative match: {cf_score:.1f}/5)"
                    
                # Card HTML
                card_html = f"""
                <div class="movie-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <div class="movie-title">#{i+1} {m['title']}</div>
                            <div class="movie-genres">{m['genres'].replace('|', ' • ')}</div>
                        </div>
                        <div class="movie-score">{score:.1f} / 5.0</div>
                    </div>
                    <div class="explanation">💡 {explanation}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.info("👈 Click the button above to discover your next favorite movie!")

# --- TAB 2: SEARCH ---
with tab_search:
    st.markdown("### 🔍 Explore the Database")
    
    # Create mapping from title to ID for the dropdown
    title_to_id = {m['title']: m['movieId'] for m in movie_dict.values()}
    movie_titles = [""] + sorted(list(title_to_id.keys()))
    
    selected_title = st.selectbox("Search and select a movie:", options=movie_titles, index=0)
    
    if selected_title:
        m_id = title_to_id[selected_title]
        m = movie_dict[m_id]
        
        # Check what rating this user would give to the searched movie
        predicted_score = model.predict(selected_user, m_id)
        
        card_html = f"""
        <div class="movie-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="movie-title">{m['title']}</div>
                    <div class="movie-genres">{m['genres'].replace('|', ' • ')}</div>
                </div>
                <div class="movie-score-secondary">Predicted for you: {predicted_score:.1f}⭐</div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        
        st.write("### Rate this movie")
        col1, col2 = st.columns([3, 1])
        with col1:
            new_rating = st.slider("Your rating", min_value=0.5, max_value=5.0, value=3.0, step=0.5, key=f"rate_{m_id}")
        with col2:
            st.write("") # spacer
            st.write("")
            if st.button("Submit Rating", key=f"btn_{m_id}"):
                add_rating(selected_user, m_id, new_rating)
                
                # Update the model's user_history_cache directly for immediate effect
                if selected_user not in model.user_history_cache:
                    model.user_history_cache[selected_user] = {'movies': [], 'ratings': []}
                model.user_history_cache[selected_user]['movies'].append(m_id)
                model.user_history_cache[selected_user]['ratings'].append(new_rating)
                
                st.success("Rating saved!")
                st.rerun()
        
        # Similar movies recommendation
        st.markdown("#### 🎬 Similar Movies You Might Like")
        
        target_idx = model.cb_model.movie_idx_map.get(m_id)
        if target_idx is not None:
            sim_scores = list(enumerate(model.cb_model.cosine_sim[target_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            top_similar = []
            for idx, score in sim_scores:
                if idx != target_idx:
                    top_similar.append((idx, score))
                if len(top_similar) >= 5:
                    break
            
            for idx, score in top_similar:
                sim_m_id = model.cb_model.idx_movie_map[idx]
                sim_m = movie_dict[sim_m_id]
                
                st.markdown(f"""
                <div style="padding: 10px; border-left: 3px solid #11998e; background: rgba(255,255,255,0.02); margin-bottom: 8px; border-radius: 0 8px 8px 0;">
                    <strong style="color: white; font-size: 1.1rem;">{sim_m['title']}</strong><br/>
                    <span style="color: #A0AEC0; font-size: 0.9rem;">{sim_m['genres'].replace('|', ' • ')}</span>
                </div>
                """, unsafe_allow_html=True)

# --- TAB 3: NLP SENTIMENT ---
with tab_nlp:
    st.markdown("### ✍️ IMDB Review Sentiment Analyzer")
    st.write("Powered by a custom Naive Bayes NLP model trained on the Stanford IMDB dataset.")
    
    review_text = st.text_area("Write a movie review:", placeholder="I absolutely loved this movie! The acting was phenomenal...", height=150)
    
    if st.button("Analyze Sentiment"):
        if not review_text.strip():
            st.warning("Please enter a review first.")
        elif not nlp_model.is_trained:
            st.error("NLP Model is not trained yet. Please wait for the background training task to complete or run `python train_nlp.py`.")
        else:
            with st.spinner("Analyzing text semantics..."):
                prob_pos = nlp_model.predict(review_text)
                
                if prob_pos >= 0.5:
                    st.success(f"**Positive Sentiment!** (Confidence: {prob_pos*100:.1f}%)")
                else:
                    st.error(f"**Negative Sentiment!** (Confidence: {(1-prob_pos)*100:.1f}%)")
                
                st.progress(float(prob_pos))
                st.caption("0 = Negative | 1 = Positive")
