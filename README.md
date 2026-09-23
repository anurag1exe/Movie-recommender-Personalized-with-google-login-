# 🎬 Cinemate AI - Hybrid Movie Recommendation Engine

**Cinemate AI** is a next-generation, AI-powered movie recommendation platform built with Streamlit. It combines the power of collaborative filtering and content-based recommendation techniques into a hybrid model to provide highly personalized movie suggestions. 

With a premium glassmorphism UI, real-time dynamic ratings, and Google OAuth integration, Cinemate AI goes beyond simple data visualization to offer a full-stack, personalized web application experience.

---

## ✨ Key Features

- **🧠 Hybrid AI Recommendation Engine**: 
  - **Collaborative Filtering**: Discovers movies based on the tastes of similar users.
  - **Content-Based Filtering**: Recommends movies based on genres and metadata of what you already love.
- **🔐 Secure Google OAuth Login**: Seamlessly log in using your Google account. Your session and data are securely managed.
- **📊 Real-time Dynamic Ratings**: Search for any movie, rate it from 0.5 to 5.0 stars, and watch your personalized recommendations update instantly.
- **✍️ NLP Sentiment Analyzer**: A built-in Naive Bayes model trained on IMDB data to analyze the sentiment of any movie review you type in real-time.
- **💎 Premium UI/UX**: Custom CSS featuring glassmorphism cards, dynamic gradient text, and smooth animations that break the mold of standard Streamlit apps.

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Python) with custom CSS injection.
- **Machine Learning**: `scikit-learn` (Cosine Similarity, Naive Bayes), `scikit-surprise` (SVD Collaborative Filtering).
- **Authentication**: `Authlib` & `httpx` (Streamlit Native OAuth).
- **Database**: SQLite3 (Local dynamic user and rating storage).
- **Environment**: `uv` (Modern Python Package Management).

## 🚀 Getting Started Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/cinemate-ai.git
cd cinemate-ai
```

### 2. Set up your Virtual Environment (using `uv`)
```bash
# Initialize uv and install dependencies
uv venv
uv pip install -r requirements.txt
```

### 3. Configure Google OAuth Secrets
You must configure your OAuth credentials for the login to work:
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new Web Application OAuth 2.0 Client ID.
3. Add `http://localhost:8501/oauth2callback` to your **Authorized redirect URIs**.
4. Create a `.streamlit/secrets.toml` file in the project root:
```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "a_random_secure_string_here"

[auth.google]
client_id = "YOUR_GOOGLE_CLIENT_ID"
client_secret = "YOUR_GOOGLE_CLIENT_SECRET"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

### 4. Run the Application
```bash
python -m streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

## ☁️ Deploying to Streamlit Community Cloud

If you choose to deploy this to [Streamlit Community Cloud](https://share.streamlit.io/):
1. Push your repository to GitHub (ensure `.streamlit/secrets.toml` and `database.db` are in your `.gitignore`!).
2. Connect your repository on the Streamlit dashboard.
3. Paste the contents of your `secrets.toml` into the **Advanced settings > Secrets** block.
4. **Important**: Update the `redirect_uri` in both the Streamlit secrets AND your Google Cloud Console to match your new production URL (e.g., `https://your-app-name.streamlit.app/oauth2callback`).

*Note: Streamlit Community Cloud occasionally reboots containers. Because this project uses a local SQLite database (`database.db`), user ratings will be wiped upon reboot. For a production-ready application, swap out the SQLite implementation in `database.py` for a cloud database like Firebase, Supabase, or PostgreSQL.*

---
*Built with ❤️ using Python & Streamlit.*
