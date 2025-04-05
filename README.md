# SHL

# 🧠 SHL Assessment Recommendation Engine

This is a smart recommendation engine built using Streamlit and Machine Learning (TF-IDF + Cosine Similarity) to suggest the most relevant SHL assessments based on a job description or keywords.

## 🔍 Features

- 💡 Upload or paste job descriptions
- 🤖 Get top matching assessments using NLP
- 🛠️ Admin Panel: Add/Delete assessments dynamically
- ⚙️ Backend option using FastAPI for scalability
- 💾 Stores data in a persistent CSV file (`shl_catalogue.csv`)

## 🛠️ Tools & Tech Stack

- **Frontend**: Streamlit
- **Backend** (Optional): FastAPI
- **ML/NLP**: Scikit-learn (TF-IDF, Cosine Similarity)
- **Data Handling**: Pandas
- **Deployment**: Streamlit Cloud

## 📦 Installation & Run Locally

```bash
# Clone this repo
git clone https://github.com/yourusername/shl-recommendation-engine.git
cd shl-recommendation-engine

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
