from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Load and prepare dataset
df = pd.read_csv("shl_catalogue.csv")
df["combined_text"] = df["Assessment Name"] + " " + df["Skills/Tags"]
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["combined_text"])

class JDRequest(BaseModel):
    job_description: str

@app.post("/recommend")
async def recommend(data: JDRequest):
    jd_text = data.job_description
    keywords = [kw.strip() for kw in jd_text.lower().replace(",", " ").split() if kw.strip()]

    if not keywords:
        return {"message": "No valid keywords found in your input."}

    sim_scores = []
    for kw in keywords:
        query_vec = tfidf.transform([kw])
        sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
        sim_scores.append(sim)

    final_similarity = np.mean(sim_scores, axis=0)
    top_indices = final_similarity.argsort()[::-1][:10]
    results = df.iloc[top_indices].copy()
    results["Similarity Score"] = final_similarity[top_indices].round(2)
    dynamic_threshold = max(final_similarity[top_indices]) * 0.7
    best_matches = results[results["Similarity Score"] >= dynamic_threshold]

    if len(best_matches) < 10:
        best_matches = results.iloc[:4]

    best_matches = best_matches[[
        "Assessment Name",
        "Remote Testing Support",
        "Adaptive Support",
        "Duration (min)",
        "Test Type",
        "Similarity Score",
        "URL"
    ]]

    return best_matches.to_dict(orient="records")
