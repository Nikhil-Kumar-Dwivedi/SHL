from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/recommend")
def recommend(data: QueryRequest):
    
    df = pd.read_csv("shl_catalogue.csv")
    df["combined_text"] = df["Assessment Name"] + " " + df["Skills/Tags"]
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["combined_text"])

    query_text = data.query
    keywords = [kw.strip() for kw in query_text.lower().split() if kw.strip()]

    if not keywords:
        return []

    sim_scores = []
    for kw in keywords:
        query_vec = tfidf.transform([kw])
        sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
        sim_scores.append(sim)

    final_similarity = np.mean(sim_scores, axis=0)
    top_indices = final_similarity.argsort()[::-1][:10]

    recommended = []
    for idx in top_indices:
        row = df.iloc[idx]
        recommended.append({
            "Assessment Name": row["Assessment Name"],
            "Remote Testing Support": row["Remote Testing Support"],
            "Adaptive Support": row["Adaptive Support"],
            "Duration (min)": int(row["Duration (min)"]),
            "Test Type": row["Test Type"],
            "Similarity Score": round(final_similarity[idx], 2),
            "URL": row.get("URL", "")
        })

    return recommended










# to run the API run the below line in terminal
# uvicorn api.main:app --reload

