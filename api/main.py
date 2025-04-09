from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()


df = pd.read_csv("shl_catalogue.csv")
df["combined_text"] = df["Assessment Name"] + " " + df["Skills/Tags"]
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["combined_text"])


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/recommend")
def recommend(data: QueryRequest):
    query_text = data.query
    keywords = [kw.strip() for kw in query_text.lower().split() if kw.strip()]

    if not keywords:
        return {"recommended_assessments": []}

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
            "url": row["URL"],
            "adaptive_support": row["Adaptive Support"],
            "description": row["Assessment Name"],  
            "duration": int(row["Duration (min)"]),
            "remote_support": row["Remote Testing Support"],
            "test_type": [row["Test Type"]] if pd.notna(row["Test Type"]) else []
        })

    return {"recommended_assessments": recommended}



# to run the API run the below line in terminal
# uvicorn api.main:app --reload

