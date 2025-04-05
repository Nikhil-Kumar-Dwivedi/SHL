# 🧠 SHL Assessment Recommendation Engine

This project is part of an internship assignment aimed at building a smart recommendation system for SHL’s assessment catalogue. It uses NLP techniques to match job descriptions with the most relevant assessments based on their skills and attributes.

---

## 📌 Problem Statement

Given a job description (JD), suggest the most relevant assessments from SHL’s catalogue that best match the role's required skills and expectations. The solution must support dynamic updates (add/delete assessments) and optionally provide backend API support.

---

## 🚀 Features

- ✅ Upload or paste Job Description or Keywords
- 🔎 Smart matching using TF-IDF and Cosine Similarity
- 🎯 Highlights the best matches
- 🧠 Shows other potential matches in expandable view
- 🧑‍💻 Admin Panel to:
  - ➕ Add New Assessments to the CSV file
  - ❌ Delete Existing Assessments
- ⚡ Optionally connect to a FastAPI backend
- 💾 Data persistence with `shl_catalogue.csv`
- 📈 Dynamic similarity thresholding
- 📑 CSV-based assessment catalogue for easy management

---

## 🛠️ Tech Stack

| Layer         | Tools & Libraries             |
|---------------|-------------------------------|
| Frontend      | Streamlit                     |
| Backend       | FastAPI (Optional)            |
| Data Handling | Pandas                        |
| NLP/ML        | Scikit-learn (TF-IDF, Cosine Similarity), NumPy |
| Deployment    | Streamlit Cloud / Localhost   |

---

## 🧠 How It Works

1. **Data Source**: A CSV file (`shl_catalogue.csv`) containing assessment details like name, tags, support, duration, and type.
2. **Text Processing**: TF-IDF Vectorization of `Assessment Name + Skills/Tags`.
3. **Similarity Check**: Job description is tokenized, vectorized, and compared using cosine similarity.
4. **Best Matches**: Top results are filtered using a dynamic threshold (70% of highest score).
5. **Admin Panel**: Enables adding or deleting assessments directly from UI with instant CSV updates.

---

## 📂 Project Structure


---

## 📡 API (Optional Backend)

If FastAPI backend is activated:

- **Endpoint**: `POST /recommend`
- **Request**:
```json
{ "job_description": "Data science, Python, machine learning" }


[
  {
    "Assessment Name": "Data Scientist - Tech",
    "Similarity Score": 0.86,
    ...
  },
  ...
]


