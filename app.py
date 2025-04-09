import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


st.set_page_config(page_title="SHL Assessment Recommendation Engine", layout="wide")




CSV_FILE = "shl_catalogue.csv"


@st.cache_data
def load_data():
    return pd.read_csv(CSV_FILE)

def load_fresh_data():  
    return pd.read_csv(CSV_FILE)

df = load_data()


st.title("🧠 SHL Assessment Recommendation Engine")
st.markdown("Enter a job description below and get smart assessment recommendations!")


with st.sidebar:
    st.header("🔐 Admin Panel")
    admin_action = st.radio("Choose Action", ["None", "Add Assessment", "Delete Assessment"])

    if admin_action == "Add Assessment":
        st.subheader("➕ Add New Assessment")
        name = st.text_input("Assessment Name")
        tags = st.text_input("Skills/Tags (comma-separated)")
        remote = st.selectbox("Remote Testing Support", ["Yes", "No"])
        adaptive = st.selectbox("Adaptive Support", ["Yes", "No"])
        duration = st.number_input("Duration (min)", min_value=1, step=1)
        test_type = st.text_input("Test Type")
        url = st.text_input("URL")

        if st.button("Add to Catalogue"):
            if name and tags and test_type:
                new_entry = {
                    "Assessment Name": name,
                    "Skills/Tags": tags,
                    "Remote Testing Support": remote,
                    "Adaptive Support": adaptive,
                    "Duration (min)": duration,
                    "URL": url,
                    "Test Type": test_type
                }
                try:
                    
                    current_df = load_fresh_data()

                    
                    updated_df = pd.concat([current_df, pd.DataFrame([new_entry])], ignore_index=True)
                    updated_df.to_csv(CSV_FILE, index=False)
                    st.success(f"✅ '{name}' added to catalogue!")
                except PermissionError:
                    st.error("❌ File permission denied. Please close the Excel file if it's open.")
            else:
                st.warning("Please fill all required fields.")

    elif admin_action == "Delete Assessment":
        st.subheader("🗑️ Delete an Assessment")
        
        fresh_df = load_fresh_data()
        assessment_to_delete = st.selectbox("Select Assessment to Delete", fresh_df["Assessment Name"].unique())

        if st.button("Delete from Catalogue"):
            fresh_df = fresh_df[fresh_df["Assessment Name"] != assessment_to_delete]
            fresh_df.to_csv(CSV_FILE, index=False)
            st.success(f"❌ '{assessment_to_delete}' has been removed from catalogue.")


jd_text = st.text_area("Paste Job Description or Keywords", height=200)



use_backend = st.checkbox("Use FastAPI Backend for Recommendations", value=False)

if st.button("Get Recommendations"):
    
    df = load_fresh_data()

    if jd_text.strip() == "":
        st.warning("Please enter a job description or keywords.")
    elif use_backend:
   
        with st.spinner("Fetching recommendations from backend..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/recommend",
                    json={"query": jd_text}  
                )

                if response.status_code == 200:
                    results = pd.DataFrame(response.json())

                    if results.empty:
                        st.info("No suitable assessments found.")
                    else:
                        results.insert(0, "S.No", range(1, len(results) + 1))
                        st.markdown("### 🎯 Best Match for Your Job Description")
                        st.dataframe(results[[ 
                            "S.No",
                            "Assessment Name",
                            "Remote Testing Support",
                            "Adaptive Support",
                            "Duration (min)",
                            "Test Type",
                            "Similarity Score"
                        ]], use_container_width=True, hide_index=True)
                else:
                    st.error(f"Backend error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")


    else:
        
        df["combined_text"] = df["Assessment Name"] + " " + df["Skills/Tags"]
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(df["combined_text"])

        
        keywords = [kw.strip() for kw in jd_text.lower().replace(",", " ").split() if kw.strip()]
        if not keywords:
            st.warning("No valid keywords found in your input.")
        else:
            sim_scores = []
            for kw in keywords:
                query_vec = tfidf.transform([kw])
                sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
                sim_scores.append(sim)

            
            final_similarity = np.mean(sim_scores, axis=0)

            
            top_indices = final_similarity.argsort()[::-1][:10]
            results = df.iloc[top_indices].copy().reset_index(drop=True)
            results["Similarity Score"] = final_similarity[top_indices].round(2)
            results["Assessment Name"] = results.apply(
                lambda row: f"{row['Assessment Name']}", axis=1
            )
            results.insert(0, "S.No", range(1, len(results) + 1))

            
            dynamic_threshold = max(final_similarity[top_indices]) * 0.7

            best_matches = results[results["Similarity Score"] >= dynamic_threshold]
            if len(best_matches) < 10:
                
                best_matches = results.iloc[:3]

            other_matches = results.drop(best_matches.index).reset_index(drop=True)

            
            st.markdown("### 🎯 Best Match for Your Job Description")
            st.dataframe(best_matches[[ 
                "S.No",
                "Assessment Name",
                "Remote Testing Support",
                "Adaptive Support",
                "Duration (min)",
                "Test Type",
                "Similarity Score"
            ]], use_container_width=True, hide_index=True)

            
            if not other_matches.empty:
                with st.expander("🔍 Show Similar Other Assessments", expanded=False):
                    st.markdown("### 🧠 Other Recommended Assessments")
                    st.dataframe(other_matches[[ 
                        "S.No",
                        "Assessment Name",
                        "Remote Testing Support",
                        "Adaptive Support",
                        "Duration (min)",
                        "Test Type",
                        "Similarity Score"
                    ]], use_container_width=True, hide_index=True)


# streamlit run app.py
