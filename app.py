import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from PIL import Image


st.set_page_config(page_title="SHL Assessment Recommendation Engine", layout="wide")




st.markdown("""
    <style>
        .top-nav {
            background-color: #F4F4F4;
            padding: 10px 20px;
            display: flex;
            justify-content: flex-end;
            gap: 20px;
            font-size: 14px;
        }
        .top-nav a {
            text-decoration: none;
            color: #444;
        }
        .menu-bar {
            background-color: #ffffff;
            padding: 15px 20px 10px 20px;
            display: flex;
            justify-content: flex-end;
            gap: 30px;
            font-weight: 600;
            font-size: 16px;
            border-bottom: 1px solid #eee;
        }
        .menu-bar a {
            text-decoration: none;
            color: #0073e6;
        }
    </style>
    <div class="top-nav">
        <a href="https://www.shl.com/contact-us/" target="_blank">Contact</a>
        <a href="https://www.shl.com/resources/assessments/practice-tests/" target="_blank">Practice Test</a>
        <a href="https://www.shl.com/support/" target="_blank">Support</a>
    </div>
    <div class="menu-bar">
        <a href="https://www.shl.com/solutions/" target="_blank">Solutions</a>
        <a href="https://www.shl.com/solutions/hr-priorities/" target="_blank">HR Priorities</a>
        <a href="https://www.shl.com/resources/" target="_blank">Resources</a>
        <a href="https://www.shl.com/about/careers/" target="_blank">Careers</a>
        <a href="https://www.shl.com/about/" target="_blank">About</a>
    </div>
""", unsafe_allow_html=True)




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
                    "https://shl-9hkt.onrender.com/recommend/",
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



st.markdown("""
<style>
.footer {
    background-color: #2a2a2a;
    color: #fff;
    padding: 40px 80px;
    font-size: 14px;
    margin-top: 50px;
}
.footer-columns {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
}
.footer-column {
    flex: 1;
    min-width: 200px;
    margin: 10px;
}
.footer-column h4 {
    border-bottom: 1px solid #fff;
    padding-bottom: 10px;
    margin-bottom: 10px;
}
.footer-column a {
    display: block;
    color: #ccc;
    text-decoration: none;
    margin: 6px 0;
}
.footer-column a:hover {
    color: #fff;
}
.footer-bottom {
    text-align: center;
    padding-top: 30px;
    border-top: 1px solid #444;
    color: #bbb;
}
            
.social-bar {
    background-color: #2a2a2a;
    text-align: center;
    padding: 20px 0 10px 0;
}
.social-icons a {
    display: inline-block;
    margin: 0 15px;
    color: white;
    font-size: 22px;
    transition: transform 0.3s ease;
}
.social-icons a:hover {
    transform: scale(1.2);
    color: #00acee; /* Twitter color on hover */
}            
</style>
            
<!-- Font Awesome CDN for icons -->
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">

       

<div class="footer">
            
<div class="social-bar">
  <div class="social-icons">
    <a href="https://www.youtube.com/@SHLGlobal" target="_blank"><i class="fab fa-youtube"></i></a>
    <a href="https://www.instagram.com/shlglobal/" target="_blank"><i class="fab fa-instagram"></i></a>
    <a href="https://www.facebook.com/SHLGlobal/" target="_blank"><i class="fab fa-facebook-f"></i></a>
    <a href="https://twitter.com/shlglobal" target="_blank"><i class="fab fa-twitter"></i></a>
    <a href="https://www.linkedin.com/company/shl/" target="_blank"><i class="fab fa-linkedin-in"></i></a>
  </div>
</div>                
  <div class="footer-columns">
    <div class="footer-column">
      <h4>Company</h4>
      <a href="https://www.shl.com/about/" target="_blank">About SHL</a>
      <a href="https://www.shl.com/solutions/" target="_blank">Solutions</a>
      <a href="https://www.shl.com/products/" target="_blank">Products</a>
      <a href="https://www.shl.com/resources/case-studies/" target="_blank">Case Studies</a>
      <a href="https://www.shl.com/about/careers/" target="_blank">SHL Careers</a>
      <a href="https://www.shl.com/about/global-offices/" target="_blank">Global Offices</a>
      <a href="https://www.shl.com/about/media-inquiries/" target="_blank">Media Inquiries</a>
      <a href="https://www.shl.com/subscribe/" target="_blank">Subscribe</a>
    </div>
    <div class="footer-column">
      <h4>Client Resources</h4>
      <a href="https://www.shl.com/contact-us/" target="_blank">Sales Inquiries</a>
      <a href="https://platform.shl.com/" target="_blank">Platform Login</a>
      <a href="https://www.shl.com/support/client-support/" target="_blank">Client Support ↗</a>
      <a href="https://www.shl.com/resources/product-catalog/" target="_blank">Product Catalog</a>
      <a href="https://www.shl.com/resources/training-calendar/" target="_blank">Training Calendar</a>
      <a href="https://shop.shl.com/" target="_blank">Buy Online ↗</a>
    </div>
    <div class="footer-column">
      <h4>Candidate Resources</h4>
      <a href="https://www.shl.com/support/candidate-support/" target="_blank">Candidate Support ↗</a>
      <a href="https://www.shl.com/support/raise-an-issue/" target="_blank">Raise An Issue ↗</a>
      <a href="https://www.shl.com/about/neurodiversity-hub/" target="_blank">Neurodiversity Hub</a>
      <a href="https://www.shl.com/resources/assessments/practice-tests/" target="_blank">Practice Tests</a>
    </div>
    <div class="footer-column">
      <h4>Legal</h4>
      <a href="https://www.shl.com/legal/cookie-policy/" target="_blank">Cookie Policy</a>
      <a href="https://www.shl.com/legal/privacy-notice/" target="_blank">Privacy Notice</a>
      <a href="https://www.shl.com/legal/security-compliance/" target="_blank">Security & Compliance</a>
      <a href="https://www.shl.com/legal/resources/" target="_blank">Legal Resources</a>
      <a href="https://www.shl.com/legal/uk-modern-slavery-act/" target="_blank">UK Modern Slavery</a>
      <a href="https://www.shl.com/sitemap/" target="_blank">Site Map</a>
      <a href="https://www.shl.com/search/" target="_blank">Site Search</a>
    </div>
  </div>
  <div class="footer-bottom">
    © 2025 SHL and its affiliates. All rights reserved.
  </div>
</div>
""", unsafe_allow_html=True)




# streamlit run app.py
