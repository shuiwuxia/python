import streamlit as st
import time
import os
import pandas as pd
import main

st.set_page_config(
    page_title="Legal Precedent Classifier",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded"
)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file (ensure style.css is in the same directory)
try:
    load_css("style.css")
except FileNotFoundError:
    st.error("style.css not found. Please ensure it exists in the project directory.")

# --- Sidebar: Project Context ---
with st.sidebar:
    st.header("📁 Project Context")
    st.markdown(
        "This internal tool connects a legal-text classifier with a Streamlit UI to surface citation relationships in case law."
    )
    st.markdown("---")

    st.subheader("👥 Roles")

    st.markdown(
        """
        <div class="role-card">
            <div class="role-title">Deployment Engineer</div>
            <div class="role-desc">A. HANEETH (AD24B1005)</div>
        </div>
        <div class="role-card">
            <div class="role-title">ML Developer</div>
            <div class="role-desc">A. HANEETH (AD24B1005) &amp; A. Ranvitha Reddy (AD24B1004)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.subheader("🛠️ Tech Stack")
    st.markdown(
        """
        <div style="background-color: #112240; padding: 12px; border-radius: 8px; color: #8892B0; font-family: monospace; font-size: 0.9rem; line-height: 1.6;">
        Streamlit<br>Flask<br>Pandas<br>NLTK<br>Scikit-learn (TF-IDF, Logistic Regression)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<div style="color: #8892B0; font-size: 0.8rem;">Project Version v1.0</div>', unsafe_allow_html=True)

# --- Main Content Area ---
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)
st.markdown('<h1>⚖️ Legal Precedent Classifier</h1>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="description-text">
        Tool to analyze legal text for the presence of citation relationships<br>
        <b>(Cited, Followed, Referred To)</b>. Paste text below to classify.
    </div>
    """,
    unsafe_allow_html=True,
)


def predict_citation(text: str):
    """Use the trained ML model from main.py to predict citation type.

    Returns (has_citation:int, label:str, probs:dict).
    """
    time.sleep(0.2)  # Small delay to simulate processing

    # main.model and main.tfidf_vectorizer are created when main.py is imported
    model = main.model
    vectorizer = main.tfidf_vectorizer
    classes = model.classes_

    cleaned_text = text.lower()
    features = vectorizer.transform([cleaned_text])

    # Raw model prediction
    label = model.predict(features)[0]

    # Build probability dictionary
    proba = model.predict_proba(features)[0]
    probs = {str(cls): float(p) for cls, p in zip(classes, proba)}

    has_citation = 1 if str(label).lower() in ["cited", "followed", "referred to"] else 0
    return has_citation, label, probs

# Action Button
# Using columns to center the button if needed, or just full width
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("ANALYZE CITATION")

# Input Field
legal_text = st.text_area(
    "Paste Legal Text Here.",
    height=220,
    placeholder="Enter the legal case text or citation context here...",
    label_visibility="visible",
)

if analyze_button:
    if not legal_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text for citation patterns..."):
            has_relation, label, probs = predict_citation(legal_text)
            
            if has_relation == 1:
                pretty_label = str(label).title()
                st.markdown(
                    """
                    <div class="result-card success-card">
                        <div class="success-title">✅ CITATION: {}</div>
                        <div class="confidence-score">Predicted Type: <b>Citation - {}</b></div>
                    </div>
                    """.format(pretty_label, pretty_label),
                    unsafe_allow_html=True,
                )
            else:
                pretty_label = str(label).title()
                st.markdown(
                    """
                    <div class="result-card failure-card">
                        <div class="failure-title">❌ NO CITATION RELATIONSHIP DETECTED</div>
                        <div class="confidence-score">
                            Cited / Followed / Referred To not found.<br>
                            Model label: <b>{}</b>
                        </div>
                    </div>
                    """.format(pretty_label),
                    unsafe_allow_html=True,
                )

            if probs:
                probs_df = pd.DataFrame(
                    {
                        "Class": [str(k).title() for k in probs.keys()],
                        "Probability": list(probs.values()),
                    }
                ).sort_values("Probability", ascending=False)

                st.markdown("**Model class probabilities**")
                st.dataframe(
                    probs_df.style.format({"Probability": "{:.2%}"}),
                    use_container_width=True,
                )

                probs_chart = probs_df.set_index("Class")
                st.bar_chart(probs_chart)

# --- Footer ---
st.markdown('<div class="footer">Legal Tech AI Pipeline • Internal Tool • 2025</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
