import re
import io
import joblib
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = "resume_model.joblib"

# Demo examples only. Replace with a properly labeled, representative dataset
# before using this for real decisions.
TRAINING_DATA = [
    ("Experienced Python developer with projects in Flask, SQL and Git. Built a task tracker.", 1),
    ("Completed internship in web development and worked on HTML CSS JavaScript projects.", 1),
    ("B.Tech student with C++ practice, data structures and a college project in Python.", 1),
    ("Worked on a team project and documented my contribution and implementation details.", 1),
    ("Software engineer with verifiable work history and project links.", 1),
    ("Data analyst familiar with pandas, Excel and SQL; project notebooks are available.", 1),
    ("Claimed 40 years of experience despite being a first year student.", 0),
    ("Copied entire resume from another candidate and kept their name and phone number.", 0),
    ("Claims expert in every programming language but provides no projects or evidence.", 0),
    ("Fake employment history with made up company names and impossible dates.", 0),
    ("Duplicate generic profile with inconsistent dates and unrelated job titles.", 0),
    ("Claims senior architect role while listing no work experience or project details.", 0),
]

@st.cache_resource
def train_demo_model():
    texts, labels = zip(*TRAINING_DATA)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(texts, labels)
    return model

def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError("Upload a PDF or TXT file.")

def basic_flags(text):
    flags = []
    if len(text.split()) < 40:
        flags.append("Very short text: there may not be enough information to assess.")
    if re.search(r"\b(20\d{2})\s*[-–]\s*(20\d{2})\b", text):
        flags.append("Review the listed date ranges manually for consistency.")
    if re.search(r"\b(expert in all|master of every|100% guaranteed)\b", text, re.I):
        flags.append("Contains broad or unusually absolute skill claims.")
    if not re.search(r"\b(project|built|developed|implemented|created)\b", text, re.I):
        flags.append("No clear project/action description detected by this simple rule.")
    if not flags:
        flags.append("No simple rule-based flags found; this does not prove the resume is genuine.")
    return flags

st.set_page_config(page_title="Resume Text Review Demo", page_icon="📄", layout="centered")
st.title("Resume Text Review — Demo")
st.caption("Educational prototype only. It cannot determine whether a resume is genuine and must not be used to make hiring decisions.")
uploaded = st.file_uploader("Upload a resume (PDF or TXT)", type=["pdf", "txt"])

if uploaded:
    try:
        text = extract_text(uploaded)
        if not text.strip():
            st.error("No readable text found. For scanned PDFs, run OCR first or upload a TXT file.")
        else:
            st.subheader("Basic review")
            st.write(f"Extracted approximately **{len(text.split())} words**.")
            for flag in basic_flags(text):
                st.write(f"- {flag}")
            model = train_demo_model()
            probability = float(model.predict_proba([text])[0][1])
            st.metric("Demo similarity score", f"{probability * 100:.1f}%")
            st.warning("This score is based on a tiny hand-written demo dataset. It is not an authenticity probability, and it will not reliably detect fraud.")
            with st.expander("View extracted text"):
                st.text(text[:12000])
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
