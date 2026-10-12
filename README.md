# Fake Resume Detector — Educational Prototype

A small Streamlit app that extracts text from PDF/TXT resumes and demonstrates TF-IDF + Logistic Regression classification, plus simple rule-based review flags.

## Important limitation
The included examples are a tiny, hand-written demo dataset. The score is **not** an authenticity probability and is not suitable for hiring decisions. Replace the demo data with a large, legally obtained, representative, labeled dataset and evaluate for bias, false positives, and false negatives before making stronger claims.

## Run locally
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL printed by Streamlit. Upload a text-based PDF or TXT file. Scanned PDFs may require OCR first.

## Tech
Python, Streamlit, scikit-learn, TF-IDF, Logistic Regression, pypdf.
