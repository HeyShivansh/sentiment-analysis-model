import streamlit as st
import requests

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis with BERT",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS (font + theme + widgets)
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f9fafb;
        color: #111827;
    }

    /* Project header */
    .project-title {
        font-size: 2.4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.25rem;
    }

    .project-subtitle {
        font-size: 1rem;
        text-align: center;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    /* Text area styling */
    textarea {
        background-color: #ffffff !important;
        color: #111827 !important;
        border-radius: 10px !important;
        border: 1px solid #d1d5db !important;
        font-size: 1rem !important;
    }

    /* Button styling */
    button[kind="primary"] {
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important;
        border: none !important;
    }

    button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
    }

    /* Sentiment result box */
    .sentiment-box {
        padding: 1.4rem;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        margin-top: 1.8rem;
    }

    .positive {
        background-color: #d1fae5;
        color: #065f46;
    }

    .neutral {
        background-color: #e5e7eb;
        color: #374151;
    }

    .negative {
        background-color: #fee2e2;
        color: #991b1b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="project-title">
        🧠 Sentiment Analysis with BERT
    </div>
    <div class="project-subtitle">
        Transformer-based sentiment classifier trained on the SST dataset
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Input
# --------------------------------------------------
text_input = st.text_area(
    "Enter a sentence to analyze:",
    height=120,
    placeholder="e.g. it is not bad"
)

analyze_btn = st.button("Analyze Sentiment", type="primary")

# --------------------------------------------------
# Backend API endpoint
# --------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict"

# --------------------------------------------------
# Inference
# --------------------------------------------------
if analyze_btn:
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": text_input},
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    sentiment = result["sentiment"]
                    confidence = result["confidence"]

                    if sentiment == "Positive":
                        css_class = "positive"
                    elif sentiment == "Negative":
                        css_class = "negative"
                    else:
                        css_class = "neutral"

                    st.markdown(
                        f"""
                        <div class="sentiment-box {css_class}">
                            {sentiment}<br/>
                            Confidence: {confidence}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:
                    st.error("Backend error. Please try again.")

            except Exception:
                st.error("Could not connect to the backend API.")
