import html
import re
from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(page_title="Food Review Sentiment", page_icon="🍽️")

MODEL_PATH = Path(__file__).resolve().parent / "sentiment_model.joblib"


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


st.title("Food Review Sentiment")
st.write("Enter an English food review to predict its sentiment.")

if not MODEL_PATH.is_file():
    st.error("Place sentiment_model.joblib in the same folder as app.py.")
    st.stop()

bundle = load_bundle()

review = st.text_area(
    "Your review",
    placeholder="The biscuits taste delicious. I would buy them again!",
    height=150,
)

if st.button("Predict sentiment"):
    cleaned_review = clean_text(review)

    if not cleaned_review:
        st.warning("Please enter a review first.")
    else:
        features = bundle["vectorizer"].transform([cleaned_review])

        if features.nnz == 0:
            st.warning("No recognised words were found. Try a longer English review.")
        else:
            prediction = bundle["model"].predict(features)[0]
            sentiment = bundle["labels"][int(prediction)]

            if sentiment == "Positive":
                st.success("Predicted sentiment: Positive")
            elif sentiment == "Negative":
                st.error("Predicted sentiment: Negative")
            else:
                st.info("Predicted sentiment: Neutral")

