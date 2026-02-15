import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Language Detection", layout="wide")

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e6f0ff 0%, #f5f7fa 100%);
}

.block-container {
    padding-top: 5rem;
    padding-bottom: 4rem;
}

.big-title {
    font-size: 36px;
    font-weight: 800;
    color: #0f172a;
    white-space: nowrap;
    text-align: left;
}

.description {
    font-size: 18px;
    color: #475569;
    margin-top: 25px;
    line-height: 1.8;
    max-width: 700px;
    text-align: justify;
}

.custom-label {
    font-size: 22px;
    font-weight: 400;
    color: #0f172a;
    margin-bottom: 10px;
}

textarea {
    font-size: 18px !important;
    padding: 15px !important;
}

.card {
    background: white;
    padding: 35px;
    border-radius: 18px;
    box-shadow: 0px 20px 40px rgba(0,0,0,0.08);
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 10px 30px;
    border: none;
}

.stButton>button:hover {
    background-color: #1e40af;
}

.result-box {
    margin-top: 20px;
    padding: 15px;
    border-radius: 10px;
    background-color: #f1f5f9;
    font-size: 18px;
    font-weight: 600;
    color: #0f172a;
}

</style>
""", unsafe_allow_html=True)

left, right = st.columns([1.2, 1])

with left:
    st.markdown(
        "<div class='big-title'>NLP Based Language Identification System</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """<div class='description'>
        Detect the language of your text instantly, whether it's in English, Spanish, or other languages.
        Enter your text and click the button to find out what language your text is in.
        </div>""",
        unsafe_allow_html=True
    )

@st.cache_data
def load_data():
    data = pd.read_csv("Language_Detection.csv")
    data.columns = ["Text", "Language"]
    data = data.dropna()
    return data

data = load_data()

@st.cache_resource
def train_model(data):
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(analyzer='char', ngram_range=(2,4), max_features=5000)),
        ("classifier", MultinomialNB())
    ])
    pipeline.fit(data["Text"], data["Language"])
    return pipeline

model = train_model(data)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("<div class='custom-label'>Enter your text:</div>", unsafe_allow_html=True)

    user_input = st.text_area("", height=220)

    if st.button("Detect language"):
        if user_input.strip() != "":
            prediction = model.predict([user_input])[0]
            st.markdown(
                f"<div class='result-box'>Language detected: {prediction}</div>",
                unsafe_allow_html=True
            )
        else:
            st.warning("Please enter some text.")

    st.markdown("</div>", unsafe_allow_html=True)
