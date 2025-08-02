import streamlit as st
from transformers import pipeline

# Set up Streamlit app UI
st.set_page_config(page_title="Text Summarizer", layout="wide")
st.title("📝 Hugging Face Text Summarizer")

# Text input
user_input = st.text_area("Enter the text you want to summarize", height=250)

# Load summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Button to generate summary
if st.button("Summarize"):
    if user_input.strip():
        with st.spinner("Summarizing..."):
            summary = summarizer(user_input, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
            st.subheader("Summary:")
            st.success(summary)
    else:
        st.warning("Please enter some text to summarize.")
