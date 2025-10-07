import streamlit as st
from transformers import pipeline
import pandas as pd

# Load AI model
sentiment_analyzer = pipeline("sentiment-analysis")

# Streamlit page settings
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🤖", layout="centered")

# Page title
st.title("🤖 AI Sentiment Analyzer")
st.write("Type a sentence below and see what the AI thinks!")

# Text input
user_text = st.text_input("Enter a sentence:", "")

# When user types something
if user_text:
    result = sentiment_analyzer(user_text)[0]
    label = result['label']
    score = result['score']

    # Set emoji and color
    if label == "POSITIVE":
        emoji = "😄"
        color = "green"
    elif label == "NEGATIVE":
        emoji = "😞"
        color = "red"
    else:
        emoji = "😐"
        color = "blue"

    # Display sentiment result
    st.markdown(
        f"<h3 style='text-align:center; color:{color};'>{emoji} {label}</h3>",
        unsafe_allow_html=True
    )
    st.markdown(f"**Confidence:** {score:.2f}")

    # Create a bar chart for confidence
    chart_data = pd.DataFrame({
        "Sentiment": [label],
        "Confidence": [score]
    })
    st.bar_chart(chart_data.set_index("Sentiment"))

else:
    st.info("👆 Type a sentence to analyze sentiment.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Built with ❤️ using Streamlit and Transformers")
