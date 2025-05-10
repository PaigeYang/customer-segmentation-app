# app.py
import streamlit as st
import pandas as pd
from segmentation import segment_customers
from gpt_wrapper import generate_segment_insights
from transformers import pipeline

st.set_page_config(page_title="Customer Segmentation Assistant")
st.title("📊 Customer Segmentation Assistant")

# Step 1: Data Upload
st.sidebar.header("Upload or Use Sample Data")
uploaded_file = st.sidebar.file_uploader("Upload customer CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset uploaded successfully!")
else:
    df = pd.read_csv("sample_data/sample_customers.csv")
    st.info("Using sample dataset.")

st.subheader("Customer Data Preview")
st.dataframe(df.head())

generator = pipeline("text2text-generation", model="google/flan-t5-base")
st.write(generator("who are you?"))
st.write(generator("WHat is 1+1?"))

# Step 2: Segmentation
if st.button("🔍 Segment Customers"):
    segments = segment_customers(df)
    st.subheader("📌 Segments")

    for i, seg in enumerate(segments):
        st.markdown(f"### Segment {i+1}: {seg['name']}")
        st.write("Summary Stats:", seg['stats'])

        insights = generate_segment_insights(seg['stats'])
        st.write(insights['description'])
        st.markdown(f"**Suggested Campaign:** {insights['message']}")
        st.write(insights)
