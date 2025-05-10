# gpt_wrapper.py
from transformers import pipeline
import re
import streamlit as st

# Load the lightweight FLAN model
try:
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
except Exception as e:
    generator = None
    print("Model loading failed. Check environment or dependencies.", e)

def generate_segment_insights(segment_stats):
    if generator is None:
        
        st.write(generator("Who are you?"))
        return {
            "name": "Unnamed Segment",
            "description": "Model not available.",
            "message": "No suggestion generated."
        }

    # Structured prompt
    prompt = f"""
You are a customer insights assistant.

DO NOT add any intro or explanation text.
Return your response ONLY in the exact format below:

Segment Name: [Name]  
Description: [Customer behavior summary]  
Message: [Marketing recommendation]

Given the following stats:
- Average Spend: ${segment_stats.get('avg_spend', 'N/A')}
- Purchase Frequency: {segment_stats.get('avg_frequency', 'N/A')} times
- Recency: {segment_stats.get('avg_recency', 'N/A')} days since last purchase
"""

    result = generator(prompt, max_length=256, do_sample=False, temperature=0)
    text = result[0]['generated_text'] if result else "No output"
    print("GPT raw output:\n", text)  # Debug: print full raw output

    # Loosened regex to match flexible model formatting
    name_match = re.search(r"(Segment Name|Name)[:：]?\s*(.*)", text, re.IGNORECASE)
    desc_match = re.search(r"(Description)[:：]?\s*(.*)", text, re.IGNORECASE)
    msg_match  = re.search(r"(Message)[:：]?\s*(.*)", text, re.IGNORECASE)

    return {
        "name": name_match.group(2).strip() if name_match else "Unnamed Segment",
        "description": desc_match.group(2).strip() if desc_match else "No description found.",
        "message": msg_match.group(2).strip() if msg_match else "No suggestion generated."
    }