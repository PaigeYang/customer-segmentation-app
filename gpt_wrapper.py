from transformers import pipeline
import re

# Load the lightweight FLAN model
try:
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
except Exception as e:
    generator = None
    print("Model loading failed. Check environment or dependencies.", e)

def generate_segment_insights(segment_stats):
    if generator is None:
        return {
            "name": "Unnamed Segment",
            "description": "Model not available.",
            "message": "No suggestion generated."
        }

    # Build structured prompt
    prompt = f"""
You are a customer insights assistant. Given the following segment statistics:
- Average Spend: ${segment_stats.get('avg_spend', 'N/A')}
- Purchase Frequency: {segment_stats.get('avg_frequency', 'N/A')} times
- Recency: {segment_stats.get('avg_recency', 'N/A')} days since last purchase

Please generate:
1. A segment name
2. A 1-2 sentence summary of customer behavior
3. A recommended marketing message

Format your response exactly like this:
Segment Name: <name>
Description: <summary>
Message: <marketing suggestion>
"""

    result = generator(prompt, max_length=256, do_sample=True, temperature=0.7)
    text = result[0]['generated_text'] if result else "No output"

    # Use regex to extract fields
    name_match = re.search(r"Segment Name:\s*(.*)", text)
    desc_match = re.search(r"Description:\s*(.*)", text)
    msg_match  = re.search(r"Message:\s*(.*)", text)

    return {
        "name": name_match.group(1).strip() if name_match else "Unnamed Segment",
        "description": desc_match.group(1).strip() if desc_match else "No description found.",
        "message": msg_match.group(1).strip() if msg_match else "No suggestion generated."
    }
