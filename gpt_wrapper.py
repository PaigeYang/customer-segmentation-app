# gpt_wrapper.py
from transformers import pipeline

# Load a free Hugging Face model
# You can replace "google/flan-t5-large" with another open-source model
try:
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
except Exception as e:
    generator = None
    print("Model loading failed. Check environment or dependencies.", e)

def generate_segment_insights(segment_stats):
    if generator is None:
        return {
            "description": "Model not available. Please check setup.",
            "message": "Unable to generate recommendation."
        }

    prompt = f"""
    Given the following customer segment:
    - Average Spend: ${segment_stats.get('avg_spend', 'N/A')}
    - Frequency: {segment_stats.get('avg_frequency', 'N/A')} purchases
    - Recency: {segment_stats.get('avg_recency', 'N/A')} days since last purchase

    Give this segment a name, describe their behavior in plain English, and suggest a marketing message.
    """

    result = generator(prompt, max_length=200, do_sample=True, temperature=0.7)
    output = result[0]['generated_text'] if result else "No output."

    # Simple split to extract name + description + message (if present)
    return {
        "description": output.strip(),
        "message": "See description above."
    }