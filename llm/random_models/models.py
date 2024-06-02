from transformers import pipeline

# GPT-2 from Hugging Face
text_generator = pipeline("text-generation", model="gpt2")