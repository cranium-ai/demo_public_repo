from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask import Flask, request, jsonify

# Load the model and tokenizer from Hugging Face
model_name = "meta-llama/LLaMA-3"  # Replace with the correct model name if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the function to generate a response
def generate_response(question, model, tokenizer):
    inputs = tokenizer.encode("Q: " + question + "\nA:", return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Create the Flask app
app = Flask(__name__)

@app.route('/qa', methods=['POST'])
def qa():
    data = request.json
    question = data.get('question')
    response = generate_response(question, model, tokenizer)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
