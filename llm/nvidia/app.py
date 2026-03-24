from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer from Hugging Face
model_name = (
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text(prompt, max_length=50):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate text using the model
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
    )

    # Decode the generated tokens back to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    prompt = "Once upon a time in a land far, far away,"
    generated_text = generate_text(prompt)
    print(generated_text)