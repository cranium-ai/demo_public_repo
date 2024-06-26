# Q&A Bot

This is a simple question and answer bot implemented in Python using the Hugging Face Transformers library. It uses the model "meta-llama/LLaMA-3" for generating responses to questions.

## Requirements

- Python 3.6 or higher
- Flask
- PyTorch
- Transformers

You can install the required Python packages using pip:

`pip install flask torch transformers`

## Usage

To start the server, run:

`python qa_bot.py`

The server will start on http://localhost:5000.

To ask a question, send a POST request to http://localhost:5000/qa with a JSON body containing the question. For example:

`curl -X POST -H "Content-Type: application/json" -d '{"question":"What is the capital of France?"}' http://localhost:5000/qa`

The server will respond with the answer generated by the model.

# License
This project is licensed under the terms of the MIT license.

