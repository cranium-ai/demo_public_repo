# LLM RAG Bot

This is a simple question and answer bot implemented in Python using the LangChain and Google Generative AI libraries. It uses the RAG (Retrieval-Augmented Generation) model for generating responses to questions from PDF documents.

## Requirements

- Python 3.6 or higher
- LangChain
- dotenv

You can install the required Python packages using pip:

```sh
pip install langchain langchain-google-genai langchain-community langchain-core python-dotenv
```

## Environment Variables
Create a .env file in the root directory of your project and add your Google API key:

GOOGLE_API_KEY=your_google_api_key

## Usage
To run the RAG model, use the following command:

```sh
python llm_rag.py  --doc_path path_to_your_pdf --prompt "Your question here"
```