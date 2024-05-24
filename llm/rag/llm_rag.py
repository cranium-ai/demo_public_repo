import pinecone
from langchain import LangChain, RetrievalQA, OpenAIEmbeddings
from google.cloud import gemini
from flask import Flask, request, jsonify

# Initialize Pinecone
pinecone.init(api_key='your-pinecone-api-key', environment='us-west1-gcp')

# Create an index if it doesn't exist
if 'langchain-rag' not in pinecone.list_indexes():
    pinecone.create_index('langchain-rag', dimension=768, metric='cosine')

# Connect to the index
index = pinecone.Index('langchain-rag')

# Initialize Google Gemini
gemini_client = gemini.Client(api_key='your-google-gemini-api-key')

# Initialize LangChain with Google Gemini
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = gemini_client.LanguageModel()

langchain = LangChain(
    embeddings=embeddings,
    llm=llm,
    retriever=PineconeRetriever(index)
)

# Define the retrieval-augmented generation (RAG) pipeline
rag = RetrievalQA(
    retriever=langchain.retriever,
    generator=langchain.llm
)

# Function to perform RAG
def generate_response(query):
    response = rag({"query": query})
    return response

# Create the Flask app
app = Flask(__name__)

@app.route('/qa', methods=['POST'])
def qa():
    data = request.json
    question = data.get('question')
    response = generate_response(question)
    return jsonify({'response': response['result']})

if __name__ == "__main__":
    app.run(debug=True)
