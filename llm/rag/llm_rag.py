import os
import argparse

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_rag(doc_path, prompt):
    google_api_key = os.environ.get("GOOGLE_API_KEY")

    chat_model = ChatGoogleGenerativeAI(google_api_key=google_api_key, 
                                    model="gemini-1.5-pro-latest")

    loader = PyPDFLoader(doc_path)
    pages = loader.load_and_split()
    text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")
    db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
    db.persist()
    db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)
    retriever = db_connection.as_retriever(search_kwargs={"k": 5})

    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a Helpful AI Bot.
                    Given a context and question from user,
                    you should answer based on the given context."""),
        HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
            Context: {context}
            Question: {question}
            Answer: """)
    ])

    output_parser = StrOutputParser()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | chat_template
        | chat_model
        | output_parser
    )

    response = rag_chain.invoke(prompt)

    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG with a given prompt")
    parser.add_argument("prompt", type=str, help="The prompt to be processed by the RAG model")
    parser.add_argument("doc_path", type=str, help="The path to the PDF document to be used as context")
    args = parser.parse_args()

    response = run_rag(args.doc_path, args.prompt)
