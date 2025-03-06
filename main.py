import sys
import os
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from sentence_transformers import SentenceTransformer
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import torch
import requests
import pyttsx3
import logging

# Set up logging for better debugging and device visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server utility functions
def clean_text(text):
    """Clean text by removing extra whitespace and HTML tags."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip()

def split_into_chunks(text, chunk_size=512):
    """Split text into chunks of specified word size."""
    words = word_tokenize(text)
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def run_server():
    """Start the FastAPI server with RAG and LLM."""
    # Get the model path from environment variable or use a default
    model_path = os.getenv('MODEL_PATH', 'default/path/to/model')
    logger.info(f"Loading model from: {model_path}")

    # Preprocess documents
    try:
        with open("disneyland_doc.txt", "r") as f:
            raw_text = f.read()
        cleaned_text = clean_text(raw_text)
        chunks = split_into_chunks(cleaned_text)
    except FileNotFoundError:
        logger.error("disneyland_doc.txt not found. Please ensure the file is in the correct directory.")
        sys.exit(1)

    # Generate embeddings
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(chunks, show_progress_bar=True)
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        sys.exit(1)

    # Setup Chroma
    try:
        client = chromadb.Client()
        collection = client.create_collection("disneyland_docs")
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=[str(i) for i in range(len(chunks))]
        )
    except Exception as e:
        logger.error(f"Error setting up Chroma: {e}")
        sys.exit(1)

    # Detect the appropriate device: MPS for Apple Silicon, CUDA if available, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device for Apple Silicon.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU.")

    # Load LLM model and move to the selected device
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(device)  # Move model to the selected device
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

    # Setup LangChain components
    try:
        embedding_function = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        vectorstore = Chroma(collection_name="disneyland_docs", embedding_function=embedding_function, client=client)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        logger.error(f"Error setting up LangChain components: {e}")
        sys.exit(1)

    # Setup LLM pipeline with the selected device
    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device.type == "cuda" else -1,  # Use GPU if CUDA, else CPU
            max_new_tokens=200
        )
        llm = HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        logger.error(f"Error setting up LLM pipeline: {e}")
        sys.exit(1)

    # Define prompt template
    template = """Using the following context from Disneyland Parks documentation: {context}, answer the query: {query}"""
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])
    chain = LLMChain(llm=llm, prompt=prompt)

    # Define FastAPI app
    app = FastAPI()

    class Query(BaseModel):
        text: str

    @app.post("/query")
    def query_model(query: Query):
        try:
            docs = retriever.get_relevant_documents(query.text)
            context = " ".join([doc.page_content for doc in docs])
            # Move inputs to the selected device
            inputs = tokenizer(query.text, return_tensors="pt").to(device)
            # Generate response using the model on the selected device
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=200)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"answer": response}
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"answer": "An error occurred while processing your query."}

    # Start the server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# TTS client functions
def query_server(query):
    """Send query to the FastAPI server and return the response."""
    try:
        response = requests.post("http://localhost:8000/query", json={"text": query}, timeout=10)
        response.raise_for_status()
        return response.json()["answer"]
    except requests.RequestException as e:
        return f"Error: Unable to get response from server ({str(e)})"

def speak_text(text):
    """Convert text to speech using pyttsx3."""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 0.9)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error generating TTS: {e}")

def run_tts_client():
    """Run the TTS client to query the server and speak responses."""
    print("Starting TTS query script. Type 'quit' to exit.")
    while True:
        user_input = input("Enter your query: ")
        if user_input.lower() == "quit":
            break
        answer = query_server(user_input)
        print("Answer:", answer)
        speak_text(answer)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [server|tts]")
        sys.exit(1)
    mode = sys.argv[1].lower()
    if mode == "server":
        run_server()
    elif mode == "tts":
        run_tts_client()
    else:
        print("Invalid mode. Choose 'server' or 'tts'.")
        sys.exit(1)