import sys
import os
import re
import spacy
from sentence_transformers import SentenceTransformer
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import BSHTMLLoader
from langchain import hub
import torch
import requests
import pyttsx3
import logging
import uvicorn
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

from data_links import *  

# Set up logging for better debugging and device visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server utility functions
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip()

def split_into_chunks(text, chunk_size=512):
    doc = nlp(text)
    words = [token.text for token in doc]
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def process_disneyland_data(links):
    cleaned_text = ""
    link_count = len(links)
    all_chunks = []

    chrome_options = Options()
    # chrome_options.add_argument("--headless=new")

    driver = webdriver.Chrome(options=chrome_options)
    logger.info(f"Processing {link_count} disneyland help articles")
    for link in links:
        # Download the content
        logger.info(f"Loading {link}")
        # response = requests.get(link)
        driver.get(link)
        
        # Wait for the page to load completely
        wait = WebDriverWait(driver, 120)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.content .middle-section')))

        # Write it to a file
        logger.info(f"Write {link} to a file")

        contentElement = driver.find_element(By.CSS_SELECTOR, 'div.content')

        with open("data.html", "w", encoding="utf-8") as f:
            content_to_write = contentElement.get_attribute("innerHTML")
            f.write(content_to_write)
        # Load it with an HTML parser
        loader = BSHTMLLoader("data.html")
        document = loader.load()[0]
        # Clean up code
        # Replace consecutive new lines with a single new line
        raw_text = re.sub("\n\n+", "\n", document.page_content)
        logger.info(f"\n\nData found {raw_text}\n\n")
        cleaned_text += "\n" + clean_text(raw_text)
        all_chunks += split_into_chunks(cleaned_text)
    return all_chunks

def run_server():
    """Start the FastAPI server with RAG and LLM."""
    # Get the model path from environment variable or use a default
    model_name = os.getenv('MODEL_NAME', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    # model_name = os.getenv('MODEL_NAME', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    # model_name = os.getenv('MODEL_NAME', 'meta-llama/Llama-3.1-8B-Instruct')
    logger.info(f"Loading model: {model_name}")

    # Preprocess documents
    try:
        chunks = process_disneyland_data(data_links)
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
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, load_in_8bit=False)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, load_in_8bit=False)
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
            max_new_tokens=10000
        )
        llm = HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        logger.error(f"Error setting up LLM pipeline: {e}")
        sys.exit(1)

    # Define prompt template
    template = """Using the following context from Disneyland Parks documentation: {context}, answering the query: {query}"""
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])
    chain = prompt | llm | StrOutputParser()

    # Define FastAPI app
    app = FastAPI()

    class Query(BaseModel):
        text: str

    @app.post("/query")
    def query_model(query: Query):
        try:
            logger.info(f"Processing request for {query.text}")
            docs = retriever.get_relevant_documents(query.text)
            context = " ".join([doc.page_content for doc in docs])
            logger.info(f"Processing request with this context: {context}")
            response = chain.invoke({"query": query.text, "context": context})
            # response = llm.invoke(messages)
            answer = response.split("</think>")[-1] # Split thinking and get only the answer
            return {"answer": response, "final": answer}
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"answer": "An error occurred while processing your query."}

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=80)

# TTS client functions
def query_server(query):
    server_path = os.getenv('SERVER_URL', 'http://0.0.0.0/query')
    """Send query to the FastAPI server and return the response."""
    try:
        response = requests.post(server_path, json={"text": query}, timeout=1000)
        response.raise_for_status()
        return response.json()["final"]
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
        speak_text(answer) # TODO: Look into custom ai tts

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