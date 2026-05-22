import os
import glob
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = "faiss_index"
UPLOADS_DIR = "uploads"

def create_vector_store(*args, **kwargs):
    # OFFLINE MODE: Bypasses Render's broken network completely
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(DB_DIR):
        print("Connecting to existing FAISS database...", flush=True)
        return FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    
    print("Building new FAISS database from uploads...", flush=True)
    documents = []
    
    if not os.path.exists(UPLOADS_DIR):
        os.makedirs(UPLOADS_DIR)
        return None

    for filepath in glob.glob(f"{UPLOADS_DIR}/*"):
        if filepath.endswith(".txt"):
            documents.extend(TextLoader(filepath).load())
        elif filepath.endswith(".pdf"):
            documents.extend(PyPDFLoader(filepath).load())

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(DB_DIR)
    
    return vector_db

def init_db(*args, **kwargs):
    create_vector_store()

def load_vector_store(*args, **kwargs):
    return create_vector_store()
