# GenAI Bank Chatbot

An AI banking assistant that answers questions about a bank's products, loans, services, and policies — using only the bank's own documents as its source of truth.

It pairs an LLM with RAG (Retrieval-Augmented Generation): the app first searches the bank's documents for the relevant passages, then the LLM writes the answer from those passages only. That keeps answers grounded and cuts down on made-up information.

## Features

- RAG answers built from the bank's policy documents (.txt and .pdf)
- Fast local vector search with FAISS, so retrieval takes milliseconds
- Conversation memory — follow-up questions like "and what about car loans?" actually work
- Source citations shown as chips under each answer, so you can see where it came from
- Secure sessions with JWT token validation
- Clean chat UI with a smooth typing animation

## How it works

1. The bank's documents are loaded, split into chunks, and stored as embeddings in a FAISS vector index.
2. A customer asks a question in the chat.
3. The app retrieves the most relevant chunks for that question.
4. A fast cloud LLM (Groq) writes the answer using only those chunks.
5. The answer appears in the chat along with its source citations.

## Tech stack

- Flask — web app and API
- LangChain — the RAG pipeline
- FAISS — local vector store for fast similarity search
- Sentence Transformers — text embeddings
- Groq — fast LLM generation
- PyJWT — secure session tokens

## Getting started

1. Install Python 3.10+ and the dependencies:

   pip install -r requirements.txt

2. Create a .env file in the project root:

   GROQ_API_KEY="your_groq_api_key"
   SECRET_KEY="your_jwt_secret"

3. Run the app:

   python app.py

4. Open http://localhost:3000 in your browser and start chatting. The port can be changed with the PORT environment variable.

## Project structure

- app.py — Flask entry point
- chatbot/ — RAG logic: document loading, chunking, retrieval, and generation
- docs/ — documentation
- static/ — styles and chat JavaScript
- templates/ — HTML pages
- uploads/ — documents uploaded through the app

## License

MIT — use it, change it, build on it.