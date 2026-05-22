# GenAI Bank Support Chatbot

An intelligent AI-powered banking assistant that combines Large Language Model (LLM) capabilities with Retrieval-Augmented Generation (RAG) to provide accurate information about GenAI Bank's products, services, loan procedures, and financial policies.

This project is fully optimized to perform fast local embedding searches and utilizes high-performance cloud generation to provide grounded customer support answers with source citations.

## Key Features

* **RAG-Powered Responses:** Ingests official policy documents (.txt, .pdf) and runs precise context retrieval.
* **Context Retention (Memory):** Tracks session history over multiple turns to correctly handle follow-up pronoun inquiries.
* **Interactive UI/UX:** Responsive chat interface featuring a custom fluid typing animation and dynamic visual source citation chips.
* **Secure Sessions:** Leverages JWT bearer token validation states to manage customer connection authentication.

## Setup and Installation

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your system.

### 2. Environment Configuration
Create a file named `.env` in the root directory and define the following variables:

GROQ_API_KEY="your_groq_api_key"
SECRET_KEY="your_jwt_auth_signing_secret"

### 3. Install Dependencies
Run the following command in your terminal to install the required packages:

pip install -r requirements.txt

### 4. Build Knowledge Base & Launch Application
Place your banking policy documents or FAQs inside the `uploads/` directory. Run the server to automatically index the documents and start the application:

python app.py

Once the server is running, open your browser and navigate to http://localhost:3000 to interact with the chatbot.
