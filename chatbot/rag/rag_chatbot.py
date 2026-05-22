import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.documents import Document

# Import local vector store and loader
from chatbot.rag.vector_store import load_vector_store, create_vector_store
from chatbot.rag.document_loader import load_documents, split_documents

load_dotenv()

class GenAIChatbot:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GenAIChatbot, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, persist_directory=None):
        from chatbot.config import VECTOR_DB_DIR
        
        if persist_directory is None:
            persist_directory = VECTOR_DB_DIR
            
        if getattr(self, '_initialized', False):
            return
            
        self._ensure_vector_store_exists(persist_directory)
        self.vector_store = load_vector_store(persist_directory)
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Updated System Prompt to account for conversational history context
        self.system_prompt = """
        You are an AI Support Agent for GenAI Bank. Provide accurate information 
        about banking products, loan queries, credit cards, and policies based ONLY on the official documentation provided 
        and the ongoing conversation history. If you're unsure or the information isn't in the provided context, 
        acknowledge that and suggest the user contact GenAI Bank support directly. Always be professional, helpful, and concise.
        
        IMPORTANT: You must ONLY answer questions related to banking, financial services, or policies.
        For any questions outside of these domains, politely decline to answer.
        """
        
        # Functional Requirement: Session-based conversation history tracker
        self.conversation_history = []
        
        self._initialized = True
    
    def _ensure_vector_store_exists(self, persist_directory):
        from chatbot.config import DOCS_DIRECTORY
        if not os.path.exists(persist_directory):
            print("Vector store not found. Creating new store...")
            if os.path.exists(DOCS_DIRECTORY):
                documents = load_documents(DOCS_DIRECTORY)
                if documents:
                    chunks = split_documents(documents)
                    create_vector_store(chunks, persist_directory)
                    return
            
            dummy_doc = [Document(page_content="Welcome to GenAI Bank Support.", metadata={"source": "system"})]
            create_vector_store(dummy_doc, persist_directory)
    
    def answer_question(self, question):
        try:
            # 1. Retrieve the top 5 relevant document chunks from ChromaDB based on current question
            docs = self.vector_store.similarity_search(question, k=5)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # 2. Format the recent conversation history to provide conversational context
            # Keeping the last 4 turns (2 user messages, 2 bot answers) to stay safely within context limits
            history_text = ""
            for turn in self.conversation_history[-4:]:
                history_text += f"{turn['role'].upper()}: {turn['content']}\n"
            
            # 3. Construct an advanced context-aware prompt bridging history and retrieval
            enhanced_prompt = f"""
            {self.system_prompt}
            
            CONTEXT FROM OFFICIAL DOCUMENTS:
            {context_text}
            
            RECENT CONVERSATION HISTORY:
            {history_text}
            
            NEW USER QUESTION: {question}
            """
            
            # 4. Invoke Groq
            response = self.llm.invoke(enhanced_prompt)
            answer = response.content
            
            # 5. Append the exchange to history to maintain session memory
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            return {
                "answer": answer,
                "sources": list(set([doc.metadata.get("source", "unknown") for doc in docs]))
            }
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "sources": []}
