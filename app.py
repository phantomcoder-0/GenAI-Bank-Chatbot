import os, json
from flask import Flask, render_template, request, jsonify, abort
import jwt
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename

# Import database
from chatbot.database import auth_user, init_db

# Import direct RAG pipeline
from chatbot.rag.document_loader import load_documents, split_documents
from chatbot.rag.vector_store import create_vector_store
from chatbot.rag.rag_chatbot import GenAIChatbot

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Upload Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# JWT configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15

# Initialize the direct RAG Chatbot
bot = GenAIChatbot()

# Helpers for JWT
def create_access_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = payload.get("sub")
        if not user:
            abort(401, "Invalid token payload")
        return user
    except jwt.ExpiredSignatureError:
        abort(401, "Token has expired")
    except jwt.InvalidTokenError:
        abort(401, "Invalid token")

# ==========================================
# Routes
# ==========================================
@app.route("/", methods=["GET"])
def index():
    return render_template("chat.html")

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Server is healthy"}), 200

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and (file.filename.lower().endswith('.pdf') or file.filename.lower().endswith('.txt')):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            raw_docs = load_documents(app.config['UPLOAD_FOLDER'])
            chunked_docs = split_documents(raw_docs)
            create_vector_store(chunked_docs)
            
            # Re-initialize the bot to pick up the newly embedded documents
            global bot
            bot = GenAIChatbot()
            
            return jsonify({
                "status": "success", 
                "message": f"File {filename} uploaded and processed. Context updated."
            }), 200
            
        except Exception as e:
            return jsonify({"error": f"Failed to process document: {str(e)}"}), 500
    else:
        return jsonify({"error": "Unsupported file type. Only PDF and TXT allowed."}), 400

@app.route("/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json() or {}
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        abort(400, 'Missing "username" or "password"')

    if not auth_user(username, password):
        return jsonify({"status": "fail"}), 401

    token = create_access_token(username)
    return jsonify({"status": "success", "access_token": token, "token_type": "bearer"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    # 1) Authentication
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"reply": "🔒 Please login to continue."}), 401
    
    token = auth_header.split(" ", 1)[1]
    user = verify_access_token(token)

    # 2) Get User Message
    msg = request.json.get("message", "").strip()
    if not msg:
        return jsonify({"reply": "💡 I didn’t get any text."}), 400

    # 3) Query the Vector Database using the synchronized RAG bot
    try:
        result = bot.answer_question(msg)
        answer_text = result.get("answer", "I'm sorry, I couldn't generate a response.")
        
        # Optional: Append sources if they exist to prove the RAG pipeline is working
        sources = result.get("sources", [])
        if sources:
            source_list = ", ".join([os.path.basename(s) for s in sources])
            answer_text += f"\n\n*(Source: {source_list})*"

        return jsonify({"reply": answer_text})
        
    except Exception as e:
        return jsonify({"reply": f"❌ Server error: {str(e)}"}), 500

if __name__ == "__main__":
    import os
    
    print("Starting database initialization...")
    init_db()
    print("Database initialized successfully!")
    
    # Read the PORT environment variable provided by Render
    port = int(os.environ.get("PORT", 3000))
    print(f"Binding to port {port}...")
    
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
