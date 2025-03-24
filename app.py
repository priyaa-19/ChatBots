import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.schema import HumanMessage, AIMessage
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# ✅ Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")

if not OPENAI_API_KEY or not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("Missing environment variables. Check your .env file.")

# ✅ Initialize Flask app
app = Flask(__name__, template_folder="templates")
app.secret_key = SECRET_KEY

# ✅ Connect to Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

# ✅ Ensure collections exist
def create_collection_if_not_exists(collection_name):
    collections = qdrant_client.get_collections()
    collection_names = [col.name for col in collections.collections]

    if collection_name not in collection_names:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print(f"✅ Created collection: {collection_name}")
    else:
        print(f"🔹 Collection '{collection_name}' already exists.")

create_collection_if_not_exists("troubleshooting_guide_preprocessed")
create_collection_if_not_exists("T_solution")

# ✅ Initialize OpenAI Embeddings
embedding_model = OpenAIEmbeddings()

# ✅ Initialize Qdrant Vector Stores
vector_store_main = QdrantVectorStore(
    client=qdrant_client,
    collection_name="troubleshooting_guide_preprocessed",
    embedding=embedding_model
)

vector_store_solutions = QdrantVectorStore(
    client=qdrant_client,
    collection_name="T_solution",
    embedding=embedding_model
)

# ✅ Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

# ✅ Home Page
@app.route("/")
def index():
    return render_template("index.html")

# ✅ Admin Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    try:
        if request.method == "POST":
            username = request.form.get("username")
            password = request.form.get("password")
            if username == "admin" and password == "password":  # Replace with real authentication
                session["user"] = username
                return redirect(url_for("dashboard"))
            return "❌ Invalid credentials", 401
        return render_template("login.html")
    except Exception as e:
        return f"⚠️ Error in login: {str(e)}"

# ✅ Dashboard Route
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    
    if request.method == "POST":
        new_solution = request.form.get("solution", "").strip()
        category = request.form.get("category", "General").strip()

        if not new_solution:
            return "❌ Solution cannot be empty!", 400
        
        # 🔹 Embed the solution using OpenAI
        embedded_vector = embedding_model.embed_documents([new_solution])[0]

        # 🔹 Generate a valid UUID for the point ID
        point_id = str(uuid4())

        # 🔹 Store in Qdrant with metadata
        qdrant_client.upsert(
            collection_name="T_solution",
            points=[PointStruct(
                id=point_id,
                vector=embedded_vector,
                payload={"content": new_solution, "category": category}
            )]
        )
        
        return "✅ Solution added successfully!", 200
    
    return render_template("dashboard.html")

# ✅ Admin Logout
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ✅ Chatbot Route
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"response": "Please enter a message."})

    # ✅ Manually Embed the User Query
    query_vector = embedding_model.embed_documents([user_input])[0]

    # 🔍 Retrieve relevant troubleshooting data
    search_results_main = vector_store_main.similarity_search(user_input, k=5)

    # 🔍 Retrieve from T_solution using Vector Search
    retrieved_docs = qdrant_client.search(
        collection_name="T_solution",
        query_vector=query_vector,
        limit=5
    )

    # ✅ Debugging Logs
    print("\n🔹 Retrieved from troubleshooting_guide_preprocessed:")
    for doc in search_results_main:
        print(f"➡️ {doc.page_content}")

    print("\n🔹 Retrieved from T_solution:")
    for doc in retrieved_docs:
        print(f"✅ ID: {doc.id}, Content: {doc.payload.get('content', 'No content found')}, Category: {doc.payload.get('category', 'General')}")

    # 📌 Combine and structure the retrieved results
    combined_results = []
    
    # 🔹 Add results from troubleshooting_guide_preprocessed
    for idx, doc in enumerate(search_results_main):
        combined_results.append({
            "step": idx + 1,
            "content": doc.page_content,
            "category": doc.metadata.get("category", "General")
        })

    # 🔹 Add results from T_solution
    for idx, doc in enumerate(retrieved_docs, start=len(combined_results) + 1):
        combined_results.append({
            "step": idx,
            "content": doc.payload.get("content", "No content found"),
            "category": doc.payload.get("category", "General")
        })

    # 📝 Format the response step-by-step
    if not combined_results:
        return jsonify({"response": "No relevant solutions found."})

    structured_response = "**Here’s what you can do step-by-step:**\n\n"
    for result in combined_results:
        step_text = f"**Step {result['step']}:** {result['content']} (*Category: {result['category']}*)"
        structured_response += step_text + "\n\n"

    # 🧠 LLM Response - Summarized Action Plan
    messages = [
        HumanMessage(content=f"User's issue: {user_input}"),
        AIMessage(content=f"Here are the retrieved troubleshooting steps:\n\n{structured_response}"),
        HumanMessage(content="Can you summarize this and provide a final action plan?")
    ]

    response = llm.invoke(messages)

    return jsonify({
        "retrieved_steps": structured_response,
        "final_summary": response.content
    })
       
if __name__ == "__main__":
    app.run(debug=True)
