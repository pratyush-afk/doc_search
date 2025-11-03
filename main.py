import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# --- LangChain Core Components ---
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# --- API Setup ---
app = FastAPI(
    title="Document Q&A Chatbot API",
    description="An API for chatting with your documents using RAG.",
    version="1.0.0"
)

# --- Configuration & Global Variables ---
DATA_PATH = "data"
VECTORSTORE_PATH = "vectorstore"

# We will load these models on startup
embeddings = None
vector_store = None
chain = None

# --- Pydantic Models for API ---
class Query(BaseModel):
    question: str
    session_id: Optional[str] = None # Placeholder for future chat history

class Answer(BaseModel):
    answer: str
    source_documents: List[dict] = []

class IndexingStatus(BaseModel):
    status: str
    message: str
    files_indexed: int = 0

# --- Helper Functions ---

def load_documents():
    """Loads all documents from the DATA_PATH directory."""
    # Configure loaders for different file types
    pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    md_loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader)

    all_loaders = [pdf_loader, txt_loader, md_loader]
    
    docs = []
    for loader in all_loaders:
        docs.extend(loader.load())
    
    return docs

def create_vector_store():
    """Creates and persists the vector store from documents in DATA_PATH."""
    global vector_store, embeddings

    print("Loading documents...")
    documents = load_documents()
    
    if not documents:
        print("No documents found in /data directory. Skipping vector store creation.")
        return 0

    print(f"Loaded {len(documents)} document chunks.")
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)
    
    print(f"Split into {len(splits)} chunks.")
    
    # Use a high-quality, free, and local embedding model
    print("Initializing embeddings model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Creating vector store with FAISS...")
    # This creates the in-memory vector store
    vector_store = FAISS.from_documents(splits, embeddings)
    
    # Save the vector store to disk
    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH) # Clear old store
    vector_store.save_local(VECTORSTORE_PATH)
    
    print(f"Vector store created and saved to {VECTORSTORE_PATH}")
    return len(splits)

def initialize_qa_chain():
    """Initializes the RAG query chain."""
    global chain, vector_store, embeddings
    
    # Load the persisted vector store
    if os.path.exists(VECTORSTORE_PATH):
        print(f"Loading vector store from {VECTORSTORE_PATH}...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(VECTORSTORE_PATH, embeddings)
    else:
        print("No vector store found. Indexing documents first.")
        create_vector_store()
        if vector_store is None: # Still no store (e.g., no docs)
            print("Failed to create vector store. Q&A chain will not be available.")
            return

    print("Initializing Q&A Chain...")
    
    # Set up the LLM (Requires OPENAI_API_KEY environment variable)
    # You can swap this with any LangChain-compatible LLM
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",  # Use a fast and cost-effective model
        temperature=0.1
    )
    
    # Create the RAG chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" puts all retrieved docs into the context
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}), # Retrieve top 4 chunks
        return_source_documents=True
    )
    print("Q&A Chain ready.")

# --- FastAPI Events ---

@app.on_event("startup")
async def startup_event():
    """
    On server startup, load embeddings, load/create the vector store,
    and initialize the Q&A chain.
    """
    global chain
    
    # Check for OpenAI API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set. API will fail.")
        # Or raise an error:
        # raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set.")
    
    print("Server starting up...")
    initialize_qa_chain()

# --- FastAPI Endpoints ---

@app.get("/", summary="Check API status")
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "RAG API is running."}

@app.post("/ask", response_model=Answer, summary="Ask a question")
async def ask_question(query: Query):
    """
    Asks a question to the documents and returns an answer.
    """
    if chain is None:
        raise HTTPException(status_code=503, detail="Q&A chain is not initialized. Check server logs.")
    
    try:
        print(f"Received query: {query.question}")
        
        # Invoke the RAG chain
        # The chain handles finding relevant docs, passing them to the LLM, and getting an answer
        response = chain.invoke({"query": query.question})
        
        # Format the source documents for the response
        sources = []
        if response.get("source_documents"):
            for doc in response["source_documents"]:
                sources.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "page_content_preview": doc.page_content[:150] + "..."
                })
        
        return Answer(
            answer=response.get("result", "No answer found."),
            source_documents=sources
        )
        
    except Exception as e:
        print(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-document", summary="Upload a single document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a new document (.pdf, .txt, .md) to the data directory.
    NOTE: This demo implementation requires a /re-index call to process it.
    """
    # Simple security check for file extension
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_ext = os.path.splitext(file.filename)[1]
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed_extensions}")

    file_path = os.path.join(DATA_PATH, file.filename)
    
    try:
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {
            "status": "success",
            "filename": file.filename,
            "message": "File uploaded. Call /re-index to add it to the knowledge base."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

@app.post("/re-index", response_model=IndexingStatus, summary="Re-index all documents")
async def re_index_documents():
    """
    Forces the server to re-scan the /data directory, rebuild the
    vector store, and re-initialize the Q&A chain.
    """
    global chain, vector_store
    
    print("Re-indexing requested...")
    try:
        # Clear old chain
        chain = None
        vector_store = None
        
        # Re-build
        files_indexed = create_vector_store()
        initialize_qa_chain()
        
        return IndexingStatus(
            status="success",
            message="Re-indexing complete. Q&A chain is ready.",
            files_indexed=files_indexed
        )
    except Exception as e:
        print(f"Error during re-indexing: {e}")
        raise HTTPException(status_code=500, detail=f"Re-indexing failed: {e}")

# --- Main execution ---
if __name__ == "__main__":
    # Create the data directory if it doesn't exist
    os.makedirs(DATA_PATH, exist_ok=True)
    
    # Start the server
    print("Starting Uvicorn server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
