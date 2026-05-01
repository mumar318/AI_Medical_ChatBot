from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from simple_rag import ask_medical_bot
from typing import List
import tempfile
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI(title="Medical AI Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    enhanced: bool = False

@app.post("/ask")
def ask(query: Query):
    try:
        result = ask_medical_bot(query.question)
        return {
            "answer": result,
            "enhanced": query.enhanced,
            "status": "success"
        }
    except Exception as e:
        return {
            "answer": f"I apologize, but I encountered an error: {str(e)}. Please consult a healthcare professional.",
            "enhanced": False,
            "status": "error"
        }

@app.post("/upload-pdf")
async def upload_pdf(files: List[UploadFile] = File(...)):
    """Upload and process PDF files to expand knowledge base"""
    try:
        processed_files = []
        
        for file in files:
            if file.content_type != "application/pdf":
                continue
                
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            # Process PDF
            loader = PyMuPDFLoader(tmp_path)
            docs = loader.load()
            
            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            
            # Add to existing vector database
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            db = Chroma(
                persist_directory="../chroma_db",
                embedding_function=embeddings
            )
            
            db.add_documents(chunks)
            
            processed_files.append({
                "filename": file.filename,
                "chunks_added": len(chunks),
                "pages": len(docs)
            })
            
            # Clean up
            os.unlink(tmp_path)
        
        return {
            "status": "success",
            "message": f"Successfully processed {len(processed_files)} PDF files",
            "files": processed_files
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing PDFs: {str(e)}"
        }

@app.post("/upload-image")
async def upload_image(files: List[UploadFile] = File(...)):
    """Upload and analyze medical images"""
    try:
        processed_images = []
        
        for file in files:
            if not file.content_type.startswith("image/"):
                continue
            
            processed_images.append({
                "filename": file.filename,
                "content_type": file.content_type,
                "message": "Image received. For detailed medical image analysis, please consult with a medical professional."
            })
        
        return {
            "status": "success",
            "message": f"Successfully received {len(processed_images)} medical images",
            "images": processed_images
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing images: {str(e)}"
        }

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "message": "Medical AI Chatbot API is running",
        "features": ["text_qa", "pdf_upload", "image_upload"]
    }

@app.get("/stats")
def get_stats():
    """Get knowledge base statistics"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        db = Chroma(
            persist_directory="../chroma_db",
            embedding_function=embeddings
        )
        
        # Get collection info
        collection = db._collection
        count = collection.count()
        
        return {
            "status": "success",
            "total_documents": count,
            "base_documents": 5895,
            "uploaded_documents": max(0, count - 5895)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
