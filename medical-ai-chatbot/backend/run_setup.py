"""
Setup script to run from backend directory
"""

import os
import sys
from pathlib import Path

# Change to parent directory
os.chdir('..')
sys.path.append('.')

def check_requirements():
    """Check if all required packages are available"""
    required_packages = [
        'langchain_community',
        'langchain',
        'sentence_transformers', 
        'chromadb',
        'pymupdf'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    return len(missing) == 0

def create_vector_db():
    """Create the vector database"""
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        
        # Check PDF exists
        pdf_path = "data/Medical_book.pdf"
        if not Path(pdf_path).exists():
            print(f"❌ PDF not found: {pdf_path}")
            return False
        
        print("📚 Loading PDF...")
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        print(f"✅ Loaded {len(docs)} pages")
        
        print("🔪 Splitting documents...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        print(f"✅ Created {len(chunks)} chunks")
        
        print("🤖 Loading embeddings (this may take a while)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("✅ Embeddings loaded")
        
        print("💾 Creating vector database...")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
        db.persist()
        print("✅ Vector database created")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating vector DB: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_chain():
    """Test the RAG chain"""
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        print("🔍 Testing RAG chain...")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        db = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
        
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        # Test retrieval
        test_docs = retriever.get_relevant_documents("What is diabetes?")
        print(f"✅ Retrieved {len(test_docs)} documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Medical RAG Setup")
    print("=" * 40)
    
    print(f"Current directory: {os.getcwd()}")
    
    print("\n1. Checking requirements...")
    if not check_requirements():
        print("❌ Missing packages. Run: pip install -r requirements.txt")
        return
    
    print("\n2. Creating vector database...")
    if not create_vector_db():
        print("❌ Failed to create vector database")
        return
    
    print("\n3. Testing RAG chain...")
    if not test_rag_chain():
        print("❌ Failed to test RAG chain")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Start API: uvicorn backend.api:app --reload")
    print("2. Start UI: streamlit run frontend/app.py")

if __name__ == "__main__":
    main()