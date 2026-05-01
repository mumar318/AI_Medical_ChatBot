import os
import sys
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def main():
    # Check if PDF file exists
    pdf_path = "data/Medical_book.pdf"
    if not Path(pdf_path).exists():
        print(f"❌ Error: PDF file not found at {pdf_path}")
        print("Please ensure the medical PDF is placed in the data/ folder")
        sys.exit(1)
    
    try:
        print("📚 Loading medical PDF...")
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        print(f"✅ Loaded {len(docs)} pages from medical book")
        
        if len(docs) == 0:
            print("❌ Error: No content found in PDF")
            sys.exit(1)
        
        print("🔪 Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        print(f"✅ Created {len(chunks)} text chunks")
        
        print("🤖 Initializing embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("✅ Embedding model loaded")
        
        print("💾 Creating vector database...")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
        
        db.persist()
        print("✅ Medical documents ingested successfully!")
        print(f"📊 Database contains {len(chunks)} document chunks")
        
    except Exception as e:
        print(f"❌ Error during ingestion: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
