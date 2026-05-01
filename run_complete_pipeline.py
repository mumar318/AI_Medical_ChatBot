"""
Complete RAG Pipeline Runner
Runs all components of the Medical AI Knowledge Helper
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "data/Medical_book.pdf",
        "backend/ingest.py",
        "backend/rag_chain.py",
        "backend/api.py",
        "frontend/app.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found")
    return True

def run_data_ingestion():
    """Run data ingestion process"""
    print("\n📚 Running data ingestion...")
    try:
        result = subprocess.run([sys.executable, "backend/ingest.py"], 
                              capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print("✅ Data ingestion completed successfully")
            print(result.stdout)
        else:
            print("❌ Data ingestion failed")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running data ingestion: {e}")
        return False
    
    return True

def run_ml_components():
    """Run ML components training"""
    print("\n🤖 Training ML components...")
    try:
        result = subprocess.run([sys.executable, "backend/ml_components.py"], 
                              capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print("✅ ML components trained successfully")
            print(result.stdout)
        else:
            print("❌ ML components training failed")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error training ML components: {e}")
        return False
    
    return True

def run_evaluation():
    """Run RAG evaluation"""
    print("\n📊 Running RAG evaluation...")
    try:
        result = subprocess.run([sys.executable, "backend/evaluation.py"], 
                              capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print("✅ Evaluation completed successfully")
            print(result.stdout)
        else:
            print("❌ Evaluation failed")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False
    
    return True

def main():
    """Run the complete pipeline"""
    print("🚀 Starting Medical AI Knowledge Helper Pipeline")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Pipeline aborted due to missing files")
        return
    
    # Run data ingestion
    if not run_data_ingestion():
        print("\n❌ Pipeline aborted due to ingestion failure")
        return
    
    # Run ML components
    if not run_ml_components():
        print("\n⚠️  ML components failed, but continuing with basic RAG")
    
    # Run evaluation
    if not run_evaluation():
        print("\n⚠️  Evaluation failed, but RAG system should still work")
    
    print("\n" + "=" * 50)
    print("🎉 Pipeline completed!")
    print("\nNext steps:")
    print("1. Start the API server:")
    print("   uvicorn backend.api:app --reload")
    print("\n2. In another terminal, start the frontend:")
    print("   streamlit run frontend/app.py")
    print("\n3. Open your browser and go to the Streamlit URL")
    print("\n📁 Generated files:")
    print("   - chroma_db/ (vector database)")
    print("   - question_classifier.pkl (ML model)")
    print("   - document_clusters.json (clustering results)")
    print("   - cluster_summaries.json (document summaries)")
    print("   - evaluation_results.json (evaluation metrics)")

if __name__ == "__main__":
    main()