"""
Enhanced RAG Chain with ML Components Integration
"""

import os
import pickle
from typing import List, Dict, Tuple
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from ml_components import QuestionClassifier, DocumentClusterer

class EnhancedRAGChain:
    def __init__(self):
        # Initialize base components
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.db = Chroma(
            persist_directory="chroma_db",
            embedding_function=self.embeddings
        )
        
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )
        
        # Load ML components if available
        self.question_classifier = self._load_question_classifier()
        self.document_clusterer = self._load_document_clusterer()
        
        # Enhanced prompt template
        self.enhanced_prompt = PromptTemplate(
            input_variables=["context", "question", "question_type"],
            template="""
You are a medical information assistant specialized in {question_type} questions.
Use ONLY the context below to answer the question.
Do NOT diagnose or prescribe treatment.
If information is not found in the context, say "Please consult a healthcare professional."

Context:
{context}

Question: {question}

Answer:"""
        )
        
    def _load_question_classifier(self):
        """Load trained question classifier if available"""
        try:
            with open('question_classifier.pkl', 'rb') as f:
                model_data = pickle.load(f)
                classifier = QuestionClassifier()
                classifier.vectorizer = model_data['vectorizer']
                classifier.classifier = model_data['classifier']
                classifier.categories = model_data['categories']
                return classifier
        except FileNotFoundError:
            print("Question classifier not found. Using basic RAG.")
            return None
    
    def _load_document_clusterer(self):
        """Load document clusterer if available"""
        try:
            clusterer = DocumentClusterer()
            # Load clustering results if they exist
            return clusterer
        except:
            print("Document clusterer not available. Using basic retrieval.")
            return None
    
    def _get_enhanced_retriever(self, question: str, question_type: str = None):
        """Get retriever with enhanced filtering based on question type"""
        
        # Base retriever
        base_retriever = self.db.as_retriever(search_kwargs={"k": 5})
        
        # If we have question classification, adjust retrieval
        if question_type:
            # Modify search based on question type
            if question_type == "symptoms":
                # For symptom questions, get more diverse results
                return self.db.as_retriever(search_kwargs={"k": 4})
            elif question_type == "treatment":
                # For treatment questions, focus on precision
                return self.db.as_retriever(search_kwargs={"k": 3})
            elif question_type == "medication":
                # For medication questions, get comprehensive info
                return self.db.as_retriever(search_kwargs={"k": 5})
        
        return base_retriever
    
    def ask_enhanced(self, question: str) -> Dict[str, str]:
        """Enhanced RAG with ML components"""
        
        # Classify question if classifier is available
        question_type = "general"
        confidence = 0.0
        
        if self.question_classifier:
            question_type, confidence = self.question_classifier.classify_question(question)
        
        # Get enhanced retriever
        retriever = self._get_enhanced_retriever(question, question_type)
        
        # Create enhanced QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": self.enhanced_prompt}
        )
        
        # Get answer with enhanced context
        answer = qa_chain.run({
            "question": question,
            "question_type": question_type
        })
        
        return {
            "answer": answer,
            "question_type": question_type,
            "classification_confidence": confidence,
            "enhancement_used": self.question_classifier is not None
        }

# Global enhanced RAG instance
enhanced_rag = EnhancedRAGChain()

def ask_enhanced_medical_bot(question: str) -> Dict[str, str]:
    """Enhanced medical bot with ML components"""
    return enhanced_rag.ask_enhanced(question)