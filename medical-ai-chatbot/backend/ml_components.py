"""
Optional ML Components for RAG Enhancement
Implements question classification, document clustering, and summarization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
import pickle
import json

class QuestionClassifier:
    """
    Classifies medical questions into categories to improve RAG routing
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = LogisticRegression(random_state=42)
        self.categories = [
            'symptoms', 'treatment', 'diagnosis', 'prevention', 
            'medication', 'anatomy', 'general_info'
        ]
        
    def create_training_data(self) -> Tuple[List[str], List[str]]:
        """Create synthetic training data for question classification"""
        
        training_data = [
            # Symptoms
            ("What are the symptoms of diabetes?", "symptoms"),
            ("How do I know if I have a fever?", "symptoms"),
            ("What does chest pain indicate?", "symptoms"),
            ("Signs of high blood pressure", "symptoms"),
            ("Symptoms of heart attack", "symptoms"),
            
            # Treatment
            ("How to treat a cold?", "treatment"),
            ("What is the treatment for diabetes?", "treatment"),
            ("How to manage high blood pressure?", "treatment"),
            ("Treatment options for cancer", "treatment"),
            ("How to treat a wound?", "treatment"),
            
            # Diagnosis
            ("How is diabetes diagnosed?", "diagnosis"),
            ("What tests are used for heart disease?", "diagnosis"),
            ("How do doctors diagnose cancer?", "diagnosis"),
            ("Blood tests for liver function", "diagnosis"),
            ("MRI scan for brain tumor", "diagnosis"),
            
            # Prevention
            ("How to prevent heart disease?", "prevention"),
            ("Ways to avoid diabetes", "prevention"),
            ("Preventing high blood pressure", "prevention"),
            ("How to prevent infections?", "prevention"),
            ("Cancer prevention methods", "prevention"),
            
            # Medication
            ("Side effects of antibiotics", "medication"),
            ("What medications treat hypertension?", "medication"),
            ("Insulin dosage for diabetes", "medication"),
            ("Pain medication options", "medication"),
            ("Drug interactions to avoid", "medication"),
            
            # Anatomy
            ("What is the function of the heart?", "anatomy"),
            ("How do kidneys work?", "anatomy"),
            ("Structure of the brain", "anatomy"),
            ("What are blood vessels?", "anatomy"),
            ("Function of the liver", "anatomy"),
            
            # General Info
            ("What is blood pressure?", "general_info"),
            ("Definition of diabetes", "general_info"),
            ("What is cholesterol?", "general_info"),
            ("Types of cancer", "general_info"),
            ("What is metabolism?", "general_info")
        ]
        
        questions, labels = zip(*training_data)
        return list(questions), list(labels)
    
    def train(self):
        """Train the question classifier"""
        questions, labels = self.create_training_data()
        
        # Vectorize questions
        X = self.vectorizer.fit_transform(questions)
        
        # Train classifier
        self.classifier.fit(X, labels)
        
        # Evaluate on training data (for demonstration)
        predictions = self.classifier.predict(X)
        print("Question Classifier Training Results:")
        print(classification_report(labels, predictions))
        
        # Save model
        with open('question_classifier.pkl', 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'categories': self.categories
            }, f)
    
    def classify_question(self, question: str) -> Tuple[str, float]:
        """Classify a question and return category with confidence"""
        X = self.vectorizer.transform([question])
        prediction = self.classifier.predict(X)[0]
        confidence = max(self.classifier.predict_proba(X)[0])
        
        return prediction, confidence

class DocumentClusterer:
    """
    Clusters medical documents to improve RAG retrieval
    """
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = None
        
    def load_documents(self) -> List[str]:
        """Load documents from ChromaDB"""
        db = Chroma(
            persist_directory="chroma_db",
            embedding_function=self.embeddings
        )
        
        # Get all documents
        collection = db._collection
        results = collection.get()
        documents = results['documents']
        
        return documents
    
    def cluster_documents(self) -> Dict:
        """Cluster documents and analyze clusters"""
        documents = self.load_documents()
        
        # Generate embeddings
        print("Generating embeddings for clustering...")
        doc_embeddings = [self.embeddings.embed_query(doc) for doc in documents]
        
        # Perform clustering
        print(f"Clustering {len(documents)} documents into {self.n_clusters} clusters...")
        self.cluster_labels = self.kmeans.fit_predict(doc_embeddings)
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(self.n_clusters):
            cluster_docs = [doc for j, doc in enumerate(documents) if self.cluster_labels[j] == i]
            
            # Get representative terms using TF-IDF
            vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
            if len(cluster_docs) > 0:
                tfidf_matrix = vectorizer.fit_transform(cluster_docs)
                feature_names = vectorizer.get_feature_names_out()
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                top_terms = [feature_names[idx] for idx in mean_scores.argsort()[-5:][::-1]]
            else:
                top_terms = []
            
            cluster_analysis[f"cluster_{i}"] = {
                "size": len(cluster_docs),
                "top_terms": top_terms,
                "sample_docs": cluster_docs[:2]  # First 2 docs as samples
            }
        
        # Save clustering results
        with open('document_clusters.json', 'w') as f:
            json.dump(cluster_analysis, f, indent=2)
        
        print("Document clustering complete!")
        return cluster_analysis
    
    def get_cluster_for_query(self, query: str) -> int:
        """Find the best cluster for a given query"""
        if self.cluster_labels is None:
            raise ValueError("Documents must be clustered first")
        
        query_embedding = self.embeddings.embed_query(query)
        cluster_centers = self.kmeans.cluster_centers_
        
        # Find closest cluster center
        distances = [np.linalg.norm(query_embedding - center) for center in cluster_centers]
        return np.argmin(distances)

class DocumentSummarizer:
    """
    Generates automatic summaries of document clusters
    """
    
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )
        
    def summarize_cluster(self, documents: List[str], max_docs: int = 5) -> str:
        """Summarize a cluster of documents"""
        
        # Limit number of documents for summarization
        docs_to_summarize = documents[:max_docs]
        
        # Convert to Document objects
        doc_objects = [Document(page_content=doc) for doc in docs_to_summarize]
        
        # Create summarization chain
        summarize_chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce"
        )
        
        # Generate summary
        summary = summarize_chain.run(doc_objects)
        return summary
    
    def create_cluster_summaries(self, cluster_analysis: Dict) -> Dict:
        """Create summaries for all document clusters"""
        
        summaries = {}
        
        for cluster_id, cluster_info in cluster_analysis.items():
            if cluster_info["size"] > 0:
                print(f"Summarizing {cluster_id}...")
                
                # Load full documents for this cluster (simplified approach)
                sample_docs = cluster_info["sample_docs"]
                
                if len(sample_docs) > 0:
                    summary = self.summarize_cluster(sample_docs)
                    summaries[cluster_id] = {
                        "summary": summary,
                        "top_terms": cluster_info["top_terms"],
                        "size": cluster_info["size"]
                    }
        
        # Save summaries
        with open('cluster_summaries.json', 'w') as f:
            json.dump(summaries, f, indent=2)
        
        return summaries

def run_ml_components():
    """Run all ML components"""
    
    print("=== Running ML Components ===\n")
    
    # 1. Question Classification
    print("1. Training Question Classifier...")
    classifier = QuestionClassifier()
    classifier.train()
    
    # Test classification
    test_questions = [
        "What are the symptoms of flu?",
        "How to treat diabetes?",
        "What tests diagnose cancer?"
    ]
    
    print("\nTesting Question Classification:")
    for question in test_questions:
        category, confidence = classifier.classify_question(question)
        print(f"'{question}' -> {category} (confidence: {confidence:.3f})")
    
    # 2. Document Clustering
    print("\n2. Clustering Documents...")
    clusterer = DocumentClusterer(n_clusters=3)
    cluster_analysis = clusterer.cluster_documents()
    
    print("\nCluster Analysis:")
    for cluster_id, info in cluster_analysis.items():
        print(f"{cluster_id}: {info['size']} docs, top terms: {info['top_terms']}")
    
    # 3. Document Summarization
    print("\n3. Generating Cluster Summaries...")
    summarizer = DocumentSummarizer()
    summaries = summarizer.create_cluster_summaries(cluster_analysis)
    
    print("\nCluster Summaries Generated:")
    for cluster_id, summary_info in summaries.items():
        print(f"{cluster_id}: {summary_info['summary'][:100]}...")
    
    print("\n=== ML Components Complete ===")
    print("Files generated:")
    print("- question_classifier.pkl")
    print("- document_clusters.json") 
    print("- cluster_summaries.json")

if __name__ == "__main__":
    run_ml_components()