"""
RAG Evaluation Module
Implements evaluation metrics for retrieval quality and answer relevance
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag_chain import ask_medical_bot

class RAGEvaluator:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db = Chroma(
            persist_directory="chroma_db",
            embedding_function=self.embeddings
        )
        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})
        
    def evaluate_retrieval_relevance(self, query: str, retrieved_docs: List[str]) -> float:
        """
        Evaluate how relevant retrieved documents are to the query
        Returns average cosine similarity between query and retrieved docs
        """
        query_embedding = self.embeddings.embed_query(query)
        doc_embeddings = [self.embeddings.embed_query(doc) for doc in retrieved_docs]
        
        similarities = []
        for doc_emb in doc_embeddings:
            similarity = cosine_similarity([query_embedding], [doc_emb])[0][0]
            similarities.append(similarity)
            
        return np.mean(similarities)
    
    def evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """
        Evaluate how relevant the generated answer is to the query
        """
        query_embedding = self.embeddings.embed_query(query)
        answer_embedding = self.embeddings.embed_query(answer)
        
        similarity = cosine_similarity([query_embedding], [answer_embedding])[0][0]
        return similarity
    
    def evaluate_context_precision(self, query: str, retrieved_docs: List[str], k: int = 3) -> float:
        """
        Evaluate precision of retrieved context
        Measures how many of the top-k retrieved docs are relevant
        """
        relevance_scores = []
        query_embedding = self.embeddings.embed_query(query)
        
        for doc in retrieved_docs[:k]:
            doc_embedding = self.embeddings.embed_query(doc)
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            # Consider relevant if similarity > 0.5
            relevance_scores.append(1 if similarity > 0.5 else 0)
            
        return np.mean(relevance_scores)
    
    def run_evaluation_suite(self, test_queries: List[Dict]) -> Dict:
        """
        Run comprehensive evaluation on test queries
        test_queries format: [{"query": "...", "expected_topics": [...]}]
        """
        results = {
            "retrieval_relevance": [],
            "answer_relevance": [],
            "context_precision": [],
            "detailed_results": []
        }
        
        for test_case in test_queries:
            query = test_case["query"]
            
            # Get retrieved documents
            retrieved_docs = self.retriever.get_relevant_documents(query)
            retrieved_texts = [doc.page_content for doc in retrieved_docs]
            
            # Get RAG answer
            answer = ask_medical_bot(query)
            
            # Calculate metrics
            retrieval_rel = self.evaluate_retrieval_relevance(query, retrieved_texts)
            answer_rel = self.evaluate_answer_relevance(query, answer)
            context_prec = self.evaluate_context_precision(query, retrieved_texts)
            
            results["retrieval_relevance"].append(retrieval_rel)
            results["answer_relevance"].append(answer_rel)
            results["context_precision"].append(context_prec)
            
            results["detailed_results"].append({
                "query": query,
                "answer": answer,
                "retrieved_docs": retrieved_texts[:2],  # First 2 for brevity
                "retrieval_relevance": retrieval_rel,
                "answer_relevance": answer_rel,
                "context_precision": context_prec
            })
        
        # Calculate averages
        results["avg_retrieval_relevance"] = np.mean(results["retrieval_relevance"])
        results["avg_answer_relevance"] = np.mean(results["answer_relevance"])
        results["avg_context_precision"] = np.mean(results["context_precision"])
        
        return results

def run_evaluation():
    """Run evaluation with sample medical queries"""
    
    # Sample test queries for medical domain
    test_queries = [
        {
            "query": "What are the symptoms of diabetes?",
            "expected_topics": ["diabetes", "symptoms", "blood sugar"]
        },
        {
            "query": "How is blood pressure measured?",
            "expected_topics": ["blood pressure", "measurement", "systolic", "diastolic"]
        },
        {
            "query": "What causes heart disease?",
            "expected_topics": ["heart disease", "cardiovascular", "causes", "risk factors"]
        },
        {
            "query": "What are the side effects of antibiotics?",
            "expected_topics": ["antibiotics", "side effects", "medication"]
        },
        {
            "query": "How to treat a fever?",
            "expected_topics": ["fever", "treatment", "temperature"]
        }
    ]
    
    evaluator = RAGEvaluator()
    results = evaluator.run_evaluation_suite(test_queries)
    
    # Print results
    print("=== RAG System Evaluation Results ===\n")
    print(f"Average Retrieval Relevance: {results['avg_retrieval_relevance']:.3f}")
    print(f"Average Answer Relevance: {results['avg_answer_relevance']:.3f}")
    print(f"Average Context Precision: {results['avg_context_precision']:.3f}")
    
    print("\n=== Detailed Results ===")
    for i, result in enumerate(results["detailed_results"]):
        print(f"\nQuery {i+1}: {result['query']}")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Retrieval Relevance: {result['retrieval_relevance']:.3f}")
        print(f"Answer Relevance: {result['answer_relevance']:.3f}")
        print(f"Context Precision: {result['context_precision']:.3f}")
        print("-" * 50)
    
    # Save results to file
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nEvaluation complete! Results saved to evaluation_results.json")

if __name__ == "__main__":
    run_evaluation()