#!/usr/bin/env python3

import requests
import json

def test_api():
    try:
        print("Testing API health...")
        response = requests.get("http://127.0.0.1:8000/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        print("\nTesting RAG query...")
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"question": "What is diabetes?"}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Answer length: {len(result['answer'])}")
        print(f"Answer: {result['answer'][:200]}...")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()