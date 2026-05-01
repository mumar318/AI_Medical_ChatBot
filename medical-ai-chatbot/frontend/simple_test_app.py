"""
Simple test app to verify basic functionality
"""

import streamlit as st
import requests

st.title("🧪 Medical AI Test")

st.write("Testing basic functionality...")

# Simple form
with st.form("test_form"):
    question = st.text_input("Ask a question:")
    submit = st.form_submit_button("Test API")
    
    if submit and question:
        st.write(f"You asked: {question}")
        
        try:
            # Test API call
            response = requests.post(
                "http://localhost:8000/ask",
                json={"question": question, "enhanced": False},
                timeout=10
            )
            
            st.write(f"API Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "No answer")
                st.success("✅ API Response Received!")
                st.write(f"Answer length: {len(answer)} characters")
                st.text_area("Response:", answer, height=200)
            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.write(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Direct API test button
if st.button("Test API Health"):
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API is healthy!")
            st.json(response.json())
        else:
            st.error(f"❌ API health check failed: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Health check error: {e}")

st.write("---")
st.write("If this works, the issue is with the main app interface.")
st.write("If this doesn't work, the issue is with the API connection.")