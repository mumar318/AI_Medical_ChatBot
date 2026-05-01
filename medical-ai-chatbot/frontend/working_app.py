"""
Simple, guaranteed-working Medical AI interface
"""

import streamlit as st
import requests
import time
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide"
)

# Simple CSS for better appearance
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        border-left: 4px solid #10a37f;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    
    .bot-message {
        background-color: #e8f4f8;
        border-left-color: #10a37f;
    }
    
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 0.75rem;
    }
    
    .stButton > button {
        background-color: #10a37f;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #0d8f6f;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 Medical AI Assistant</h1>
    <p>Your intelligent medical knowledge companion</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("🔧 Controls")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.header("📊 System Status")
    
    # Test API connection
    try:
        response = requests.get("http://localhost:8000/health", timeout=3)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")
    
    st.markdown("---")
    st.header("💡 Try Asking")
    st.markdown("""
    • What is diabetes?
    • How is blood pressure measured?
    • What causes heart disease?
    • How do antibiotics work?
    """)

# Display chat messages
if st.session_state.messages:
    st.header("💬 Conversation")
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🏥 Medical AI:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)

# Input form
st.header("❓ Ask a Medical Question")

with st.form("question_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your question:",
            placeholder="e.g., What is diabetes?",
            label_visibility="collapsed"
        )
    
    with col2:
        submit = st.form_submit_button("Send 🚀")

# Process input
if submit and user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Show thinking
    with st.spinner("🤔 Thinking..."):
        try:
            # Make API request
            response = requests.post(
                "http://localhost:8000/ask",
                json={"question": user_input, "enhanced": False},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                bot_response = result.get("answer", "No response received")
            else:
                bot_response = f"❌ API Error: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            bot_response = "❌ Cannot connect to API. Please check if the backend is running."
        except requests.exceptions.Timeout:
            bot_response = "❌ Request timed out. Please try again."
        except Exception as e:
            bot_response = f"❌ Error: {str(e)}"
    
    # Add bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response
    })
    
    # Rerun to show new messages
    st.rerun()

# Welcome message if no conversation
if not st.session_state.messages:
    st.info("""
    👋 **Welcome to Medical AI Assistant!**
    
    Ask me any medical question to get started. I can provide educational information about:
    - Medical conditions and diseases
    - Symptoms and treatments
    - How medical procedures work
    - General health information
    
    *This information is for educational purposes only. Always consult healthcare professionals for medical advice.*
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏥 <strong>Medical AI Assistant</strong> | Educational use only</p>
    <p style="font-size: 0.8rem;">Powered by Emergency API | Always consult healthcare professionals</p>
</div>
""", unsafe_allow_html=True)