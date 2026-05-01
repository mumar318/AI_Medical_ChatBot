import streamlit as st
import requests
import time

# Page configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="centered"
)

# Title
st.title("🏥 Medical AI Assistant")
st.markdown("*Your intelligent medical knowledge companion*")
st.markdown("---")

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me any medical question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Thinking...")
        
        try:
            # Make API request
            response = requests.post(
                "http://localhost:8000/ask",
                json={"question": prompt, "enhanced": False},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                full_response = result.get("answer", "No answer received")
            else:
                full_response = f"❌ API Error: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            full_response = "❌ Connection Error: Cannot connect to backend API. Please ensure the backend is running."
        except requests.exceptions.Timeout:
            full_response = "❌ Timeout Error: The request took too long. Please try again."
        except Exception as e:
            full_response = f"❌ Error: {str(e)}"
        
        # Display the response
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This is an AI-powered medical information assistant.
    
    **Features:**
    - Medical Q&A
    - Knowledge base of 5,895 medical documents
    - Powered by Groq Llama-3.1-8b-instant
    
    **Example Questions:**
    - What are the symptoms of diabetes?
    - How is blood pressure measured?
    - What causes heart disease?
    - How do antibiotics work?
    """)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**⚠️ Disclaimer**")
    st.markdown("*Educational use only. Not medical advice. Always consult healthcare professionals.*")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
    "🏥 Medical AI Assistant | Powered by Groq & LangChain"
    "</div>",
    unsafe_allow_html=True
)
