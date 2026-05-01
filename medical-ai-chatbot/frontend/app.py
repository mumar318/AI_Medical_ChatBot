import streamlit as st
import requests
import time
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ChatGPT-style CSS with sidebar
st.markdown("""
<style>
    /* Main layout - ChatGPT style */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 100%;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Sidebar styling - ChatGPT style */
    .css-1d391kg {
        background-color: #171717;
        padding: 1rem 0.5rem;
    }
    
    .sidebar .sidebar-content {
        background-color: #171717;
        color: white;
    }
    
    /* Sidebar header */
    .sidebar-header {
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-bottom: 1px solid #333;
    }
    
    /* New chat button */
    .new-chat-btn {
        background: #343541;
        color: white;
        border: 1px solid #565869;
        border-radius: 6px;
        padding: 0.75rem;
        width: 100%;
        margin-bottom: 1rem;
        cursor: pointer;
        font-size: 0.9rem;
        transition: background-color 0.2s;
    }
    
    .new-chat-btn:hover {
        background: #40414f;
    }
    
    /* Chat history items */
    .chat-history-item {
        background: transparent;
        color: #ececf1;
        border: none;
        border-radius: 6px;
        padding: 0.75rem;
        width: 100%;
        margin-bottom: 0.25rem;
        cursor: pointer;
        font-size: 0.85rem;
        text-align: left;
        transition: background-color 0.2s;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .chat-history-item:hover {
        background: #2a2b32;
    }
    
    .chat-history-item.active {
        background: #343541;
    }
    
    /* Main chat area */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        height: calc(100vh - 200px);
        display: flex;
        flex-direction: column;
    }
    
    /* Header */
    .chat-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .chat-header h1 {
        font-size: 1.5rem !important;
        margin: 0 !important;
    }
    
    /* Chat messages area */
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    
    /* Chat messages - ChatGPT style */
    .user-message {
        background: #343541;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        margin-left: 20%;
        position: relative;
    }
    
    .user-message::before {
        content: "You";
        position: absolute;
        top: -1.5rem;
        left: 0;
        font-size: 0.8rem;
        color: #8e8ea0;
        font-weight: 600;
    }
    
    .bot-message {
        background: #444654;
        color: #ececf1;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        margin-right: 20%;
        position: relative;
        border-left: 3px solid #10a37f;
    }
    
    .bot-message::before {
        content: "🏥 Medical AI";
        position: absolute;
        top: -1.5rem;
        left: 0;
        font-size: 0.8rem;
        color: #10a37f;
        font-weight: 600;
    }
    
    /* Input area - ChatGPT style */
    .chat-input-container {
        background: #40414f;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin: 1rem 0;
        border: 1px solid #565869;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .chat-input {
        flex: 1;
        background: transparent;
        border: none;
        color: white;
        font-size: 1rem;
        outline: none;
        resize: none;
        max-height: 120px;
    }
    
    .chat-input::placeholder {
        color: #8e8ea0;
    }
    
    /* Buttons */
    .upload-button {
        background: #10a37f;
        color: white;
        border: none;
        border-radius: 6px;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 1rem;
        transition: background-color 0.2s;
    }
    
    .upload-button:hover {
        background: #0d8f6f;
    }
    
    .send-button {
        background: #10a37f;
        color: white;
        border: none;
        border-radius: 6px;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 0.9rem;
        transition: background-color 0.2s;
    }
    
    .send-button:hover {
        background: #0d8f6f;
    }
    
    /* Welcome message */
    .welcome-message {
        text-align: center;
        padding: 2rem;
        color: #8e8ea0;
        font-size: 1.1rem;
    }
    
    .welcome-message h2 {
        color: #ececf1;
        margin-bottom: 1rem;
    }
    
    /* Hide Streamlit elements */
    .stTextInput > div > div > input {
        display: none;
    }
    
    .stButton > button {
        display: none;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .user-message, .bot-message {
            margin-left: 5%;
            margin-right: 5%;
        }
        
        .chat-container {
            height: calc(100vh - 150px);
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = {}
if 'current_session' not in st.session_state:
    st.session_state.current_session = None
if 'session_counter' not in st.session_state:
    st.session_state.session_counter = 0

# Sidebar - ChatGPT style
with st.sidebar:
    st.markdown('<div class="sidebar-header">🏥 Medical AI Chat</div>', unsafe_allow_html=True)
    
    # New Chat button
    if st.button("➕ New Chat", key="new_chat", help="Start a new conversation"):
        # Save current session if it has messages
        if st.session_state.chat_history and st.session_state.current_session:
            st.session_state.chat_sessions[st.session_state.current_session] = {
                'messages': st.session_state.chat_history.copy(),
                'title': st.session_state.chat_history[0]['content'][:30] + "..." if st.session_state.chat_history else "New Chat",
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        
        # Start new session
        st.session_state.session_counter += 1
        st.session_state.current_session = f"session_{st.session_state.session_counter}"
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    
    # Chat History
    st.markdown("### 📚 Chat History")
    
    if st.session_state.chat_sessions:
        for session_id, session_data in reversed(list(st.session_state.chat_sessions.items())):
            # Create a button for each chat session
            button_text = f"💬 {session_data['title']}"
            if st.button(button_text, key=f"load_{session_id}", help=f"Created: {session_data['timestamp']}"):
                # Save current session before switching
                if st.session_state.chat_history and st.session_state.current_session:
                    st.session_state.chat_sessions[st.session_state.current_session] = {
                        'messages': st.session_state.chat_history.copy(),
                        'title': st.session_state.chat_history[0]['content'][:30] + "..." if st.session_state.chat_history else "New Chat",
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                
                # Load selected session
                st.session_state.current_session = session_id
                st.session_state.chat_history = session_data['messages'].copy()
                st.rerun()
            
            # Delete button for each session
            if st.button("🗑️", key=f"delete_{session_id}", help="Delete this chat"):
                del st.session_state.chat_sessions[session_id]
                if st.session_state.current_session == session_id:
                    st.session_state.current_session = None
                    st.session_state.chat_history = []
                st.rerun()
    else:
        st.markdown("*No chat history yet*")
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    use_enhanced = st.checkbox("🔬 Enhanced Mode", value=False, help="Enable ML classification features")
    
    # Upload section
    st.markdown("### 📁 Upload Files")
    uploaded_pdf = st.file_uploader("📄 PDF Documents", type=['pdf'], key="pdf")
    uploaded_img = st.file_uploader("🖼️ Medical Images", type=['png','jpg','jpeg'], key="img")
    
    if uploaded_pdf:
        st.success("📄 PDF ready for processing!")
    if uploaded_img:
        st.success("🖼️ Image ready for analysis!")

# Main chat area
st.markdown("""
<div class="chat-header">
    <h1>🏥 Medical AI Assistant</h1>
    <p>Your intelligent medical knowledge companion</p>
</div>
""", unsafe_allow_html=True)

# Chat messages container
chat_container = st.container()

with chat_container:
    if st.session_state.chat_history:
        # Display chat messages
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    else:
        # Welcome message
        st.markdown("""
        <div class="welcome-message">
            <h2>👋 Welcome to Medical AI Assistant!</h2>
            <p>Ask me any medical question to get started.</p>
            <div style="margin: 1.5rem 0; font-size: 0.9rem; color: #8e8ea0;">
                <strong>💡 Try asking:</strong><br>
                • What are the symptoms of diabetes?<br>
                • How is blood pressure measured?<br>
                • What causes heart disease?<br>
                • How do antibiotics work?
            </div>
            <p style="font-size: 0.8rem; color: #666;">Educational use only - Not medical advice</p>
        </div>
        """, unsafe_allow_html=True)

# Chat input area - ChatGPT style
st.markdown("""
<div class="chat-input-container">
    <button class="upload-button" onclick="toggleUpload()" title="Upload files">📎</button>
    <input type="text" class="chat-input" placeholder="Ask about medical topics..." id="chat-input">
    <button class="send-button" onclick="sendMessage()" title="Send message">🚀</button>
</div>

<script>
function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (message) {
        // Find the hidden Streamlit input and set its value
        const hiddenInput = document.querySelector('input[data-testid="textInput-RootElement"] input');
        if (hiddenInput) {
            hiddenInput.value = message;
            hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
            
            // Find and click the submit button
            setTimeout(() => {
                const submitButton = document.querySelector('button[data-testid="baseButton-primary"]');
                if (submitButton) {
                    submitButton.click();
                }
            }, 100);
        }
        input.value = '';
    }
}

// Add enter key support
document.addEventListener('keydown', function(e) {
    const chatInput = document.getElementById('chat-input');
    if (e.key === 'Enter' && document.activeElement === chatInput) {
        e.preventDefault();
        sendMessage();
    }
});

// Ensure the chat input is focused and working
setInterval(function() {
    const chatInput = document.getElementById('chat-input');
    if (chatInput && !chatInput.hasAttribute('data-listener')) {
        chatInput.setAttribute('data-listener', 'true');
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendMessage();
            }
        });
    }
}, 1000);
</script>
""", unsafe_allow_html=True)

# Hidden Streamlit form for functionality
with st.form("chat_form", clear_on_submit=True):
    user_question = st.text_input("Type your medical question:", key="hidden_input", placeholder="Ask about medical topics...")
    submit_button = st.form_submit_button("Send Question", type="primary")

# Process input
if submit_button and user_question:
    # Create new session if none exists
    if not st.session_state.current_session:
        st.session_state.session_counter += 1
        st.session_state.current_session = f"session_{st.session_state.session_counter}"
    
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question
    })
    
    # Show processing
    with st.spinner("🤔 Thinking..."):
        try:
            # Make API request
            response = requests.post(
                "http://localhost:8000/ask",
                json={"question": user_question, "enhanced": use_enhanced},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                bot_response = result.get("answer", "No answer received")
                
                # Add enhanced info if available
                if result.get("enhanced", False):
                    question_type = result.get("question_type", "unknown")
                    confidence = result.get("classification_confidence", 0.0)
                    bot_response += f"\n\n*Question classified as: {question_type} (confidence: {confidence:.2f})*"
                    
            else:
                bot_response = f"API Error: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            bot_response = "❌ Connection Error: Cannot connect to backend API"
        except Exception as e:
            bot_response = f"❌ Error: {str(e)}"
    
    # Add bot response
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": bot_response
    })
    
    # Save session
    if st.session_state.current_session:
        st.session_state.chat_sessions[st.session_state.current_session] = {
            'messages': st.session_state.chat_history.copy(),
            'title': st.session_state.chat_history[0]['content'][:30] + "..." if st.session_state.chat_history else "New Chat",
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    
    # Rerun to show updated chat
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem; font-size: 0.8rem;">
    <p>🏥 <strong>Medical AI Assistant</strong> | Educational use only</p>
    <p style="font-size: 0.75rem; color: #999;">
        Powered by Groq Llama-3.1-8b-instant | Knowledge Base: 5,895 medical chunks
    </p>
</div>
""", unsafe_allow_html=True)