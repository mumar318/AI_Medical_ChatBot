import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="centered"
)

@st.cache_resource
def initialize_rag_chain():
    """Initialize the RAG chain"""

    # ✅ API key - st.secrets first, then .env
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("❌ GROQ_API_KEY not found! Add it in Streamlit Secrets.")
        st.stop()

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ChromaDB path - absolute path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_dir = os.path.join(base_dir, '..', 'chroma_db')

    db = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=api_key
    )

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a medical information assistant.
Use ONLY the context below to answer the question.
Do NOT diagnose or prescribe treatment.
If the answer is not in the context, say "I don't have enough information. Please consult a healthcare professional."

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# Title
st.title("🏥 Medical AI Assistant")
st.markdown("*Your intelligent medical knowledge companion*")
st.markdown("---")

# Session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load RAG chain
try:
    with st.spinner("🔄 Loading AI model..."):
        rag_chain = initialize_rag_chain()
    st.success("✅ AI model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading AI model: {str(e)}")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me any medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Thinking...")

        try:
            context_prompt = prompt

            # Follow-up question handling
            if len(prompt.split()) <= 3 and len(st.session_state.messages) > 1:
                for i in range(len(st.session_state.messages) - 2, -1, -1):
                    if st.session_state.messages[i]["role"] == "user":
                        previous_question = st.session_state.messages[i]["content"]
                        context_prompt = f"Previous question: {previous_question}\n\nFollow-up: {prompt}"
                        break

            full_response = rag_chain.invoke(context_prompt)

        except Exception as e:
            full_response = f"❌ Error: {str(e)}\n\nPlease consult a healthcare professional."

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Medical AI Assistant** uses:
    - 🤖 Groq Llama-3.1-8b-instant
    - 📚 5,895 medical documents
    - 🔍 RAG Technology

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
    st.markdown("*Educational use only. Not medical advice.*")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
    "🏥 Medical AI Assistant | Powered by Groq & LangChain"
    "</div>",
    unsafe_allow_html=True
)