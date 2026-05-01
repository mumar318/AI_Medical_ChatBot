"""
Simplified RAG implementation that works with current langchain versions
"""

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

# Load environment variables from the .env file in parent directory
load_dotenv('../.env')

# Check if API key is loaded
api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    print("Warning: GROQ_API_KEY not found in environment variables")
    print("Please check your .env file")
    # Try to load from current directory as fallback
    load_dotenv('.env')
    api_key = os.getenv('GROQ_API_KEY')

# Initialize components
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="../chroma_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# Initialize LLM with updated model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=api_key
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a medical information assistant.
Use ONLY the context below.
Do NOT diagnose or prescribe treatment.
If not found, say consult a healthcare professional.

Context:
{context}

Question: {question}

Answer:"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask_medical_bot(question: str) -> str:
    """Ask the medical bot a question"""
    try:
        return rag_chain.invoke(question)
    except Exception as e:
        return f"Error: {str(e)}. Please consult a healthcare professional."