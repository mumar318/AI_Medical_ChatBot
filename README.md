# 🏥 AI Medical Chatbot

A comprehensive RAG (Retrieval-Augmented Generation) system for medical information retrieval and question answering, built with FastAPI, Streamlit, and advanced ML components.

## 🎯 Features

- **📚 Medical Knowledge Base**: 5,895 document chunks from medical literature
- **🤖 AI-Powered Responses**: Groq Llama-3.1-8b-instant for accurate medical Q&A
- **📱 Mobile-Friendly UI**: Responsive design that works on all devices
- **📄 File Upload**: Support for PDF documents and medical images
- **🔬 Enhanced Mode**: ML classification, clustering, and summarization
- **⚡ Fast Performance**: Optimized for quick response times

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Groq API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/mumar318/AI_Medicalchatbot.git
cd AI_Medicalchatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file with your Groq API key
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

4. **Run the complete setup**
```bash
python run_complete_pipeline.py
```

5. **Start the application**
```bash
# Terminal 1: Start backend API
cd backend
python -m uvicorn api:app --reload --port 8000

# Terminal 2: Start frontend UI
cd frontend
python -m streamlit run app.py --server.port=8501
```

6. **Access the application**
- Local: http://localhost:8501
- Network: http://your-ip:8501

## 📁 Project Structure

```
AI_Medicalchatbot/
├── backend/
│   ├── api.py              # FastAPI backend
│   ├── simple_rag.py       # RAG implementation
│   ├── enhanced_rag.py     # Enhanced RAG with ML
│   ├── ml_components.py    # ML classification/clustering
│   ├── evaluation.py       # System evaluation
│   ├── ingest.py          # Data processing
│   └── run_setup.py       # Setup script
├── frontend/
│   ├── app.py             # Streamlit interface
│   └── .streamlit/        # Configuration
├── chroma_db/             # Vector database
├── data/                  # Medical dataset
├── notebooks/             # Data processing
├── .env                   # API keys (create this)
├── README.md              # This file
├── requirements.txt       # Dependencies
├── run_complete_pipeline.py # Complete setup
├── run_streamlit.py       # UI launcher
├── SYSTEM_STATUS.md       # System status
└── PART_4_REPORT_REFLECTION.md # Development report
```

## 🎯 Usage

### Basic Usage
1. Open http://localhost:8501 in your browser
2. Type medical questions like "What is diabetes?"
3. Click "🚀 Ask" to get AI-powered responses
4. Use "➕" button to upload PDF documents or images

### Enhanced Mode
- Toggle "🔬 Enhanced Mode" for ML features
- Get question classification and improved retrieval
- View confidence scores for AI responses

### Example Questions
- "What are the symptoms of diabetes?"
- "How is blood pressure measured?"
- "What causes heart disease?"
- "What are the side effects of antibiotics?"

## 🔧 Technical Details

### Architecture
- **Backend**: FastAPI with RAG pipeline
- **Frontend**: Streamlit with responsive design
- **Database**: ChromaDB vector database
- **LLM**: Groq Llama-3.1-8b-instant
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2

### ML Components
1. **Question Classification**: Categorizes medical queries
2. **Document Clustering**: Groups similar content
3. **Summarization**: Generates context summaries

### Performance
- **Response Time**: < 3 seconds average
- **Knowledge Base**: 5,895 medical document chunks
- **Concurrent Users**: Supports multiple simultaneous users
- **Mobile Optimized**: Works on phones, tablets, desktops

## 📊 Assignment Components

This project fulfills all assignment requirements:

- ✅ **Part 1**: Data processing and embeddings (notebooks/data_processing.ipynb)
- ✅ **Part 2**: Complete RAG implementation (backend/simple_rag.py)
- ✅ **Part 3**: All ML components implemented (backend/ml_components.py)
- ✅ **Part 4**: Comprehensive report (PART_4_REPORT_REFLECTION.md)

## 🛠️ Development

### Running in Development Mode
```bash
# Backend with auto-reload
cd backend
python -m uvicorn api:app --reload --port 8000

# Frontend with auto-reload
cd frontend
streamlit run app.py --server.port=8501
```

### Testing
The system includes comprehensive evaluation metrics and testing capabilities.

## 📝 Documentation

- **System Status**: See SYSTEM_STATUS.md
- **Development Report**: See PART_4_REPORT_REFLECTION.md
- **Data Processing**: See notebooks/data_processing.ipynb

## ⚠️ Important Notes

- **Educational Use Only**: Not for medical diagnosis or treatment
- **API Key Required**: Get your Groq API key from https://console.groq.com/
- **Data Privacy**: All processing is done locally
- **Medical Disclaimer**: Always consult healthcare professionals

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please ensure compliance with medical data regulations in your jurisdiction.

## 🙏 Acknowledgments

- Medical literature dataset for knowledge base
- Groq for fast LLM inference
- ChromaDB for vector storage
- Streamlit for rapid UI development

---

**🌐 Live Demo**: http://localhost:8501 (when running locally)

**📧 Contact**: For questions about this educational project

**⭐ Star this repo** if you find it helpful for learning RAG systems!
