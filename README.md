# 🏛️ Legal Advisory Chatbot

A **Legal Advisory Chatbot** designed to help EU businesses comply with **DORA (Digital Operational Resilience Act)** regulations using advanced Retrieval-Augmented Generation (RAG) techniques.

## 🎯 Project Overview

This project provides an intelligent legal advisory system that assists EU businesses in understanding and complying with DORA regulations. The chatbot delivers precise, context-aware legal guidance by retrieving relevant regulatory information and generating professional advisory responses.

### Key Features
- 🔍 **Intelligent Document Retrieval**: Advanced RAG pipeline with semantic search and reranking
- ⚖️ **Legal Expert Responses**: Professional legal advisor tone and terminology  
- 🎯 **DORA Regulation Focus**: Specialized knowledge base for EU digital operational resilience
- 🚀 **Flexible LLM Options**: Support for both local (Ollama) and cloud-based (Gemini) models
- 📊 **Comprehensive Analytics**: Detailed retrieval analysis and reranking insights

## 🏗️ Technical Architecture

### RAG Pipeline
```
📄 DORA PDF → 🔪 Chunking → 🧮 Embeddings → 🗄️ FAISS Vector DB
                                                        ↓
💬 User Query → 🔍 Retrieval → 🎯 Reranking → 🤖 LLM Generation → 📝 Response
```

### Tech Stack
- **📄 Document Processing**: PDF parsing and intelligent text chunking
- **🧮 Embeddings**: `BAAI/bge-large-en-v1.5` (1024 dimensions)
- **🗄️ Vector Database**: FAISS for efficient similarity search
- **🔍 Retrieval**: Semantic search with configurable top-k
- **🎯 Reranking**: `BAAI/bge-reranker-base` cross-encoder for relevance refinement
- **🤖 LLM Generation**: 
  - **Local**: Ollama (llama3.2-3B-Instruct)
  - **Cloud**: Google Gemini (2.0-flash, 1.5-flash, 1.5-pro)
- **🖥️ User Interface**: Streamlit for interactive PoC demonstration

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+
pip install -r requirements.txt

# For local LLM (optional)
# Install Ollama: https://ollama.ai/
ollama pull llama3.2

# For Gemini LLM
# in the .env file:
GOOGLE_API_KEY = <YOUR_GOOGLE_API_KEY>
```

### Environment Setup
1. Create a `.env` file:
```env
# Required for Gemini LLM
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

2. Ensure DORA document is available:
```
📁 RAG_model/
├── DORA.pdf                    # Source regulation document
├── dora_chunks_simple.json     # Pre-processed chunks
└── ...
```

### Running the Application
```bash
# Navigate to project directory
cd RAG_model

# Launch Streamlit interface
streamlit run frontend/app.py
```

Visit `http://localhost:8501` to access the chatbot interface.

## 📊 System Configuration

### Model Selection
- **🏠 Local Processing**: Choose Ollama for privacy and offline operation
- **☁️ Cloud Processing**: Choose Gemini for faster responses and advanced capabilities

### Retrieval Settings
- **Initial Retrieval**: 15-50 documents from vector database
- **Reranking**: Cross-encoder refinement for top-k selection
- **Final Output**: 5-15 most relevant chunks for context

### Performance Optimization
- **Smart Caching**: Embeddings and models cached for faster startup
- **Batch Processing**: Efficient document chunking and embedding generation
- **GPU Support**: Optional GPU acceleration for embedding models

## 🔧 Project Structure

```
RAG_model/
├── src/
│   ├── embeddings.py          # Embedding model and FAISS management
│   ├── reranker.py            # Cross-encoder reranking logic
│   ├── main.py                # RAG orchestration and LLM chains
│   ├── utils.py               # Utility functions and document processing
│   └── generator_prompt.txt   # Legal advisor system prompt
├── frontend/
│   └── app.py                 # Streamlit user interface
├── embeddings_index/
│   └── faiss.index           # FAISS vector database
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
└── README.md                 # This file
```

## 🎯 Use Cases

### Target Audience
- **🏢 EU Financial Institutions**: Banks, investment firms, insurance companies
- **💳 Payment Service Providers**: Payment processors, e-money institutions
- **📊 Data Service Providers**: Trade repositories, data reporting services
- **🏗️ ICT Service Providers**: Third-party technology vendors

### Example Queries
```
"What are the requirements for financial entities?"
"What are the ICT risk management requirements for financial entities?"
"How should we implement business continuity plans under DORA?"
"What are the reporting obligations for major ICT-related incidents?"
"What governance arrangements are required for ICT third-party risk?"
```

## 📈 Advanced Features

### Retrieval Analytics
- **📊 Stage-by-Stage Visibility**: Track initial retrieval → reranking → final selection
- **🎯 Relevance Scoring**: Detailed reranking scores and position changes
- **📝 Document Metadata**: Source tracking and content analysis
- **⏱️ Performance Metrics**: Response times and processing statistics

### Legal Advisory Mode
- **⚖️ Professional Tone**: Responses formatted as legal advisory guidance
- **📖 Article References**: Specific regulatory citations when available
- **🔍 Compliance Focus**: Actionable steps for regulatory adherence
- **❌ Hallucination Prevention**: Strict context-only responses

## 🛠️ Development

### Adding New Documents
1. Place PDF in project root
2. Update `CHUNKS_FILE` in `src/main.py`
3. Run chunking process: `python src/utils.py`
4. Restart application to rebuild embeddings

### Customizing the Legal Prompt
Edit `src/generator_prompt.txt` to modify the legal advisor persona and response format.

### Model Upgrades
- **Embeddings**: Update `MODEL_NAME` in `src/embeddings.py`
- **Reranking**: Update `RERANKING_MODEL` in `src/reranker.py`
- **LLM**: Add new models to frontend selection options

## 📋 Requirements

### Core Dependencies
- `streamlit>=1.25.0` - Web interface
- `langchain-core>=0.3.29` - LLM framework
- `langchain-google-genai>=2.0.0` - Gemini integration
- `sentence-transformers>=2.2.2` - Embeddings and reranking
- `faiss-cpu>=1.7.4` - Vector database
- `pdfplumber>=0.10.1` - PDF processing
- `python-dotenv>=1.0.0` - Environment management

### Optional Dependencies
- `faiss-gpu` - GPU acceleration for vector operations
- `langchain-ollama` - Local LLM support

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m "Add new feature"`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Resources

- **[DORA Regulation](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022R2554)** - Official EU regulation text
- **[Streamlit Documentation](https://docs.streamlit.io/)** - UI framework
- **[LangChain Documentation](https://python.langchain.com/)** - RAG framework
- **[Sentence Transformers](https://www.sbert.net/)** - Embedding models

---

**⚠️ Disclaimer**: This chatbot provides informational guidance based on DORA regulation documents. Always consult qualified legal professionals for definitive legal advice and compliance strategies.
