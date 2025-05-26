# Enhanced-multimodal-rag 🤖📄

After months of dedicated design and development, I'm thrilled to share a fully functional version of an advanced document analysis and chat application, Enhanced-Multimodal-RAG. This tool seamlessly integrates multiple OCR engines, including Mistral OCR for superior accuracy and Marker enhanced with the latest Gemini 2.5 Flash model, with a state-of-the-art Retrieval-Augmented Generation (RAG) pipeline for intelligent document interaction. The system leverages a hybrid query approach, combining semantic embeddings from Gemini with BM25 keyword search to deliver precise and contextually relevant results. A special thanks to https://github.com/VikParuchuri/marker/tree/master/marker, which served as the foundation for one of the parsing engines, fine-tuned to maximize performance with Gemini integration.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### 🔍 **Advanced Document Processing**
- **Dual OCR Engines**: Mistral OCR for accuracy + Marker with Gemini for enhanced extraction
- **Multi-format Support**: PDF, images (PNG, JPG, JPEG), and web URLs
- **Smart Content Extraction**: Maintains formatting, tables, and document structure

### 🧠 **Enhanced RAG Pipeline**
- **Hybrid Search**: Combines semantic embeddings (Gemini) with BM25 keyword search
- **Configurable Chunking**: Adjustable chunk size (200-10K chars) and overlap
- **Multiple Search Methods**: Semantic-only, keyword-only, or hybrid retrieval
- **Context Visualization**: View retrieved chunks with relevance scores

### ⚡ **Smart Chat Interface**
- **LangGraph Workflow**: Advanced query processing pipeline
- **Context-Aware Responses**: Answers based on most relevant document sections
- **Source Selection**: Choose between OCR engines or combine both
- **Interactive Context Viewer**: Inspect retrieved chunks in browser

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Mistral API Key ([Get one here](https://console.mistral.ai/))
- Google API Key for Gemini ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/winstonbartlegod/enhanced-multimodal-rag.git
cd enhanced-multimodal-rag
```
2. **Install requirements**
```bash
pip install -r requirements.txt
```
3. **Run the program**
```bash
streamlit run v1.py
```
## 🔧 Configuration

### Environment Variables
Create a `.env` file with your API keys:
```env
MISTRAL_API_KEY=your_mistral_api_key
GOOGLE_API_KEY=your_google_api_key
```

### Search Configuration
- **Chunk Size**: 200-10,000 characters (default: 1000)
- **Chunk Overlap**: 0-50% of chunk size (default: 200)
- **Search Method**: hybrid, semantic, or bm25 (default: hybrid)
- **Semantic Weight**: 0-100% for hybrid search (default: 70%)

## 🎯 Usage

### 1. Document Upload
- **PDF Upload**: Drag and drop or browse for PDF files
- **Image Upload**: Support for PNG, JPG, JPEG formats
- **URL Input**: Direct processing from web URLs

### 2. Processing Options
- **Mistral OCR**: Fast, accurate text extraction
- **Marker + Gemini**: Advanced extraction with AI enhancement
- **Combined Mode**: Use both engines for comprehensive analysis

### 3. Chat Interface
- Ask questions about your document
- View retrieved context with relevance scores
- Download context analysis as HTML
- Switch between content sources

### 4. Advanced Features
- **Context Viewer**: See exactly which document sections were used
- **Relevance Scoring**: Understand why certain chunks were selected
- **Search Method Comparison**: Try different retrieval approaches

## 🏗️ Architecture

```
Document Input → OCR Processing → Text Chunking → Embedding Generation
                                       ↓
Query Input → Query Processing → Hybrid Search → Context Assembly → LLM Response
                                       ↓
                               Relevance Scoring → Context Visualization
```

### Key Components
- **OCR Layer**: Mistral OCR + Marker with Gemini
- **Retrieval Engine**: Gemini embeddings + BM25 + hybrid scoring
- **Processing Pipeline**: LangGraph workflow management
- **UI Layer**: Streamlit with interactive components

## 📊 Performance

### Search Methods Comparison
| Method | Best For | Strengths | Use Case |
|--------|----------|-----------|----------|
| **Hybrid** | General use | Balanced precision/recall | Default recommendation |
| **Semantic** | Conceptual queries | Understanding context | "What are the main themes?" |
| **BM25** | Specific terms | Exact keyword matching | "Find mentions of 'quarterly revenue'" |

### Optimization Tips
- **Large Documents**: Increase chunk size to 2000+ characters
- **Technical Content**: Use BM25 for precise terminology
- **Conceptual Queries**: Use semantic search for better understanding
- **Best Results**: Use hybrid with 70/30 semantic/keyword split

## 🛠️ Development

### Project Structure
```
enhanced-multimodal-rag/
├── v1.py                 # Main Streamlit application
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── README.md            # This file
└── docs/                # Additional documentation
```

### Adding New Features
1. **New OCR Engine**: Extend the document processing pipeline
2. **Custom Embeddings**: Integrate additional embedding models
3. **Search Methods**: Add new retrieval algorithms
4. **Export Formats**: Support additional output formats

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🚨 Known Issues

- **Large PDFs**: Processing time increases with document size
- **API Limits**: Respect rate limits for Mistral and Google APIs
- **Memory Usage**: Large documents may require sufficient RAM

## 📝 License  

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mistral AI** for OCR capabilities
- **Google** for Gemini embeddings and generation
- **LangChain** for RAG framework
- **Streamlit** for the web interface
- **Marker** for advanced PDF processing

## 🔮 Roadmap

- [ ] **Batch Processing**: Handle multiple documents simultaneously
- [ ] **Cloud Deployment**: Docker containers and cloud deployment guides
- [ ] **Advanced Analytics**: Document analysis and insights dashboard
- [ ] **Vector Database**: Integration with Pinecone, Weaviate, or Chroma
- [ ] **Custom Models**: Support for local LLMs and embedding models


