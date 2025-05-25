# Enhanced-multimodal-rag 🤖📄

After months of dedicated design and development, I'm thrilled to share a fully functional version of an advanced document analysis and chat application, Enhanced-Multimodal-RAG. This tool seamlessly integrates multiple OCR engines, including Mistral OCR for superior accuracy and Marker enhanced with the latest Gemini 2.5 Flash model, with a state-of-the-art Retrieval-Augmented Generation (RAG) pipeline for intelligent document interaction. The system leverages a hybrid query approach, combining semantic embeddings from Gemini with BM25 keyword search to deliver precise and contextually relevant results. A special thanks to https://github.com/VikParuchuri/marker/tree/master/marker, which served as the foundation for one of the parsing engines, fine-tuned to maximize performance with Gemini integration.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
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
