# Enhanced-multimodal-rag 🤖📄

An advanced document analysis and chat application that combines multiple OCR engines with state-of-the-art Retrieval-Augmented Generation (RAG) for intelligent document interaction.

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
git clone https://github.com/yourusername/smart-doc-chat.git
cd smart-doc-chat
