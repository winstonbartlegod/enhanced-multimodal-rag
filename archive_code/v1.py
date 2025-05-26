import streamlit as st
import base64
import tempfile
import os
from mistralai import Mistral
from PIL import Image
import io
from mistralai import DocumentURLChunk, ImageURLChunk
from mistralai.models import OCRResponse
from dotenv import find_dotenv, load_dotenv
import google.generativeai as genai
import sys
import time
from typing import TypedDict, List, Dict, Any, Tuple
from langchain_core.messages import HumanMessage, AIMessage
import numpy as np
import html
import webbrowser
import tempfile as tf

# Set page config first
st.set_page_config(page_title="Winston's enhanced Document OCR & Chat Query Engine", layout="wide")

# Try to import enhanced search components
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from rank_bm25 import BM25Okapi
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # NLTK QUICKFIX - Download required NLTK data with updated resource names
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)  # Updated from 'punkt'
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    ENHANCED_SEARCH_AVAILABLE = True
except ImportError:
    ENHANCED_SEARCH_AVAILABLE = False

# Try to import LangGraph components
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# Try to import marker components
try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False

# Load environment variables
load_dotenv(find_dotenv())

# Global API keys
api_key = os.getenv('MISTRAL_API_KEY', '')
google_api_key = os.getenv('GOOGLE_API_KEY', '')

# Enhanced state for LangGraph with chunking and retrieval
class ChatbotState(TypedDict):
    query: str
    document_content: str
    source_info: str
    chunks: List[Dict[str, Any]]
    query_embedding: List[float]
    chunk_embeddings: List[List[float]]
    semantic_scores: List[float]
    bm25_scores: List[float]
    hybrid_scores: List[float]
    retrieved_chunks: List[Dict[str, Any]]
    context: str
    response: str
    messages: List[Dict[str, str]]
    search_method: str
    chunk_size: int
    chunk_overlap: int

def create_context_html_file(context_content: str, query: str) -> str:
    """Create a temporary HTML file and return its path"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Retrieved Context - Enhanced RAG</title>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 15px;
                margin-bottom: 25px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}
            .header h1 {{
                margin: 0 0 10px 0;
                font-size: 2.2em;
                font-weight: 300;
            }}
            .header p {{
                margin: 0;
                opacity: 0.9;
                font-size: 1.1em;
            }}
            .query-section {{
                background: white;
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 25px;
                border-left: 5px solid #667eea;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .query-section h3 {{
                margin-top: 0;
                color: #495057;
                font-size: 1.3em;
            }}
            .query-text {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                font-style: italic;
                font-size: 1.1em;
                border-left: 3px solid #28a745;
            }}
            .stats {{
                background: white;
                padding: 15px;
                border-radius: 12px;
                margin-bottom: 25px;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 15px;
            }}
            .stat-item {{
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }}
            .stat-number {{
                font-size: 1.5em;
                font-weight: bold;
                color: #667eea;
            }}
            .stat-label {{
                color: #6c757d;
                font-size: 0.9em;
                margin-top: 5px;
            }}
            .context-section {{
                background: white;
                padding: 25px;
                border-radius: 12px;
                border: 1px solid #e1e5e9;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .context-section h3 {{
                margin-top: 0;
                color: #495057;
                font-size: 1.4em;
                border-bottom: 2px solid #e9ecef;
                padding-bottom: 10px;
            }}
            .chunk {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 20px;
                margin: 15px 0;
                border-radius: 12px;
                border-left: 4px solid #28a745;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                transition: transform 0.2s ease;
            }}
            .chunk:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            }}
            .chunk-header {{
                font-weight: bold;
                color: #495057;
                margin-bottom: 15px;
                font-size: 1em;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .chunk-score {{
                background: #28a745;
                color: white;
                padding: 4px 8px;
                border-radius: 20px;
                font-size: 0.8em;
            }}
            .chunk-content {{
                white-space: pre-wrap;
                font-family: 'Georgia', serif;
                line-height: 1.8;
                color: #2c3e50;
                font-size: 1.05em;
            }}
            .no-context {{
                text-align: center;
                padding: 40px;
                color: #6c757d;
                font-style: italic;
            }}
            .timestamp {{
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 10px 15px;
                border-radius: 20px;
                font-size: 0.9em;
            }}
            @media (max-width: 768px) {{
                body {{
                    padding: 10px;
                }}
                .header {{
                    padding: 20px;
                }}
                .header h1 {{
                    font-size: 1.8em;
                }}
                .stats-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Retrieved Context</h1>
            <p>Enhanced RAG - Context sent to LLM for response generation</p>
        </div>
        
        <div class="query-section">
            <h3>📝 Original Query</h3>
            <div class="query-text">
                {html.escape(query)}
            </div>
        </div>
        
        <div class="stats">
            <h3>📊 Context Statistics</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-number">{len(context_content):,}</div>
                    <div class="stat-label">Characters</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{len(context_content.split())}</div>
                    <div class="stat-label">Words</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{len(context_content.split('---'))}</div>
                    <div class="stat-label">Chunks</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{len(context_content.split('. '))}</div>
                    <div class="stat-label">Sentences (approx)</div>
                </div>
            </div>
        </div>
        
        <div class="context-section">
            <h3>📄 Retrieved Content</h3>
            {format_context_for_html(context_content)}
        </div>
        
        <div class="timestamp">
            Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </body>
    </html>
    """
    
    # Create temporary file
    with tf.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_content)
        return f.name

def format_context_for_html(context: str) -> str:
    """Format context content for better HTML display"""
    if not context:
        return '<div class="no-context">No context available</div>'
    
    # Split by chunk separators
    chunks = context.split('\n\n---\n\n')
    if len(chunks) == 1:
        # Try alternative separators
        chunks = context.split('---')
    
    formatted_chunks = []
    
    for i, chunk in enumerate(chunks):
        if chunk.strip():
            # Extract chunk header and content
            lines = chunk.strip().split('\n', 1)
            if len(lines) >= 2 and '[Chunk' in lines[0]:
                header_line = lines[0]
                content = lines[1] if len(lines) > 1 else ""
                
                # Extract score from header if present
                score_match = ""
                if "Relevance:" in header_line:
                    try:
                        score_part = header_line.split("Relevance:")[-1].strip().rstrip(']')
                        score_match = f'<span class="chunk-score">Score: {score_part}</span>'
                        header = header_line.split(" - Relevance:")[0] + ']'
                    except:
                        header = header_line
                else:
                    header = header_line
            else:
                header = f"[Chunk {i+1}]"
                content = chunk.strip()
                score_match = ""
            
            # Escape HTML in content
            escaped_content = html.escape(content)
            escaped_header = html.escape(header)
            
            formatted_chunk = f"""
            <div class="chunk">
                <div class="chunk-header">
                    <span>{escaped_header}</span>
                    {score_match}
                </div>
                <div class="chunk-content">{escaped_content}</div>
            </div>
            """
            formatted_chunks.append(formatted_chunk)
    
    if not formatted_chunks:
        return '<div class="no-context">No properly formatted chunks found in context</div>'
    
    return '\n'.join(formatted_chunks)

class EnhancedDocumentProcessor:
    """Enhanced document processor with Gemini text-embedding-004 and BM25 (you can also use embedding-001 as fallback)"""
    
    def __init__(self, google_api_key=None, chunk_size=8000, chunk_overlap=500):
        self.google_api_key = google_api_key
        self.embeddings = None
        self.text_splitter = None
        self.stemmer = None
        self.stop_words = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if ENHANCED_SEARCH_AVAILABLE:
            # Initialize text splitter with configurable parameters
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            
            # Initialize Gemini embeddings with latest model (quietly)
            if google_api_key:
                try:
                    # Try the latest text-embedding-004 model first
                    self.embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/text-embedding-004",  # Latest Gemini embedding model
                        google_api_key=google_api_key
                    )
                except Exception as e:
                    try:
                        # Fallback to embedding-001
                        self.embeddings = GoogleGenerativeAIEmbeddings(
                            model="models/embedding-001",
                            google_api_key=google_api_key
                        )
                    except Exception as e2:
                        self.embeddings = None
            
            # Initialize NLP components for BM25
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
    
    def update_chunk_settings(self, chunk_size: int, chunk_overlap: int):
        """Update chunking parameters"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if ENHANCED_SEARCH_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
    
    def chunk_document(self, content: str) -> List[Dict[str, Any]]:
        """Split document into semantic chunks"""
        if not self.text_splitter:
            # Fallback: simple chunking
            chunks = []
            
            for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
                chunk_text = content[i:i + self.chunk_size]
                if chunk_text.strip():
                    chunks.append({
                        'text': chunk_text,
                        'chunk_id': i // (self.chunk_size - self.chunk_overlap),
                        'start_char': i,
                        'end_char': min(i + self.chunk_size, len(content)),
                        'word_count': len(chunk_text.split()),
                        'char_count': len(chunk_text)
                    })
            return chunks
        
        # Use LangChain text splitter for better chunking
        texts = self.text_splitter.split_text(content)
        chunks = []
        
        for i, text in enumerate(texts):
            # Find the position in original content
            start_pos = content.find(text) if text in content else i * (self.chunk_size - self.chunk_overlap)
            
            chunks.append({
                'text': text,
                'chunk_id': i,
                'start_char': start_pos,
                'end_char': start_pos + len(text),
                'word_count': len(text.split()),
                'char_count': len(text),
                'sentences': text.count('.') + text.count('!') + text.count('?')
            })
        
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get Gemini embeddings for texts"""
        if not self.embeddings:
            return []
        
        try:
            # Process texts in batches to avoid API limits
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        except Exception as e:
            st.error(f"Error getting embeddings: {e}")
            return []
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get Gemini embedding for query"""
        if not self.embeddings:
            return []
        
        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            st.error(f"Error getting query embedding: {e}")
            return []
    
    def preprocess_text_for_bm25(self, text: str) -> List[str]:
        """Preprocess text for BM25"""
        if not ENHANCED_SEARCH_AVAILABLE:
            return text.lower().split()
        
        # Tokenize and clean
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens, then stem
        processed_tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token.isalpha() and token not in self.stop_words and len(token) > 2
        ]
        
        return processed_tokens
    
    def setup_bm25(self, chunks: List[Dict[str, Any]]) -> BM25Okapi:
        """Setup BM25 index"""
        if not ENHANCED_SEARCH_AVAILABLE or not chunks:
            return None
        
        try:
            processed_chunks = [self.preprocess_text_for_bm25(chunk['text']) for chunk in chunks]
            return BM25Okapi(processed_chunks)
        except Exception as e:
            st.error(f"Error setting up BM25: {e}")
            return None
    
    def semantic_search(self, query_embedding: List[float], chunk_embeddings: List[List[float]], 
                       top_k: int = 5) -> Tuple[List[int], List[float]]:
        """Perform semantic search using cosine similarity"""
        if not query_embedding or not chunk_embeddings:
            return [], []
        
        try:
            # Calculate cosine similarities
            query_emb = np.array(query_embedding).reshape(1, -1)
            chunk_embs = np.array(chunk_embeddings)
            
            similarities = cosine_similarity(query_emb, chunk_embs)[0]
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_scores = similarities[top_indices]
            
            return top_indices.tolist(), top_scores.tolist()
        except Exception as e:
            st.error(f"Error in semantic search: {e}")
            return [], []
    
    def bm25_search(self, query: str, bm25_index: BM25Okapi, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """Perform BM25 search"""
        if not bm25_index:
            return [], []
        
        try:
            processed_query = self.preprocess_text_for_bm25(query)
            scores = bm25_index.get_scores(processed_query)
            
            # Get top k indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            top_scores = scores[top_indices]
            
            return top_indices.tolist(), top_scores.tolist()
        except Exception as e:
            st.error(f"Error in BM25 search: {e}")
            return [], []
    
    def hybrid_search(self, query: str, query_embedding: List[float], chunks: List[Dict[str, Any]], 
                     chunk_embeddings: List[List[float]], semantic_weight: float = 0.7, 
                     top_k: int = 5) -> Tuple[List[int], List[float], Dict[str, Any]]:
        """Combine semantic and BM25 search"""
        if not chunks:
            return [], [], {}
        
        # Get BM25 scores
        bm25_index = self.setup_bm25(chunks)
        bm25_indices, bm25_scores = self.bm25_search(query, bm25_index, len(chunks))
        
        # Get semantic scores
        sem_indices, sem_scores = self.semantic_search(query_embedding, chunk_embeddings, len(chunks))
        
        # Normalize scores
        sem_scores_norm = np.array(sem_scores) if sem_scores else np.zeros(len(chunks))
        bm25_scores_norm = np.array(bm25_scores) if bm25_scores else np.zeros(len(chunks))
        
        # Normalize to 0-1 range
        if sem_scores_norm.max() > 0:
            sem_scores_norm = sem_scores_norm / sem_scores_norm.max()
        if bm25_scores_norm.max() > 0:
            bm25_scores_norm = bm25_scores_norm / bm25_scores_norm.max()
        
        # Combine scores
        hybrid_scores = np.zeros(len(chunks))
        
        # Add semantic scores
        for idx, score in zip(sem_indices, sem_scores_norm):
            if idx < len(hybrid_scores):
                hybrid_scores[idx] += semantic_weight * score
        
        # Add BM25 scores
        bm25_weight = 1.0 - semantic_weight
        for idx, score in zip(bm25_indices, bm25_scores_norm):
            if idx < len(hybrid_scores):
                hybrid_scores[idx] += bm25_weight * score
        
        # Get top k results
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        top_scores = hybrid_scores[top_indices]
        
        search_details = {
            'semantic_indices': sem_indices[:top_k],
            'semantic_scores': sem_scores[:top_k],
            'bm25_indices': bm25_indices[:top_k],
            'bm25_scores': bm25_scores[:top_k],
            'semantic_weight': semantic_weight,
            'bm25_weight': bm25_weight
        }
        
        return top_indices.tolist(), top_scores.tolist(), search_details

# Initialize the enhanced processor
enhanced_processor = None

def get_enhanced_processor():
    """Get or create the enhanced processor"""
    global enhanced_processor
    chunk_size = st.session_state.get('chunk_size', 8000)
    chunk_overlap = st.session_state.get('chunk_overlap', 500)
    
    if enhanced_processor is None:
        enhanced_processor = EnhancedDocumentProcessor(google_api_key, chunk_size, chunk_overlap)
    else:
        # Update chunk settings if they changed
        enhanced_processor.update_chunk_settings(chunk_size, chunk_overlap)
    return enhanced_processor

# Enhanced LangGraph nodes
def process_query(state: ChatbotState) -> ChatbotState:
    """Process and analyze the user query"""
    query = state["query"]
    processed_query = query.strip()
    
    # Get query embedding if available
    processor = get_enhanced_processor()
    if processor and processor.embeddings:
        query_embedding = processor.get_query_embedding(processed_query)
        state["query_embedding"] = query_embedding
    
    state["query"] = processed_query
    return state

def chunk_and_embed_document(state: ChatbotState) -> ChatbotState:
    """Chunk document and create embeddings"""
    document_content = state["document_content"]
    
    if not document_content:
        state["chunks"] = []
        state["chunk_embeddings"] = []
        return state
    
    processor = get_enhanced_processor()
    
    # Chunk the document
    chunks = processor.chunk_document(document_content)
    state["chunks"] = chunks
    
    # Get embeddings for chunks if available
    if processor and processor.embeddings and chunks:
        with st.spinner("Creating embeddings for document chunks..."):
            chunk_texts = [chunk['text'] for chunk in chunks]
            chunk_embeddings = processor.get_embeddings(chunk_texts)
            state["chunk_embeddings"] = chunk_embeddings
    
    return state

def retrieve_relevant_chunks(state: ChatbotState) -> ChatbotState:
    """Retrieve most relevant chunks using hybrid search"""
    chunks = state["chunks"]
    query = state["query"]
    query_embedding = state.get("query_embedding", [])
    chunk_embeddings = state.get("chunk_embeddings", [])
    search_method = state.get("search_method", "hybrid")
    
    if not chunks:
        state["context"] = state["document_content"]
        state["retrieved_chunks"] = []
        return state
    
    processor = get_enhanced_processor()
    top_k = min(5, len(chunks))
    
    if search_method == "semantic" and query_embedding and chunk_embeddings:
        # Semantic search only
        top_indices, top_scores = processor.semantic_search(query_embedding, chunk_embeddings, top_k)
        search_info = "semantic search"
        
    elif search_method == "bm25":
        # BM25 search only
        bm25_index = processor.setup_bm25(chunks)
        top_indices, top_scores = processor.bm25_search(query, bm25_index, top_k)
        search_info = "BM25 keyword search"
        
    elif search_method == "hybrid" and query_embedding and chunk_embeddings:
        # Hybrid search
        top_indices, top_scores, search_details = processor.hybrid_search(
            query, query_embedding, chunks, chunk_embeddings, semantic_weight=0.7, top_k=top_k
        )
        search_info = "hybrid search (70% semantic + 30% BM25)"
        
    else:
        # Fallback: simple keyword matching
        query_lower = query.lower()
        scores = []
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk['text'].lower()
            score = sum(1 for word in query_lower.split() if word in chunk_lower)
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scores[:top_k]]
        top_scores = [score for _, score in scores[:top_k]]
        search_info = "simple keyword matching"
    
    # Get retrieved chunks
    retrieved_chunks = []
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        if idx < len(chunks):
            chunk_info = chunks[idx].copy()
            chunk_info['relevance_score'] = float(score)
            chunk_info['rank'] = i + 1
            retrieved_chunks.append(chunk_info)
    
    # Create context from retrieved chunks
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Chunk {chunk['rank']} - Relevance: {chunk['relevance_score']:.3f}]\n{chunk['text']}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    state["retrieved_chunks"] = retrieved_chunks
    state["context"] = context
    state["search_info"] = search_info
    
    return state

def generate_response_node(state: ChatbotState) -> ChatbotState:
    """Generate response using retrieved context"""
    context = state["context"]
    query = state["query"]
    retrieved_chunks = state.get("retrieved_chunks", [])
    search_info = state.get("search_info", "unknown method")
    
    try:
        genai.configure(api_key=google_api_key)
        
        if not context or len(context) < 10:
            state["response"] = "Error: No relevant content found to answer your question."
            return state
        
        # Enhanced prompt with retrieval information
        chunk_info = f"Retrieved {len(retrieved_chunks)} most relevant chunks using {search_info}"
        if retrieved_chunks:
            scores = [f"{chunk.get('relevance_score', 0):.3f}" for chunk in retrieved_chunks]
            chunk_info += f"\nRelevance scores: {scores}"
        
        prompt = f"""I have analyzed a document and retrieved the most relevant sections to answer your question.

Retrieval Information:
{chunk_info}

Retrieved Content:
{context}

Question: {query}

Instructions:
- Base your answer primarily on the retrieved content above
- If the retrieved sections fully answer the question, provide a comprehensive response
- If information is partial, clearly state what is available and what might be missing
- Reference specific chunks when relevant (e.g., "According to Chunk 1...")
- Synthesize information from multiple chunks when they relate to the same topic
- If the retrieved content doesn't address the question, state this clearly
"""
        
        model = genai.GenerativeModel('gemma-3-27b-it')
        
        generation_config = {
            "temperature": 0.4,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
        ]
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        state["response"] = response.text
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        import traceback
        print(traceback.format_exc())
        state["response"] = f"Error generating response: {str(e)}"
    
    return state

def create_enhanced_chatbot_graph():
    """Create enhanced chatbot workflow graph"""
    if not LANGGRAPH_AVAILABLE:
        return None
        
    graph = StateGraph(ChatbotState)
    
    # Add nodes
    graph.add_node("process_query", process_query)
    graph.add_node("chunk_and_embed", chunk_and_embed_document)
    graph.add_node("retrieve_chunks", retrieve_relevant_chunks)
    graph.add_node("generate_response", generate_response_node)
    
    # Define workflow
    graph.add_edge("process_query", "chunk_and_embed")
    graph.add_edge("chunk_and_embed", "retrieve_chunks")
    graph.add_edge("retrieve_chunks", "generate_response")
    graph.add_edge("generate_response", END)
    
    graph.set_entry_point("process_query")
    
    return graph.compile()

def initialize_mistral_client(api_key):
    """Initialize Mistral client with error handling"""
    if not api_key:
        return None
    
    try:
        client = Mistral(api_key=api_key)
        client.models.list()
        return client
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Mistral client: {str(e)}")
        return None

def test_google_api(api_key):
    """Test Google API key and return status"""
    if not api_key:
        return False, "No API key provided"
    
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        return True, "Connected successfully"
    except Exception as e:
        return False, f"Failed to connect: {str(e)}"

def upload_pdf(client, content, filename):
    """Uploads a PDF to Mistral's API and retrieves a signed URL for processing."""
    if client is None:
        raise ValueError("Mistral client is not initialized")
        
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, filename)
        
        with open(temp_path, "wb") as tmp:
            tmp.write(content)
        
        try:
            with open(temp_path, "rb") as file_obj:
                file_upload = client.files.upload(
                    file={"file_name": filename, "content": file_obj},
                    purpose="ocr"
                )
            
            signed_url = client.files.get_signed_url(file_id=file_upload.id)
            return signed_url.url
        except Exception as e:
            raise ValueError(f"Error uploading PDF: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """Replace image placeholders with base64 encoded images in markdown."""
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})")
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """Combine markdown from all pages with their respective images."""
    markdowns: list[str] = []
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))

    return "\n\n".join(markdowns)

def display_pdf(file):
    """Displays a PDF in Streamlit using an iframe."""
    try:
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

def process_ocr(client, document_source):
    """Process document with OCR API based on source type"""
    if client is None:
        raise ValueError("Mistral client is not initialized")
        
    if document_source["type"] == "document_url":
        return client.ocr.process(
            document=DocumentURLChunk(document_url=document_source["document_url"]),
            model="mistral-ocr-latest",
            include_image_base64=True
        )
    elif document_source["type"] == "image_url":
        return client.ocr.process(
            document=ImageURLChunk(image_url=document_source["image_url"]),
            model="mistral-ocr-latest",
            include_image_base64=True
        )
    else:
        raise ValueError(f"Unsupported document source type: {document_source['type']}")

def process_pdf_with_marker(pdf_path, google_api_key, gemini_model_selection):
    """Process PDF with Marker library"""
    if not MARKER_AVAILABLE:
        return "Marker library is not available. Please install it to use this feature."
    
    try:
        if gemini_model_selection == "Gemini 2.5 Flash":
            llm_service = "marker.services.gemini.GoogleGeminiService2"
        else:
            llm_service = "marker.services.gemini.GoogleGeminiService"
        
        config = {
            "output_format": "markdown",
            "use_llm": True,
            "gemini_api_key": google_api_key,
            "workers": 8,
            "disable_multiprocessing": False,
            "thinking_budget": 0,
            "llm_service": llm_service
        }
        config_parser = ConfigParser(config)

        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )
        rendered = converter(pdf_path)

        return rendered.markdown
    except Exception as e:
        import traceback
        print(f"Error in marker processing: {str(e)}")
        print(traceback.format_exc())
        return f"Error processing with Marker: {str(e)}"

def generate_response_simple(context, query):
    """Simple response generation without LangGraph"""
    try:
        genai.configure(api_key=google_api_key)
        
        if not context or len(context) < 10:
            return "Error: No document content available to answer your question."
            
        prompt = f"""I have a document with the following content:

{context}

Based on this document, please answer the following question:
{query}

If you can find information related to the query in the document, please answer based on that information.
If the document doesn't specifically mention the exact information asked, please try to infer from related content or clearly state that the specific information isn't available in the document.
"""
        
        model = genai.GenerativeModel('gemma-3-27b-it')
        
        generation_config = {
            "temperature": 0.4,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
        ]
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        return response.text
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return f"Error generating response: {str(e)}"

def generate_response_with_enhanced_graph(document_content, query, source_info, search_method="hybrid"):
    """Generate response using enhanced LangGraph workflow"""
    if not LANGGRAPH_AVAILABLE:
        return generate_response_simple(document_content, query), None
    
    try:
        chatbot_graph = create_enhanced_chatbot_graph()
        
        initial_state = {
            "query": query,
            "document_content": document_content,
            "source_info": source_info,
            "chunks": [],
            "query_embedding": [],
            "chunk_embeddings": [],
            "semantic_scores": [],
            "bm25_scores": [],
            "hybrid_scores": [],
            "retrieved_chunks": [],
            "context": "",
            "response": "",
            "messages": [],
            "search_method": search_method,
            "chunk_size": st.session_state.get('chunk_size', 8000),
            "chunk_overlap": st.session_state.get('chunk_overlap', 500)
        }
        
        result = chatbot_graph.invoke(initial_state)
        
        # Return both response and context for viewing
        return result["response"], result.get("context", "")
        
    except Exception as e:
        print(f"Error in enhanced LangGraph processing: {str(e)}")
        return generate_response_simple(document_content, query), None

def save_pdf_file(content, filename):
    """Save uploaded PDF to a file and return its path."""
    upload_dir = os.path.join(tempfile.gettempdir(), "streamlit_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, filename)
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    return file_path

# Enhanced Streamlit UI
def main():
    global api_key, google_api_key
    
    # Show warnings for missing libraries
    if not ENHANCED_SEARCH_AVAILABLE:
        st.warning("⚠️ Enhanced search libraries not found. Install langchain-google-genai, rank-bm25, nltk, and scikit-learn for advanced features.")
    
    if not LANGGRAPH_AVAILABLE:
        st.warning("⚠️ LangGraph not found. Using simple query processing.")
    
    if not MARKER_AVAILABLE:
        st.warning("⚠️ Marker library not found. Only Mistral OCR will be available.")
    
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "mistral_content" not in st.session_state:
        st.session_state.mistral_content = ""
        
    if "marker_content" not in st.session_state:
        st.session_state.marker_content = ""
    
    if "document_loaded" not in st.session_state:
        st.session_state.document_loaded = False
        
    if "pdf_path" not in st.session_state:
        st.session_state.pdf_path = None
        
    if "content_source" not in st.session_state:
        st.session_state.content_source = "mistral"
    
    if "use_langgraph" not in st.session_state:
        st.session_state.use_langgraph = LANGGRAPH_AVAILABLE
    
    if "search_method" not in st.session_state:
        st.session_state.search_method = "hybrid"
    
    # Add chunk settings to session state
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 8000
    
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 500
    
    if "last_context" not in st.session_state:
        st.session_state.last_context = ""
    
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # API key inputs
        api_key_tab1, api_key_tab2 = st.tabs(["Mistral API", "Google API"])
        
        with api_key_tab1:
            user_api_key = st.text_input("Mistral API Key", value=api_key if api_key else "", type="password")
            if user_api_key:
                api_key = user_api_key
                os.environ["MISTRAL_API_KEY"] = api_key
        
        with api_key_tab2:
            user_google_api_key = st.text_input(
                "Google API Key", 
                value=google_api_key if google_api_key else "", 
                type="password",
                help="API key for Google Gemini (embeddings and generation)"
            )
            if user_google_api_key:
                google_api_key = user_google_api_key
                os.environ["GOOGLE_API_KEY"] = google_api_key
                # Reinitialize the processor with new API key
                global enhanced_processor
                enhanced_processor = None
        
        # Initialize Mistral client
        mistral_client = None
        if api_key:
            mistral_client = initialize_mistral_client(api_key)
            if mistral_client:
                st.sidebar.success("✅ Mistral API connected successfully")
        
        # Google API validation
        if google_api_key:
            is_valid, message = test_google_api(google_api_key)
            if is_valid:
                st.sidebar.success(f"✅ Google API {message}")
            else:
                st.sidebar.error(f"❌ Google API: {message}")
                google_api_key = None
        
        # Display warnings
        if not api_key or mistral_client is None:
            st.sidebar.warning("⚠️ Valid Mistral API key required for document processing")
        
        if not google_api_key:
            st.sidebar.warning("⚠️ Google API key required for enhanced search and chat")
        
        # Enhanced search settings
        if LANGGRAPH_AVAILABLE:
            st.subheader("🔍 Enhanced Search")
            use_langgraph = st.checkbox(
                "Use Enhanced RAG Pipeline",
                value=st.session_state.use_langgraph,
                help="Enable advanced retrieval with Gemini embeddings and BM25"
            )
            st.session_state.use_langgraph = use_langgraph
            
            if use_langgraph:
                if ENHANCED_SEARCH_AVAILABLE and google_api_key:
                    st.success("🚀 Enhanced RAG enabled")
                    
                    # Chunk size and overlap controls
                    st.subheader("⚙️ Chunking Settings")
                    
                    chunk_size = st.slider(
                        "Chunk Size (characters):",
                        min_value=200,
                        max_value=10000,
                        value=st.session_state.chunk_size,
                        step=100,
                        help="Size of each text chunk for processing"
                    )
                    st.session_state.chunk_size = chunk_size
                    
                    chunk_overlap = st.slider(
                        "Chunk Overlap (characters):",
                        min_value=0,
                        max_value=min(2000, chunk_size // 2),
                        value=min(st.session_state.chunk_overlap, chunk_size // 2),
                        step=50,
                        help="Overlap between consecutive chunks"
                    )
                    st.session_state.chunk_overlap = chunk_overlap
                    
                    search_method = st.selectbox(
                        "Search Method:",
                        ["hybrid", "semantic", "bm25"],
                        index=["hybrid", "semantic", "bm25"].index(st.session_state.search_method),
                        help="Choose retrieval method"
                    )
                    st.session_state.search_method = search_method
                    
                    # Method descriptions
                    if search_method == "hybrid":
                        st.caption("🔄 Combines semantic embeddings (70%) + BM25 (30%)")
                    elif search_method == "semantic":
                        st.caption("🧠 Uses Gemini embeddings for semantic similarity")
                    elif search_method == "bm25":
                        st.caption("🔍 Uses BM25 for keyword-based search")
                else:
                    st.warning("⚠️ Install enhanced search libraries and add Google API key")
            else:
                st.info("⚡ Using simple processing")
        
        # Marker settings
        gemini_model_selection = None
        if MARKER_AVAILABLE and google_api_key:
            st.subheader("Marker Settings")
            gemini_model_selection = st.selectbox(
                "Select Gemini Model for Marker:",
                ["Gemini 2.5 Flash", "Gemini 2.0 Flash"],
                index=0,
                help="Choose which Gemini model to use for Marker processing"
            )
        
        # Document upload section
        st.subheader("Document Upload")
        
        if mistral_client and (google_api_key or not MARKER_AVAILABLE):
            input_method = st.radio("Select Input Type:", ["PDF Upload", "Image Upload", "URL"])
            
            document_source = None
            
            if input_method == "URL":
                url = st.text_input("Document URL:")
                if url and st.button("Load Document from URL"):
                    document_source = {
                        "type": "document_url",
                        "document_url": url
                    }
                    st.session_state.marker_content = "URL processing is only available with Mistral OCR."
            
            elif input_method == "PDF Upload":
                uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"])
                if uploaded_file and st.button("Process PDF"):
                    content = uploaded_file.read()
                    
                    try:
                        pdf_path = save_pdf_file(content, uploaded_file.name)
                        st.session_state.pdf_path = pdf_path
                        
                        with st.spinner("Processing document with Mistral OCR..."):
                            document_source = {
                                "type": "document_url",
                                "document_url": upload_pdf(mistral_client, content, uploaded_file.name)
                            }
                        
                        if MARKER_AVAILABLE and google_api_key and gemini_model_selection:
                            with st.spinner(f"Processing document with Marker using {gemini_model_selection}..."):
                                marker_result = process_pdf_with_marker(pdf_path, google_api_key, gemini_model_selection)
                                if marker_result:
                                    st.session_state.marker_content = marker_result
                                    st.success(f"✅ PDF processed with Marker using {gemini_model_selection} successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
            
            elif input_method == "Image Upload":
                uploaded_image = st.file_uploader("Choose Image file", type=["png", "jpg", "jpeg"])
                if uploaded_image and st.button("Process Image"):
                    try:
                        image = Image.open(uploaded_image)
                        st.image(image, caption="Uploaded Image", use_column_width=True)
                        
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        document_source = {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{img_str}"
                        }
                        st.session_state.marker_content = "Image processing is only available with Mistral OCR."
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
            
            # Process with Mistral OCR
            if document_source:
                with st.spinner("Processing document with Mistral OCR..."):
                    try:
                        ocr_response = process_ocr(mistral_client, document_source)
                        
                        if ocr_response and ocr_response.pages:
                            raw_content = []
                            
                            for page in ocr_response.pages:
                                page_content = page.markdown.strip()
                                if page_content:
                                    raw_content.append(page_content)
                            
                            final_content = "\n\n".join(raw_content)
                            
                            display_content = []
                            for i, page in enumerate(ocr_response.pages):
                                page_content = page.markdown.strip()
                                if page_content:
                                    display_content.append(f"Page {i+1}:\n{page_content}")
                            
                            display_formatted = "\n\n----------\n\n".join(display_content)
                            
                            st.session_state.mistral_content = final_content
                            st.session_state.mistral_display_content = display_formatted
                            st.session_state.document_loaded = True
                            
                            st.success(f"✅ Document processed with Mistral OCR successfully! Extracted {len(final_content)} characters from {len(raw_content)} pages.")
                        else:
                            st.warning("No content extracted from document with Mistral OCR.")
                    
                    except Exception as e:
                        st.error(f"Mistral OCR processing error: {str(e)}")
    
    # Main area
    st.title("Winston's OCR + Query Engine with Smart Retrieval")
    
    if st.session_state.document_loaded or st.session_state.marker_content:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
                st.header("Uploaded PDF")
                display_pdf(st.session_state.pdf_path)
        
        with col2:
            content_source = st.radio(
                "Select content source for viewing:", 
                ["Mistral OCR", "Marker"],
                index=0 if st.session_state.content_source == "mistral" else 1,
                horizontal=True
            )
            
            st.session_state.content_source = "mistral" if content_source == "Mistral OCR" else "marker"
            
            with st.expander("Document Content", expanded=False):
                if content_source == "Mistral OCR":
                    if st.session_state.mistral_content:
                        st.markdown(st.session_state.mistral_display_content if "mistral_display_content" in st.session_state else st.session_state.mistral_content)
                    else:
                        st.warning("No content available from Mistral OCR.")
                else:
                    if st.session_state.marker_content:
                        st.markdown(st.session_state.marker_content)
                    else:
                        st.warning("No content available from Marker processing.")
        
        # Enhanced chat interface
        # st.subheader("Chat with your document")
        
        # # WORKING SOLUTION: File-based popup with downloadable link and browser opening
        # if (st.session_state.use_langgraph and 
        #     st.session_state.last_context and 
        #     st.session_state.messages and 
        #     st.session_state.messages[-1]["role"] == "assistant"):
            
        #     col_btn, col_download = st.columns([2, 1])
            
        #     with col_btn:
        #         if st.button("🔍 View Retrieved Context", key="context_btn", type="secondary"):
        #             try:
        #                 # Create HTML file
        #                 html_file_path = create_context_html_file(st.session_state.last_context, st.session_state.last_query)
                        
        #                 # Try to open in browser
        #                 try:
        #                     webbrowser.open(f'file://{html_file_path}')
        #                     st.success("✅ Context opened in your browser!")
        #                 except Exception as e:
        #                     st.warning(f"Could not auto-open browser: {e}")
                        
        #                 # Provide download link as backup
        #                 with open(html_file_path, 'r', encoding='utf-8') as f:
        #                     html_content = f.read()
                        
        #                 st.download_button(
        #                     label="📥 Download Context HTML",
        #                     data=html_content,
        #                     file_name=f"retrieved_context_{int(time.time())}.html",
        #                     mime="text/html",
        #                     key="download_context"
        #                 )
                        
        #                 st.info("💡 If the browser didn't open automatically, use the download button above and open the file manually.")
                        
        #             except Exception as e:
        #                 st.error(f"Error creating context view: {e}")
            
        #     with col_download:
        #         # Quick preview button
        #         if st.button("👁️ Quick Preview", key="preview_btn"):
        #             with st.expander("Context Preview", expanded=True):
        #                 st.text_area(
        #                     "Retrieved Context:",
        #                     value=st.session_state.last_context,
        #                     height=300,
        #                     disabled=True
        #                 )

        # Enhanced chat interface
        st.subheader("Chat with your document")

        # Context viewing buttons - only show if we have recent context
        if (st.session_state.use_langgraph and 
            st.session_state.last_context and 
            st.session_state.messages and 
            st.session_state.messages[-1]["role"] == "assistant"):
            
            col_btn, col_download = st.columns([2, 1])
            
            with col_btn:
                # Add a unique key based on the number of messages to ensure button resets
                button_key = f"context_btn_{len(st.session_state.messages)}"
                if st.button("🔍 View Retrieved Context", key=button_key, type="secondary"):
                    try:
                        # Use the current context and query
                        current_query = st.session_state.last_query
                        current_context = st.session_state.last_context
                        
                        # Create HTML file with current context
                        html_file_path = create_context_html_file(current_context, current_query)
                        
                        # Try to open in browser
                        try:
                            webbrowser.open(f'file://{html_file_path}')
                            st.success("✅ Context opened in your browser!")
                        except Exception as e:
                            st.warning(f"Could not auto-open browser: {e}")
                        
                        # Provide download link as backup
                        with open(html_file_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        # Unique key for download button too
                        download_key = f"download_context_{len(st.session_state.messages)}"
                        st.download_button(
                            label="📥 Download Context HTML",
                            data=html_content,
                            file_name=f"retrieved_context_{int(time.time())}.html",
                            mime="text/html",
                            key=download_key
                        )
                        
                        st.info("💡 If the browser didn't open automatically, use the download button above and open the file manually.")
                        
                    except Exception as e:
                        st.error(f"Error creating context view: {e}")
            
            with col_download:
                # Quick preview button with unique key
                preview_key = f"preview_btn_{len(st.session_state.messages)}"
                if st.button("👁️ Quick Preview", key=preview_key):
                    with st.expander("Context Preview", expanded=True):
                        st.text_area(
                            "Retrieved Context:",
                            value=st.session_state.last_context,
                            height=300,
                            disabled=True,
                            key=f"context_preview_{len(st.session_state.messages)}"
                        )
        
        query_source = st.select_slider(
            "Select content source for querying:",
            options=["Mistral OCR", "Balanced", "Marker"],
            value="Balanced"
        )
        
        # Display chat messages
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            if not google_api_key:
                st.error("Google API key is required for generating responses. Please add it in the sidebar settings.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.last_query = prompt  # Store the query
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Determine content source
                document_content = ""
                source_info = ""
                
                if query_source == "Mistral OCR":
                    document_content = st.session_state.mistral_content
                    source_info = "Mistral OCR"
                elif query_source == "Marker":
                    document_content = st.session_state.marker_content
                    source_info = "Marker"
                else:
                    if st.session_state.mistral_content and st.session_state.marker_content:
                        document_content = f"--- MISTRAL OCR CONTENT ---\n\n{st.session_state.mistral_content}\n\n--- MARKER CONTENT ---\n\n{st.session_state.marker_content}"
                        source_info = "combined Mistral OCR and Marker"
                    elif st.session_state.mistral_content:
                        document_content = st.session_state.mistral_content
                        source_info = "Mistral OCR only (Marker content not available)"
                    elif st.session_state.marker_content:
                        document_content = st.session_state.marker_content
                        source_info = "Marker only (Mistral OCR content not available)"
                
                # Generate response
                search_info = ""
                if st.session_state.use_langgraph and ENHANCED_SEARCH_AVAILABLE:
                    search_info = f" with {st.session_state.search_method} retrieval"
                
                processing_method = f"Enhanced RAG{search_info}" if st.session_state.use_langgraph else "simple processing"
                
                with st.chat_message("assistant"):
                    with st.spinner(f"Analyzing using {source_info} content with {processing_method}..."):
                        if st.session_state.use_langgraph:
                            response, context = generate_response_with_enhanced_graph(
                                document_content, prompt, source_info, st.session_state.search_method
                            )
                            # Store context for viewing
                            st.session_state.last_context = context or ""
                        else:
                            response = generate_response_simple(document_content, prompt)
                            st.session_state.last_context = ""
                        
                        st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Force rerun to show context button (trying to fix this. But later ba)
                #if st.session_state.use_langgraph and st.session_state.last_context:
                #    st.rerun()
    else:
        st.info("👈 Please upload a document using the sidebar to start chatting.")
        

if __name__ == "__main__":
    main()