# DocBreeze 🦙

DocBreeze is a streamlined document analysis application that leverages LLAMA3.2 and advanced RAG (Retrieval-Augmented Generation) technologies to provide intelligent document summarization and question-answering capabilities.

## Features

- **PDF Document Processing**: Upload and process PDF documents with high-quality preview functionality
- **Interactive Document Viewer**: Built-in PDF viewer with zoom controls (50% to 200%)
- **Intelligent Question Answering**: Ask questions about your documents and receive contextually relevant answers
- **RAG Implementation**: Utilizes FAISS vector store and LLAMA3.2 for accurate information retrieval
- **History Tracking**: Maintains a session history of all questions and answers
- **User-Friendly Interface**: Clean, intuitive interface built with Streamlit

## Technical Architecture

The application is built using the following key technologies:

- **Frontend Framework**: Streamlit
- **Document Processing**: PyMuPDF
- **Vector Storage**: FAISS with Nomic embeddings
- **Language Model**: LLAMA3.2
- **Text Processing**: LangChain

## Prerequisites

- Python 3.x
- Ollama server running locally
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd docbreeze
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure Ollama server is running locally with LLAMA3.2 model installed:
```bash
ollama run llama3.2
```

## Usage

1. Start the application:
```bash
streamlit run docbreeze.py
```

2. Access the application through your web browser at `http://localhost:8501`

3. Upload a PDF document using the file uploader

4. Ask questions about your document in the right panel

## Features in Detail

### Document Processing
- Automatic document chunking for optimal processing
- Vector embeddings generation using Nomic's embedding model
- Efficient document storage using FAISS vector database

### Question Answering
- MMR (Maximal Marginal Relevance) search for diverse, relevant answers
- Context-aware responses using RAG architecture
- Concise, user-friendly answer generation

### User Interface
- Split-panel design for simultaneous document viewing and Q&A
- Zoom controls for document preview
- Expandable question history
- One-click document and history clearing

## System Requirements

- Minimum 8GB RAM recommended
- Local storage space for document processing
- Internet connection for model downloads
- Modern web browser

## License

This project is licensed under the MIT License:

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
