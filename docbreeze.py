import streamlit as st
import os
import tempfile
import warnings
import shutil
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Must call set_page_config as the first Streamlit command
st.set_page_config(
    page_title="DOCBREEZE",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for document container
st.markdown("""
    <style>
    .sidebar img {
        opacity: 0.7;
    }
    .doc-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: white;
        padding: 10px;
        margin: 10px 0;
    }
    .stMarkdown {
        max-height: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def create_vector_store(file_upload):
    """
    Create a vector database from an uploaded PDF file using FAISS.
    
    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.
    Returns:
        FAISS: A vector store containing the processed document chunks.
    """
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, file_upload.name)
        
        # Save uploaded file temporarily
        with open(path, "wb") as f:
            f.write(file_upload.getvalue())
            
        # Load and process the document
        loader = PyMuPDFLoader(path)
        data = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        
        # Create embeddings and vector store
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
        vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        return vector_store
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        return None

def get_pdf_pages(file_upload):
    """Extract pages from PDF as images"""
    pdf_pages = []
    pdf_document = fitz.open(stream=file_upload.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img_bytes = pix.tobytes("png")
        pdf_pages.append(img_bytes)
    pdf_document.close()
    return pdf_pages

def setup_rag_chain(vector_store):
    """Set up the RAG chain for document querying"""
    # Initialize retriever
    retriever = vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'fetch_k': 100, 'lambda_mult': 1}
    )
    
    # Set up prompt template
    prompt = ChatPromptTemplate.from_template("""
You are a friendly and concise assistant designed to summarize and answer the User's questions based on uploaded PDFs.
Use the content of the PDFs to provide clear, accurate, and relevant answers to the User's questions.
Address the User directly in your responses, ensuring they feel acknowledged. 
When summarizing, focus on the key points and main ideas, and present them in a concise, friendly tone. 
If the information needed to answer a question is not in the PDFs, clearly state: "I'm sorry, but I couldn't find the answer in the provided documents." 
Limit answers to ten sentences or fewer, and always acknowledge the User's question explicitly.
    Question: {question}
    Context: {context}
    Answer:
    """)
    
    # Initialize model
    model = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
    
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Create and return chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "summary_history" not in st.session_state:
    st.session_state.summary_history = []
if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = None
if "zoom_level" not in st.session_state:
    st.session_state.zoom_level = 100

def main():
    title_col1, title_col2 = st.columns([0.15, 0.85])
    with title_col1:
        st.image("assets/logo.png", width=250)
    with title_col2:
        st.title("DOCBREEZE")
        st.write("A LLAMA3.2-powered document summarizer")
    
    # Create two columns for layout
    col1, col2 = st.columns([1.5, 2])
    
    # Document upload and display section (left column)
    with col1:
        st.header("Upload Document")
        file_upload = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to analyze"
        )
        
        if file_upload:
            if st.session_state.pdf_pages is None or st.session_state.vector_store is None:
                with st.spinner("Processing document..."):
                    # Create vector store
                    file_upload.seek(0)  # Reset file pointer
                    st.session_state.vector_store = create_vector_store(file_upload)
                    
                    # Extract PDF pages for display
                    file_upload.seek(0)  # Reset file pointer again
                    st.session_state.pdf_pages = get_pdf_pages(file_upload)
                    
                if st.session_state.vector_store is not None:
                    st.success("Document processed successfully!")
                else:
                    st.error("Failed to process document.")
        
        # Document viewer with zoom control
        if st.session_state.pdf_pages:
            st.header("Document Preview")
            
            # Zoom control
            st.session_state.zoom_level = st.slider(
                "Zoom Level (%)", 
                min_value=50, 
                max_value=200, 
                value=100 if st.session_state.zoom_level == 50 else st.session_state.zoom_level,
                step=10,
                key="zoom_slider"
            )
            
            # Create scrollable container for PDF pages
            with st.container():
                    pdf_container = st.container(height=400, border=True)
                    with pdf_container:
                        for page_bytes in st.session_state.pdf_pages:
                            st.image(
                                page_bytes,
                                width=(700 * st.session_state.zoom_level) // 100,
                                output_format="PNG"
                            )
    
    # Query section (right column)
    with col2:
        st.header("Ask Questions")
        
        if st.session_state.vector_store is not None:
            # Initialize RAG chain
            rag_chain = setup_rag_chain(st.session_state.vector_store)
            
            # Question input
            question = st.text_input(
                "Ask a question about the document",
                placeholder="What is the main topic of this document?"
            )
            
            if st.button("Get Answer"):
                if question:
                    with st.spinner("Generating answer..."):
                        try:
                            answer = rag_chain.invoke(question)
                            st.session_state.summary_history.append({
                                "question": question,
                                "answer": answer
                            })
                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")
            
            # Display history
            if st.session_state.summary_history:
                st.header("Question History")
                for item in reversed(st.session_state.summary_history):
                    with st.expander(f"🧍User: {item['question']}", expanded=True):
                        st.write(f"🦙LLAMA:{item['answer']}")
            
            # Reset button
            if st.button("Clear Document and History"):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()