# Retrieval-Augmented Generation (RAG) PDF Query System

## Overview
This project is a Streamlit-based application that enables users to query vector-embedded PDF documents using a **Retrieval-Augmented Generation (RAG)** pipeline. The system utilizes **Azure OpenAI** for embeddings and answer generation, **Azure Cognitive Search** for information retrieval, and **Azure Blob Storage** for storing and managing PDFs.

## Features
- Upload and process PDFs for chunking and embedding.
- Perform vector-based searches using **Azure Cognitive Search**.
- Generate responses using **Azure OpenAI's GPT models**.
- Streamlit-powered UI for interactive querying.

## Directory Structure
```
|-- config.py               # Loads environment variables and global constants
|-- main.py                 # Streamlit application entry point
|-- agent_tools.py          # Implements the RAG pipeline logic
|-- azure_blob.py           # Handles Azure Blob Storage interactions
|-- azure_search.py         # Handles Azure Cognitive Search queries
|-- pdf_chunk_processing.py # Extracts, chunks, and embeds PDFs
```

## Installation
### Prerequisites
- Python 3.8+
- An **Azure OpenAI** and **Azure Cognitive Search** subscription
- Environment variables configured in a `.env` file (see below)

### Setup
1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up a `.env` file with your Azure credentials:
   ```sh
   AZURE_OPENAI_KEY=<your-openai-key>
   AZURE_SEARCH_ENDPOINT=<your-search-endpoint>
   AZURE_SEARCH_KEY=<your-search-key>
   ```

## Running the Application
Start the Streamlit UI by running:
```sh
streamlit run main.py
```

## Usage
1. Upload a PDF document.
2. The system processes and chunks the document.
3. Enter a query in the Streamlit UI.
4. The system retrieves relevant chunks and generates a response using GPT.

## Components
### 1. **`pdf_chunk_processing.py`**
- Extracts text from PDFs using `PyPDF2`.
- Chunks text using **LlamaIndex’s SemanticSplitterNodeParser**.
- Generates embeddings using **Azure OpenAI**.

### 2. **`azure_search.py`**
- Sends vector search queries to **Azure Cognitive Search**.
- Retrieves the most relevant document chunks.

### 3. **`agent_tools.py`**
- Implements the **RAG pipeline**, orchestrating query embedding, retrieval, and response generation.

### 4. **`azure_blob.py`**
- Manages **Azure Blob Storage** for uploading and retrieving PDFs.

### 5. **`config.py`**
- Loads environment variables and manages logging.

## Future Improvements
- Implement authentication for secure API access.
- Optimize chunking strategy for better retrieval.
- Support additional document types (Word, HTML, etc.).



