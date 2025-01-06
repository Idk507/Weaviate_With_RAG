# RAG Implementation with Weaviate and LangChain

This repository demonstrates the implementation of a Retrieval-Augmented Generation (RAG) system using Weaviate as the vector database and LangChain for the orchestration. The system processes PDF documents, stores them in Weaviate, and enables semantic search with question-answering capabilities.

Let me explain Weaviate and its key features.

Weaviate is an open-source vector database that allows you to store both objects and vectors, making it particularly powerful for AI-powered applications. Here are its key aspects:

1. Vector Database + Object Storage
- Combines traditional object storage with vector embeddings
- Can store data objects (like text, images, or any structured data) alongside their vector representations
- Supports semantic search through vector similarity

2. Key Features
- Vector Search: Finds similar items based on vector proximity
- Multi-tenancy: Can handle multiple separate data spaces
- Schema Management: Flexible data modeling with custom classes and properties
- RESTful API & GraphQL Interface: Multiple ways to interact with the database
- Cloud & Self-hosted Options: Can be deployed anywhere

3. Main Use Cases
- Semantic Search: Finding content based on meaning rather than exact keywords
- Question Answering: Part of RAG (Retrieval Augmented Generation) systems
- Recommendation Systems: Suggesting similar items
- Image Search: Finding visually similar images
- Classification: Organizing and categorizing data

4. Architecture
- Uses ANN (Approximate Nearest Neighbor) algorithms for efficient vector search
- Supports multiple vectorization modules (like transformers)
- Built with scalability in mind (horizontal scaling)
- Real-time indexing and updates

5. Integration Capabilities
- Works well with machine learning frameworks
- Native clients in multiple languages (Python, JavaScript, Go)
- LangChain integration for easy RAG implementation
- Support for popular embedding models

6. Unique Features
- Cross-references between objects
- Filtered vector search
- Modular architecture for different vector index types
- Support for custom modules

7. Data Organization
- Uses "classes" (similar to tables in traditional databases)
- Properties can be scalar values or vectors
- Supports references between objects
- Automatic vectorization of data

Examples of Basic Operations:

```python
# Adding data
client.data_object.create(
    class_name="Article",
    properties={
        "title": "My Article",
        "content": "Article content..."
    }
)

# Vector search
results = client.query.get("Article") \
    .with_near_text({"concepts": ["search query"]}) \
    .do()
```

Common Patterns:
1. Content Search
```python
# Search for similar content
results = collection.query.near_text(
    query="your search term",
    limit=5
)
```

2. Hybrid Search
```python
# Combine keyword and vector search
results = collection.query.hybrid(
    query="search term",
    alpha=0.5,  # Balance between vector and keyword search
    limit=5
)
```

When using Weaviate in a RAG system (like your implementation), it serves several key functions:
1. Stores document chunks and their embeddings
2. Enables semantic retrieval of relevant context
3. Provides efficient vector similarity search
4. Maintains relationships between document parts

Best Practices:
1. Properly structure your schema based on your use case
2. Choose appropriate vector dimensions for your embedding model
3. Use batch operations for large data imports
4. Implement proper error handling and retries
5. Monitor system resources and performance
6. Regular backups of your vector database


   
## Prerequisites

Install the required packages:
```bash
pip install langchain langchain-community tiktoken pypdf rapidocr-onnxruntime
pip install -U weaviate-client
pip install -Uqq langchain-weaviate
pip install sentence-transformers
```

## Configuration

1. Set up Weaviate credentials:
```python
WEAVIATE_API_KEY = "your-api-key"
WEAVIATE_CLUSTER = "your-cluster-url"
```

2. Initialize embedding model:
```python
from langchain.embeddings import HuggingFaceEmbeddings
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
```

## Document Processing

1. Load and process PDF documents:
```python
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("path/to/your/pdf", extract_images=True)
pages = loader.load()
```

2. Split documents into chunks:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = text_splitter.split_documents(pages)
```

## Weaviate Setup

1. Connect to Weaviate:
```python
import weaviate
from weaviate.classes.init import Auth

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
)
```

2. Create vector store:
```python
from langchain_weaviate.vectorstores import WeaviateVectorStore
db = WeaviateVectorStore.from_documents(docs, embeddings, client=client, index_name="LangChain")
```

## RAG Pipeline Setup

1. Create prompt template:
```python
from langchain.prompts import ChatPromptTemplate
template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
```

2. Set up language model:
```python
from langchain import HuggingFaceHub
model = HuggingFaceHub(
    huggingfacehub_api_token="your-token",
    repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    model_kwargs={"temperature": 1, "max_length": 180}
)
```

3. Create RAG chain:
```python
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

output_parser = StrOutputParser()
retriever = db.as_retriever()
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)
```

## Usage

Query the system:
```python
response = rag_chain.invoke("your question here")
print(response)
```

## Features

- PDF document processing with image extraction
- Semantic text chunking
- Vector storage in Weaviate
- Retrieval-augmented generation using LangChain
- Integration with Hugging Face models
- Configurable prompt templates

## Utility Functions

The system includes several utility functions for:
- Document aggregation
- Collection statistics
- Object properties inspection
- Similarity search

## Notes

- Ensure proper environment setup and API key configuration
- Adjust chunk size and overlap based on your specific use case
- Monitor token usage and API rate limits
- Consider implementing error handling and retries for production use
