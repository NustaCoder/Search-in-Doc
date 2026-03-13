"""
RAG (Retrieval-Augmented Generation) Pipeline
Handles document storage and retrieval using ChromaDB
"""

import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import uuid
from pathlib import Path


class RAGPipeline:
    """
    RAG Pipeline for storing documents in ChromaDB and retrieving relevant information.
    """
    
    def __init__(
        self,
        persist_directory: str = "./chromadb_data",
        collection_name: str = "documents"
    ):
        """
        Initialize the RAG Pipeline with ChromaDB.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to store documents
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(
        self,
        content: str,
        metadata: Optional[Dict] = {"default": "none"},
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a document to the ChromaDB collection.
        
        Args:
            content: The document content/text
            metadata: Optional metadata dictionary
            doc_id: Optional custom document ID (generates UUID if not provided)
            
        Returns:
            The document ID that was stored
        """
        try:
            if doc_id is None:
                doc_id = str(uuid.uuid4())
            
            # Add document to collection
            self.collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata]
            )
            
            return doc_id
        except Exception as e:
            print(f"Error adding document: {e}")
            return None

    def add_documents_batch(
        self,
        documents: List[Dict[str, any]]
    ) -> List[str]:
        """
        Add multiple documents to the collection.
        
        Args:
            documents: List of dictionaries with keys:
                - 'content': str (required)
                - 'metadata': Dict (optional)
                - 'id': str (optional)
                
        Returns:
            List of document IDs that were stored
        """
        doc_ids = []
        contents = []
        metadatas = []
        
        for doc in documents:
            doc_id = doc.get('id', str(uuid.uuid4()))
            content = doc.get('content')
            metadata = doc.get('metadata', {})
            
            if content is None:
                raise ValueError("Each document must have 'content' key")
            
            doc_ids.append(doc_id)
            contents.append(content)
            metadatas.append(metadata)
        
        self.collection.add(
            ids=doc_ids,
            documents=contents,
            metadatas=metadatas
        )
        
        return doc_ids
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Retrieve relevant documents based on a query.
        
        Args:
            query: The search query
            n_results: Number of results to return (default: 5)
            where: Optional filter conditions
            
        Returns:
            Dictionary containing:
                - 'documents': List of retrieved document contents
                - 'ids': List of document IDs
                - 'metadatas': List of metadata dicts
                - 'distances': List of distance scores
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'ids': results['ids'][0] if results['ids'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else []
        }
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id: The document ID
            
        Returns:
            Dictionary with document content and metadata, or None if not found
        """
        results = self.collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"]
        )
        
        if results['documents']:
            return {
                'id': doc_id,
                'content': results['documents'][0],
                'metadata': results['metadatas'][0]
            }
        return None
    
    def update_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Update an existing document.
        
        Args:
            doc_id: The document ID to update
            content: New document content
            metadata: New metadata (optional)
            
        Returns:
            True if update was successful
        """
        try:
            if metadata is None:
                # Retrieve existing metadata
                existing = self.get_document(doc_id)
                metadata = existing['metadata'] if existing else {}
            
            self.collection.update(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata]
            )
            return True
        except Exception as e:
            print(f"Error updating document: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the collection.
        
        Args:
            doc_id: The document ID to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def search_with_filters(
        self,
        query: str,
        filters: Dict,
        n_results: int = 5
    ) -> Dict:
        """
        Search documents with metadata filters.
        
        Args:
            query: The search query
            filters: Filter conditions (ChromaDB where clause format)
            n_results: Number of results to return
            
        Returns:
            Dictionary with search results
        """
        return self.retrieve(query, n_results, where=filters)
    
    def get_all_documents(self) -> Dict:
        """
        Get all documents in the collection.
        
        Returns:
            Dictionary with all documents, ids, and metadata
        """
        results = self.collection.get(include=["documents", "metadatas"])
        
        return {
            'documents': results['documents'],
            'ids': results['ids'],
            'metadatas': results['metadatas']
        }
    
    def delete_all_documents(self) -> bool:
        """
        Delete all documents from the collection.
        
        Returns:
            True if successful
        """
        try:
            all_docs = self.get_all_documents()
            if all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
            return True
        except Exception as e:
            print(f"Error deleting all documents: {e}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        all_docs = self.get_all_documents()
        
        return {
            'total_documents': len(all_docs['ids']),
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory
        }


# Example usage and integration helper
class RAGIntegration:
    """Helper class to integrate RAG pipeline with the main application."""
    
    def __init__(self):
        """Initialize RAG pipeline."""
        self.rag = RAGPipeline(
            persist_directory="./chromadb_data",
            collection_name="documents"
        )
    
    def process_and_store_document(
        self,
        document_path: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Load a document from file and store it in ChromaDB.
        
        Args:
            document_path: Path to the document file
            metadata: Optional metadata to attach to the document
            
        Returns:
            The document ID of the stored document
        """
        # Read document content
        with open(document_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata['source'] = document_path
        metadata['filename'] = os.path.basename(document_path)
        
        # Store in ChromaDB
        doc_id = self.rag.add_document(content, metadata)
        
        return doc_id
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of results with content, metadata, and relevance score
        """
        results = self.rag.retrieve(query, n_results)
        
        formatted_results = []
        for doc, doc_id, metadata, distance in zip(
            results['documents'],
            results['ids'],
            results['metadatas'],
            results['distances']
        ):
            formatted_results.append({
                'id': doc_id,
                'content': doc,
                'metadata': metadata,
                'relevance_score': 1 - distance  # Convert distance to similarity
            })
        
        return formatted_results
    
    def get_rag_pipeline(self) -> RAGPipeline:
        """Get the RAG pipeline instance."""
        return self.rag

# if __name__ == "__main__":
#     # Example usage
#     rag = RAGPipeline()
    
#     # Add sample documents
#     sample_docs = [
#         {
#             'content': 'Python is a high-level programming language known for its simplicity and readability.',
#             'metadata': {'type': 'programming', 'language': 'Python'}
#         },
#         {
#             'content': 'Machine Learning is a subset of artificial intelligence that enables systems to learn from data.',
#             'metadata': {'type': 'AI', 'topic': 'Machine Learning'}
#         },
#         {
#             'content': 'ChromaDB is an AI-native open-source vector database for building LLM applications.',
#             'metadata': {'type': 'database', 'tool': 'ChromaDB'}
#         }
#     ]
    
#     # Store documents
#     # doc_ids = rag.add_documents_batch(sample_docs)
#     # print(f"Stored {len(doc_ids)} documents: {doc_ids}")
    
#     # Search for relevant documents
#     query = "What is machine learning?"
#     results = rag.retrieve(query, n_results=2)
    
#     print(f"\nSearch results for '{query}':")
#     for doc, doc_id, metadata in zip(results['documents'], results['ids'], results['metadatas']):
#         print(f"\nDocument ID: {doc_id}")
#         print(f"Content: {doc}")
#         print(f"Metadata: {metadata}")
    
#     # Get collection stats
#     stats = rag.get_collection_stats()
#     print(f"\nCollection Stats: {stats}")
