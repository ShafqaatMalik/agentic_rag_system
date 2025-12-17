"""
Vector store setup and retrieval operations using ChromaDB.
"""

import os
from typing import List, Optional
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

import structlog

from app.config import get_settings

logger = structlog.get_logger()


class VectorStoreManager:
    """Manages ChromaDB vector store operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self._embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
        self._vectorstore: Optional[Chroma] = None
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    @property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Lazy initialization of embeddings."""
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self.settings.embedding_model,
                google_api_key=self.settings.google_api_key
            )
        return self._embeddings
    
    @property
    def vectorstore(self) -> Chroma:
        """Lazy initialization of vector store."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.settings.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.settings.chroma_persist_directory
            )
        return self._vectorstore
    
    def get_retriever(self, k: Optional[int] = None):
        """Get a retriever instance."""
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k or self.settings.retrieval_k}
        )
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents to add")
            return []
        
        # Split documents into chunks
        chunks = self._text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        
        # Add to vector store
        ids = self.vectorstore.add_documents(chunks)
        logger.info(f"Added {len(ids)} chunks to vector store")
        
        return ids
    
    async def ingest_file(self, file_path: str, original_filename: Optional[str] = None) -> List[str]:
        """Ingest a single file into the vector store.

        Args:
            file_path: Path to the file to ingest
            original_filename: Optional original filename to use in metadata (useful for temp files)
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Select appropriate loader
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(path))
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        # Load and add documents
        documents = loader.load()

        # Use original filename if provided, otherwise use the file path name
        source_name = original_filename if original_filename else path.name

        # Add source metadata
        for doc in documents:
            doc.metadata["source"] = source_name
            doc.metadata["file_path"] = str(path)

        logger.info(f"Loaded {len(documents)} pages from {source_name}")

        return await self.add_documents(documents)
    
    async def ingest_directory(self, directory_path: str) -> dict:
        """Ingest all supported files from a directory."""
        path = Path(directory_path)
        
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        results = {"success": [], "failed": []}
        supported_extensions = {".pdf", ".txt", ".md"}
        
        for file_path in path.iterdir():
            if file_path.suffix.lower() in supported_extensions:
                try:
                    await self.ingest_file(str(file_path))
                    results["success"].append(file_path.name)
                except Exception as e:
                    logger.error(f"Failed to ingest {file_path.name}: {e}")
                    results["failed"].append({"file": file_path.name, "error": str(e)})
        
        logger.info(
            f"Ingestion complete: {len(results['success'])} succeeded, "
            f"{len(results['failed'])} failed"
        )
        
        return results
    
    async def similarity_search(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> List[Document]:
        """Perform similarity search."""
        return self.vectorstore.similarity_search(
            query, 
            k=k or self.settings.retrieval_k
        )
    
    async def similarity_search_with_score(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> List[tuple[Document, float]]:
        """Perform similarity search with relevance scores."""
        return self.vectorstore.similarity_search_with_score(
            query, 
            k=k or self.settings.retrieval_k
        )
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the vector store collection."""
        collection = self.vectorstore._collection
        return {
            "name": collection.name,
            "count": collection.count(),
        }

    def get_documents_list(self) -> List[dict]:
        """Get list of unique documents in the collection."""
        try:
            collection = self.vectorstore._collection
            # Get all documents with their metadata
            results = collection.get(include=["metadatas"])

            if not results or not results.get("metadatas"):
                return []

            # Extract unique documents by source
            unique_docs = {}
            for metadata in results["metadatas"]:
                source = metadata.get("source", "Unknown")
                if source not in unique_docs:
                    unique_docs[source] = {
                        "name": source,
                        "path": metadata.get("file_path", ""),
                        "page_count": 1
                    }
                else:
                    unique_docs[source]["page_count"] += 1

            # Convert to list and sort by name
            documents_list = sorted(unique_docs.values(), key=lambda x: x["name"])
            logger.info(f"Retrieved {len(documents_list)} unique documents")

            return documents_list
        except Exception as e:
            logger.error(f"Error getting documents list: {e}")
            return []
    
    def delete_document(self, document_name: str) -> int:
        """Delete all chunks of a specific document by its source name.

        Args:
            document_name: The source name of the document to delete

        Returns:
            Number of chunks deleted
        """
        try:
            collection = self.vectorstore._collection

            # Get all IDs for this document
            results = collection.get(
                where={"source": document_name},
                include=["metadatas"]
            )

            if not results or not results.get("ids"):
                logger.warning(f"No chunks found for document: {document_name}")
                return 0

            # Delete the chunks
            ids_to_delete = results["ids"]
            collection.delete(ids=ids_to_delete)

            logger.info(f"Deleted {len(ids_to_delete)} chunks for document: {document_name}")
            return len(ids_to_delete)

        except Exception as e:
            logger.error(f"Error deleting document {document_name}: {e}")
            raise

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.vectorstore._collection.delete(where={})
        logger.info("Cleared all documents from collection")


# Singleton instance
_vectorstore_manager: Optional[VectorStoreManager] = None


def get_vectorstore_manager() -> VectorStoreManager:
    """Get the singleton VectorStoreManager instance."""
    global _vectorstore_manager
    if _vectorstore_manager is None:
        _vectorstore_manager = VectorStoreManager()
    return _vectorstore_manager
