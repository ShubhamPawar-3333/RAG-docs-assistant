"""
Document Loaders Module

Handles loading documents from various formats:
- PDF files
- Markdown files
- Text files
- Directories containing mixed formats
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoaderError(Exception):
    """Custom exception for document loading errors."""
    pass


class MultiFormatDocumentLoader:
    """
    A unified document loader that handles multiple file formats.
    
    Supports:
    - PDF files (.pdf)
    - Markdown files (.md, .markdown)
    - Text files (.txt)
    - Directories containing mixed formats
    
    Example:
        >>> loader = MultiFormatDocumentLoader()
        >>> docs = loader.load_file("document.pdf")
        >>> docs = loader.load_directory("./documents")
    """
    
    # Supported file extensions and their loaders
    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".md": UnstructuredMarkdownLoader,
        ".markdown": UnstructuredMarkdownLoader,
        ".txt": TextLoader,
    }
    
    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize the document loader.
        
        Args:
            encoding: Default encoding for text files.
        """
        self.encoding = encoding
    
    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a single file and return a list of documents.
        
        Args:
            file_path: Path to the file to load.
            
        Returns:
            List of Document objects.
            
        Raises:
            DocumentLoaderError: If file cannot be loaded.
            ValueError: If file format is not supported.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DocumentLoaderError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise DocumentLoaderError(f"Not a file: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        loader_class = self.SUPPORTED_EXTENSIONS[extension]
        
        try:
            # TextLoader needs encoding parameter
            if extension == ".txt":
                loader = loader_class(str(file_path), encoding=self.encoding)
            else:
                loader = loader_class(str(file_path))
            
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata["source"] = str(file_path)
                doc.metadata["file_name"] = file_path.name
                doc.metadata["file_type"] = extension
            
            logger.info(f"Loaded {len(documents)} document(s) from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise DocumentLoaderError(f"Failed to load {file_path}: {e}") from e
    
    def load_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        extensions: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory.
            recursive: Whether to search subdirectories.
            extensions: Specific extensions to load (default: all supported).
            
        Returns:
            List of Document objects from all files.
            
        Raises:
            DocumentLoaderError: If directory cannot be accessed.
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise DocumentLoaderError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise DocumentLoaderError(f"Not a directory: {directory_path}")
        
        # Determine which extensions to load
        if extensions is None:
            extensions = list(self.SUPPORTED_EXTENSIONS.keys())
        else:
            # Validate extensions
            for ext in extensions:
                if ext not in self.SUPPORTED_EXTENSIONS:
                    raise ValueError(f"Unsupported extension: {ext}")
        
        all_documents: List[Document] = []
        
        # Find all matching files
        pattern = "**/*" if recursive else "*"
        
        for ext in extensions:
            glob_pattern = f"{pattern}{ext}"
            files = list(directory_path.glob(glob_pattern))
            
            for file_path in files:
                try:
                    docs = self.load_file(file_path)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Skipping file {file_path}: {e}")
                    continue
        
        logger.info(
            f"Loaded {len(all_documents)} document(s) from {directory_path}"
        )
        return all_documents
    
    def load_files(
        self,
        file_paths: List[Union[str, Path]]
    ) -> List[Document]:
        """
        Load multiple files.
        
        Args:
            file_paths: List of file paths to load.
            
        Returns:
            List of Document objects from all files.
        """
        all_documents: List[Document] = []
        
        for file_path in file_paths:
            try:
                docs = self.load_file(file_path)
                all_documents.extend(docs)
            except Exception as e:
                logger.warning(f"Skipping file {file_path}: {e}")
                continue
        
        return all_documents
    
    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Return list of supported file extensions."""
        return list(MultiFormatDocumentLoader.SUPPORTED_EXTENSIONS.keys())


# Convenience function for quick loading
def load_documents(
    path: Union[str, Path],
    recursive: bool = True
) -> List[Document]:
    """
    Convenience function to load documents from a file or directory.
    
    Args:
        path: Path to file or directory.
        recursive: Whether to search subdirectories (for directories).
        
    Returns:
        List of Document objects.
    """
    loader = MultiFormatDocumentLoader()
    path = Path(path)
    
    if path.is_file():
        return loader.load_file(path)
    elif path.is_dir():
        return loader.load_directory(path, recursive=recursive)
    else:
        raise DocumentLoaderError(f"Path does not exist: {path}")
