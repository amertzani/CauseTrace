"""
Base Agent Class
================

Abstract base class for all file processing agents.
Each specialized agent must implement the process() method.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import os
from datetime import datetime


class BaseAgent(ABC):
    """
    Base class for all file processing agents.
    
    Each agent specializes in processing a specific file type and provides:
    - Text extraction
    - Format-specific processing
    - Metadata extraction
    - Error handling
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize the agent.
        
        Args:
            agent_name: Human-readable name of the agent
        """
        self.agent_name = agent_name
        self.supported_extensions = []
    
    @abstractmethod
    def can_process(self, file_path: str) -> bool:
        """
        Check if this agent can process the given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if this agent can process the file, False otherwise
        """
        pass
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from the file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
            
        Raises:
            Exception: If extraction fails
        """
        pass
    
    def extract_metadata(self, file_path: str) -> Dict:
        """
        Extract metadata from the file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with metadata (file_size, pages, etc.)
        """
        metadata = {
            'agent': self.agent_name,
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'processed_at': datetime.now().isoformat(),
        }
        return metadata
    
    def process(self, file_path: str, original_filename: Optional[str] = None) -> Dict:
        """
        Process a file and return extracted content and metadata.
        
        Args:
            file_path: Path to the file
            original_filename: Original filename (if different from file_path)
            
        Returns:
            Dictionary with:
            - text: Extracted text content
            - metadata: File metadata
            - success: Whether processing succeeded
            - error: Error message if failed
        """
        try:
            if not os.path.exists(file_path):
                return {
                    'text': '',
                    'metadata': self.extract_metadata(file_path),
                    'success': False,
                    'error': f'File not found: {file_path}'
                }
            
            if not self.can_process(file_path):
                return {
                    'text': '',
                    'metadata': self.extract_metadata(file_path),
                    'success': False,
                    'error': f'Agent {self.agent_name} cannot process this file type'
                }
            
            # Extract text
            text = self.extract_text(file_path)
            
            # Extract metadata
            metadata = self.extract_metadata(file_path)
            metadata['original_filename'] = original_filename or os.path.basename(file_path)
            metadata['text_length'] = len(text)
            
            # Validate extraction
            if not text or len(text.strip()) < 10:
                return {
                    'text': text,
                    'metadata': metadata,
                    'success': False,
                    'error': 'Extracted text is too short or empty. File may be empty or corrupted.'
                }
            
            return {
                'text': text,
                'metadata': metadata,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            metadata = self.extract_metadata(file_path)
            metadata['original_filename'] = original_filename or os.path.basename(file_path)
            return {
                'text': '',
                'metadata': metadata,
                'success': False,
                'error': f'Error processing file: {str(e)}'
            }
    
    def get_supported_extensions(self) -> list:
        """
        Get list of file extensions this agent supports.
        
        Returns:
            List of supported extensions (e.g., ['.pdf', '.PDF'])
        """
        return self.supported_extensions

