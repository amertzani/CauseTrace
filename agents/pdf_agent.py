"""
PDF Agent
=========

Specialized agent for processing PDF files.
Uses PyPDF2 for text extraction.
"""

import os
import PyPDF2
from typing import Dict
from .base_agent import BaseAgent


class PDFAgent(BaseAgent):
    """Agent specialized in processing PDF files."""
    
    def __init__(self):
        super().__init__("PDF Agent")
        self.supported_extensions = ['.pdf', '.PDF']
    
    def can_process(self, file_path: str) -> bool:
        """Check if file is a PDF."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_extensions
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num} ---\n"
                            text += page_text + "\n"
                    except Exception as e:
                        # Continue with other pages if one fails
                        text += f"\n--- Page {page_num} (extraction error: {e}) ---\n"
                
                return text.strip()
                
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def extract_metadata(self, file_path: str) -> Dict:
        """Extract PDF-specific metadata."""
        metadata = super().extract_metadata(file_path)
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata['pages'] = len(pdf_reader.pages)
                
                # Try to get PDF metadata
                if pdf_reader.metadata:
                    metadata['pdf_title'] = pdf_reader.metadata.get('/Title', '')
                    metadata['pdf_author'] = pdf_reader.metadata.get('/Author', '')
                    metadata['pdf_subject'] = pdf_reader.metadata.get('/Subject', '')
        except Exception:
            # If metadata extraction fails, continue without it
            metadata['pages'] = 0
        
        return metadata

