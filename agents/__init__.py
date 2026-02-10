"""
Multi-Agent File Processing System
===================================

This module implements a multi-agent architecture where specialized agents
handle different file types. Each agent is responsible for:
- Extracting text/content from its supported file format
- Format-specific processing and optimization
- Error handling and validation
- Metadata extraction

Agents:
- PDFAgent: Handles PDF files
- DOCXAgent: Handles DOCX/DOC files
- TXTAgent: Handles plain text files
- CSVAgent: Handles CSV files
- XLSXAgent: Handles Excel XLSX files
- PPTXAgent: Handles PowerPoint PPTX files (optional)

Usage:
    from agents import AgentCoordinator
    
    coordinator = AgentCoordinator()
    result = coordinator.process_file(file_path, original_filename)
"""

from .base_agent import BaseAgent
from .pdf_agent import PDFAgent
from .docx_agent import DOCXAgent
from .txt_agent import TXTAgent
from .csv_agent import CSVAgent
from .xlsx_agent import XLSXAgent
from .coordinator import AgentCoordinator

__all__ = [
    'BaseAgent',
    'PDFAgent',
    'DOCXAgent',
    'TXTAgent',
    'CSVAgent',
    'XLSXAgent',
    'AgentCoordinator',
]

