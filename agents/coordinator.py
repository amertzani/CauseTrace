"""
Agent Coordinator
=================

Coordinates multiple specialized agents to process different file types.
Routes files to the appropriate agent based on file extension.
"""

import os
from typing import List, Dict, Optional
from .base_agent import BaseAgent
from .pdf_agent import PDFAgent
from .docx_agent import DOCXAgent
from .txt_agent import TXTAgent
from .csv_agent import CSVAgent
from .xlsx_agent import XLSXAgent


class AgentCoordinator:
    """
    Coordinates multiple file processing agents.
    
    Routes files to specialized agents based on file extension.
    Provides a unified interface for file processing.
    """
    
    def __init__(self):
        """Initialize the coordinator with all available agents."""
        self.agents: List[BaseAgent] = [
            PDFAgent(),
            DOCXAgent(),
            TXTAgent(),
            CSVAgent(),
            XLSXAgent(),
        ]
        
        # Build extension to agent mapping for fast lookup
        self.extension_map: Dict[str, BaseAgent] = {}
        for agent in self.agents:
            for ext in agent.get_supported_extensions():
                ext_lower = ext.lower()
                if ext_lower not in self.extension_map:
                    self.extension_map[ext_lower] = agent
    
    def get_agent_for_file(self, file_path: str) -> Optional[BaseAgent]:
        """
        Find the appropriate agent for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Agent that can process the file, or None if no agent supports it
        """
        ext = os.path.splitext(file_path)[1].lower()
        return self.extension_map.get(ext)
    
    def can_process(self, file_path: str) -> bool:
        """
        Check if any agent can process the file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if an agent can process the file, False otherwise
        """
        return self.get_agent_for_file(file_path) is not None
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get all supported file extensions.
        
        Returns:
            List of supported extensions
        """
        return list(set(self.extension_map.keys()))
    
    def process_file(self, file_path: str, original_filename: Optional[str] = None) -> Dict:
        """
        Process a file using the appropriate agent.
        
        Args:
            file_path: Path to the file
            original_filename: Original filename (if different from file_path)
            
        Returns:
            Dictionary with:
            - text: Extracted text content
            - metadata: File metadata including agent name
            - success: Whether processing succeeded
            - error: Error message if failed
            - agent: Name of the agent that processed the file
        """
        agent = self.get_agent_for_file(file_path)
        
        if agent is None:
            ext = os.path.splitext(file_path)[1].lower()
            supported = ', '.join(sorted(self.get_supported_extensions()))
            return {
                'text': '',
                'metadata': {
                    'file_path': file_path,
                    'file_name': original_filename or os.path.basename(file_path),
                },
                'success': False,
                'error': f'Unsupported file type: {ext}. Supported formats: {supported}',
                'agent': None
            }
        
        # Process with the appropriate agent
        result = agent.process(file_path, original_filename)
        result['agent'] = agent.agent_name
        return result
    
    def process_files(self, file_paths: List[str], original_filenames: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        Process multiple files.
        
        Args:
            file_paths: List of file paths
            original_filenames: Optional dict mapping file paths to original filenames
            
        Returns:
            List of processing results (one per file)
        """
        if original_filenames is None:
            original_filenames = {}
        
        results = []
        for file_path in file_paths:
            original_filename = original_filenames.get(file_path)
            result = self.process_file(file_path, original_filename)
            results.append(result)
        
        return results
    
    def get_agent_info(self) -> Dict[str, List[str]]:
        """
        Get information about all registered agents.
        
        Returns:
            Dictionary mapping agent names to their supported extensions
        """
        info = {}
        for agent in self.agents:
            info[agent.agent_name] = agent.get_supported_extensions()
        return info

