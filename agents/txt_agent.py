"""
TXT Agent
=========

Specialized agent for processing plain text files.
Handles various text encodings.
"""

import os
from typing import Dict
from .base_agent import BaseAgent

# Try to import chardet for encoding detection
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False


class TXTAgent(BaseAgent):
    """Agent specialized in processing plain text files."""
    
    def __init__(self):
        super().__init__("TXT Agent")
        self.supported_extensions = ['.txt', '.TXT', '.text', '.TEXT', '.md', '.MD', '.markdown', '.MARKDOWN']
    
    def can_process(self, file_path: str) -> bool:
        """Check if file is a text file."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ['.txt', '.text', '.md', '.markdown']
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from TXT file.
        Automatically detects encoding.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Extracted text content
        """
        try:
            # Try UTF-8 first (most common)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read().strip()
            except UnicodeDecodeError:
                # If UTF-8 fails, try to detect encoding or use fallback encodings
                with open(file_path, 'rb') as file:
                    raw_data = file.read()
                    
                    # Try chardet if available
                    if CHARDET_AVAILABLE:
                        try:
                            detected = chardet.detect(raw_data)
                            encoding = detected.get('encoding', 'utf-8')
                            if encoding and encoding.lower() != 'ascii':
                                return raw_data.decode(encoding).strip()
                        except Exception:
                            pass
                    
                    # Fallback to common encodings
                    for enc in ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8']:
                        try:
                            return raw_data.decode(enc).strip()
                        except UnicodeDecodeError:
                            continue
                    
                    # Last resort: decode with errors='replace'
                    return raw_data.decode('utf-8', errors='replace').strip()
                    
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")
    
    def extract_metadata(self, file_path: str) -> Dict:
        """Extract TXT-specific metadata."""
        metadata = super().extract_metadata(file_path)
        
        try:
            # Count lines
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                
                # Try to detect encoding
                encoding = 'utf-8'
                if CHARDET_AVAILABLE:
                    try:
                        detected = chardet.detect(raw_data)
                        encoding = detected.get('encoding', 'utf-8')
                    except Exception:
                        pass
                
                try:
                    text = raw_data.decode(encoding)
                    metadata['lines'] = len(text.splitlines())
                    metadata['encoding'] = encoding
                except:
                    metadata['lines'] = 0
                    metadata['encoding'] = 'unknown'
        except Exception:
            metadata['lines'] = 0
            metadata['encoding'] = 'unknown'
        
        return metadata

