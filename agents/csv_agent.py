"""
CSV Agent
=========

Specialized agent for processing CSV files.
Uses pandas for structured data extraction and intelligent formatting.
"""

import os
import pandas as pd
from typing import Dict
from .base_agent import BaseAgent


class CSVAgent(BaseAgent):
    """Agent specialized in processing CSV files."""
    
    def __init__(self):
        super().__init__("CSV Agent")
        self.supported_extensions = ['.csv', '.CSV']
    
    def can_process(self, file_path: str) -> bool:
        """Check if file is a CSV."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext == '.csv'
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from CSV file.
        Formats data in a readable way for knowledge extraction.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Formatted text representation of CSV data
        """
        try:
            # Try different encodings and separators
            df = None
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            separators = [',', ';', '\t', '|']
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, sep=sep, on_bad_lines='skip')
                        if len(df.columns) > 1:  # Valid CSV with multiple columns
                            break
                    except Exception:
                        continue
                if df is not None and len(df.columns) > 1:
                    break
            
            if df is None or len(df.columns) == 0:
                raise Exception("Could not parse CSV file with any encoding/separator")
            
            # Format CSV data for knowledge extraction
            text_parts = []
            
            # Header information
            text_parts.append(f"CSV Data File")
            text_parts.append(f"Total rows: {len(df)}")
            text_parts.append(f"Total columns: {len(df.columns)}")
            text_parts.append(f"\nColumns: {', '.join(df.columns.tolist())}")
            
            # Data summary
            text_parts.append("\n--- Data Summary ---")
            for col in df.columns:
                non_null = df[col].notna().sum()
                text_parts.append(f"{col}: {non_null} non-null values")
                if df[col].dtype in ['int64', 'float64']:
                    text_parts.append(f"  Range: {df[col].min()} to {df[col].max()}")
            
            # Sample data (first 20 rows for knowledge extraction)
            text_parts.append("\n--- Sample Data (first 20 rows) ---")
            sample_size = min(20, len(df))
            for idx, row in df.head(sample_size).iterrows():
                row_data = []
                for col in df.columns:
                    value = str(row[col]) if pd.notna(row[col]) else 'N/A'
                    row_data.append(f"{col}: {value}")
                text_parts.append(f"Row {idx + 1}: {' | '.join(row_data)}")
            
            # If there are more rows, mention it
            if len(df) > sample_size:
                text_parts.append(f"\n... ({len(df) - sample_size} more rows)")
            
            return "\n".join(text_parts).strip()
            
        except Exception as e:
            raise Exception(f"Error reading CSV: {str(e)}")
    
    def extract_metadata(self, file_path: str) -> Dict:
        """Extract CSV-specific metadata."""
        metadata = super().extract_metadata(file_path)
        
        try:
            df = pd.read_csv(file_path, on_bad_lines='skip')
            metadata['rows'] = len(df)
            metadata['columns'] = len(df.columns)
            metadata['column_names'] = df.columns.tolist()
            metadata['data_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        except Exception:
            metadata['rows'] = 0
            metadata['columns'] = 0
            metadata['column_names'] = []
        
        return metadata

