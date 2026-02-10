"""
CSV Agent
=========

Specialized agent for processing CSV files.
Uses pandas for structured data extraction and intelligent formatting.
Enhanced with direct knowledge graph storage for tabular data.
"""

import os
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .base_agent import BaseAgent


class CSVAgent(BaseAgent):
    """Agent specialized in processing CSV files with tabular data extraction."""
    
    def __init__(self):
        super().__init__("CSV Agent")
        self.supported_extensions = ['.csv', '.CSV']
    
    def can_process(self, file_path: str) -> bool:
        """Check if file is a CSV."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext == '.csv'
    
    def _read_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Read CSV file with multiple encoding and separator attempts.
        
        Returns:
            DataFrame if successful, None otherwise
        """
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        separators = [',', ';', '\t', '|']
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep, on_bad_lines='skip', low_memory=False)
                    if len(df.columns) > 1:  # Valid CSV with multiple columns
                        return df
                except Exception:
                    continue
        return None
    
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
            df = self._read_csv(file_path)
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
            df = self._read_csv(file_path)
            if df is not None:
                metadata['rows'] = len(df)
                metadata['columns'] = len(df.columns)
                metadata['column_names'] = df.columns.tolist()
                metadata['data_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            else:
                metadata['rows'] = 0
                metadata['columns'] = 0
                metadata['column_names'] = []
        except Exception:
            metadata['rows'] = 0
            metadata['columns'] = 0
            metadata['column_names'] = []
        
        return metadata
    
    def process_tabular_data_to_graph(
        self, 
        file_path: str, 
        source_document: str = None,
        uploaded_at: str = None
    ) -> Dict[str, any]:
        """
        Process CSV file and directly store structured facts in knowledge graph.
        
        This method:
        1. Reads the CSV file
        2. Analyzes column headers
        3. Processes each row
        4. Creates structured facts from tabular data
        5. Stores facts directly in the knowledge graph
        
        Args:
            file_path: Path to CSV file
            source_document: Name of source document
            uploaded_at: ISO timestamp when uploaded
            
        Returns:
            Dictionary with processing results:
            - success: bool
            - facts_added: int
            - facts_skipped: int
            - column_facts: int
            - row_facts: int
            - relationship_facts: int
            - error: str if failed
        """
        from knowledge import fact_exists, add_fact_source_document, add_fact_agent, add_fact_details, graph
        import rdflib
        from urllib.parse import quote
        from datetime import datetime
        
        if source_document is None:
            source_document = os.path.basename(file_path)
        if uploaded_at is None:
            uploaded_at = datetime.now().isoformat()
        
        try:
            # Read CSV
            df = self._read_csv(file_path)
            if df is None or len(df.columns) == 0:
                return {
                    'success': False,
                    'error': 'Could not parse CSV file',
                    'facts_added': 0,
                    'facts_skipped': 0
                }
            
            # Clean column names (remove whitespace, special chars)
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
            
            facts_added = 0
            facts_skipped = 0
            column_facts = 0
            row_facts = 0
            relationship_facts = 0
            
            # OPTIMIZATION: Build a set of existing facts ONCE for fast lookups
            # For very large graphs, we can skip duplicate checking to speed up CSV processing
            # Duplicates will just be skipped when adding to graph
            print(f"📊 Building fact lookup set for fast duplicate checking...")
            existing_facts_set = set()
            from knowledge import graph as kg_graph
            from urllib.parse import unquote
            
            graph_size = len(kg_graph)
            print(f"   Graph has {graph_size} total triples (including metadata)")
            
            # Only build lookup set if graph is reasonably sized (< 10k facts)
            # For larger graphs, skip duplicate checking to speed up processing
            if graph_size < 10000:
                # Build set of normalized fact keys (subject|predicate|object) in one pass
                fact_count = 0
                for s, p, o in kg_graph:
                    # Skip metadata triples (faster check)
                    p_str = str(p)
                    if 'fact_' in p_str or 'has_details' in p_str or 'source_document' in p_str or 'uploaded_at' in p_str or 'processed_by' in p_str:
                        continue
                    
                    fact_count += 1
                    # Normalize and create key (simplified for speed)
                    try:
                        s_str = str(s).split(':')[-1] if ':' in str(s) else str(s)
                        p_str = str(p).split(':')[-1] if ':' in str(p) else str(p)
                        o_str = str(o)
                        
                        # Quick normalization (skip unquote for speed)
                        s_str = s_str.replace('_', ' ').lower().strip()
                        p_str = p_str.replace('_', ' ').lower().strip()
                        o_str = str(o_str).lower().strip()
                        
                        fact_key = (s_str, p_str, o_str)
                        existing_facts_set.add(fact_key)
                    except:
                        continue
                
                print(f"✅ Built lookup set with {len(existing_facts_set)} facts from {fact_count} main triples")
            else:
                print(f"⚠️  Graph is large ({graph_size} triples), skipping duplicate check for speed")
                print(f"   Duplicates will be handled by RDFLib graph automatically")
            
            # BATCH PROCESSING: Collect all facts first, then add to graph in batch
            facts_to_add = []  # List of (subject, predicate, object) tuples
            
            # Helper function to collect facts (doesn't add to graph yet)
            def collect_fact(subject, predicate, obj):
                """Collect a fact for batch addition."""
                # Fast lookup using set (skip if set is empty for large graphs)
                if len(existing_facts_set) > 0:
                    subject_norm = str(subject).lower().strip().replace('_', ' ')
                    predicate_norm = str(predicate).lower().strip().replace('_', ' ')
                    object_norm = str(obj).lower().strip()
                    fact_key = (subject_norm, predicate_norm, object_norm)
                    
                    if fact_key in existing_facts_set:
                        return False
                    
                    # Add to lookup set for future checks in this batch
                    existing_facts_set.add(fact_key)
                
                # Add to collection
                facts_to_add.append((subject, predicate, obj))
                return True
            
            # Helper function to batch add all collected facts to graph (optimized)
            def batch_add_facts_to_graph():
                """Add all collected facts to graph in one optimized batch operation."""
                from knowledge import graph
                total_facts = len(facts_to_add)
                print(f"📊 Batch adding {total_facts} facts to graph...")
                
                if total_facts == 0:
                    return
                
                # Add facts directly without pre-creating URIs (faster for small batches)
                # For large batches, we'll add in chunks with progress updates
                if total_facts > 1000:
                    chunk_size = 500  # Smaller chunks for better progress feedback
                    for i in range(0, total_facts, chunk_size):
                        chunk = facts_to_add[i:i + chunk_size]
                        for subject, predicate, obj in chunk:
                            subject_clean = str(subject).strip().replace(' ', '_')
                            predicate_clean = str(predicate).strip().replace(' ', '_')
                            object_clean = str(obj).strip()
                            
                            subject_uri = rdflib.URIRef(f"urn:entity:{quote(subject_clean, safe='')}")
                            predicate_uri = rdflib.URIRef(f"urn:predicate:{quote(predicate_clean, safe='')}")
                            
                            graph.add((subject_uri, predicate_uri, rdflib.Literal(object_clean)))
                        
                        if (i + chunk_size) % 2000 == 0 or (i + chunk_size) >= total_facts:
                            print(f"   Added {min(i + chunk_size, total_facts)}/{total_facts} facts...")
                else:
                    # For smaller batches, add all at once
                    for subject, predicate, obj in facts_to_add:
                        subject_clean = str(subject).strip().replace(' ', '_')
                        predicate_clean = str(predicate).strip().replace(' ', '_')
                        object_clean = str(obj).strip()
                        
                        subject_uri = rdflib.URIRef(f"urn:entity:{quote(subject_clean, safe='')}")
                        predicate_uri = rdflib.URIRef(f"urn:predicate:{quote(predicate_clean, safe='')}")
                        
                        graph.add((subject_uri, predicate_uri, rdflib.Literal(object_clean)))
                
                print(f"✅ Added {total_facts} facts to graph")
            
            # Step 1: Collect column metadata facts
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            for col in df.columns:
                # Fact: "CSV file has column X"
                if collect_fact(file_name, "has_column", col):
                    facts_added += 1
                    column_facts += 1
                else:
                    facts_skipped += 1
            
            # Step 2: Identify potential entity columns (columns that might be identifiers)
            # Look for columns with unique values or common ID patterns
            entity_columns = []
            for col in df.columns:
                # Check if column looks like an ID or entity name
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                col_lower = col.lower()
                is_id_like = any(keyword in col_lower for keyword in ['id', 'name', 'entity', 'item', 'product', 'drug', 'compound'])
                
                if is_id_like or (unique_ratio > 0.7 and unique_ratio <= 1.0):
                    entity_columns.append(col)
            
            # If no obvious entity columns, use first column as default
            if not entity_columns and len(df.columns) > 0:
                entity_columns = [df.columns[0]]
            
            # Step 3: Process each row and create facts
            # Limit processing to prevent memory issues and timeouts
            # Further reduced for faster initial processing
            max_rows = 2000  # Process up to 2,000 rows initially (can be increased if needed)
            rows_to_process = min(len(df), max_rows)
            
            print(f"📊 Processing {rows_to_process} rows from CSV (total: {len(df)} rows, {len(df.columns)} columns)")
            if len(df) > max_rows:
                print(f"   Note: Processing first {max_rows} rows only for speed. Remaining rows will be skipped.")
            
            # Progress tracking - more frequent updates
            progress_interval = max(20, rows_to_process // 20)  # Show progress every 5%
            
            start_time = datetime.now()
            for idx in range(rows_to_process):
                # Show progress for large files
                if idx > 0 and idx % progress_interval == 0:
                    progress_pct = (idx / rows_to_process) * 100
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = idx / elapsed if elapsed > 0 else 0
                    print(f"📊 Progress: {progress_pct:.1f}% ({idx}/{rows_to_process} rows, {len(facts_to_add)} facts, {rate:.1f} rows/sec)")
                
                row = df.iloc[idx]
                
                # Get primary entity (from first entity column or first column)
                primary_entity = None
                if entity_columns:
                    primary_entity = str(row[entity_columns[0]]).strip() if pd.notna(row[entity_columns[0]]) else None
                if not primary_entity and len(df.columns) > 0:
                    primary_entity = str(row[df.columns[0]]).strip() if pd.notna(row[df.columns[0]]) else None
                
                # Skip rows with invalid entities
                if not primary_entity or primary_entity.lower() in ['nan', 'none', 'null', '']:
                    continue
                
                # Clean entity name (remove special characters that might break URIs)
                primary_entity = primary_entity.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                primary_entity = ' '.join(primary_entity.split())  # Normalize whitespace
                
                if len(primary_entity) > 200:  # Skip entities that are too long
                    continue
                
                # Create facts for each column-value pair
                for col in df.columns:
                    # Skip the entity column itself to avoid redundant facts
                    if col in entity_columns and col == entity_columns[0]:
                        continue
                    
                    value = row[col]
                    if pd.isna(value):
                        continue
                    
                    value_str = str(value).strip()
                    # Skip empty or invalid values
                    if not value_str or value_str.lower() in ['nan', 'none', 'null', '']:
                        continue
                    
                    # Clean value (remove newlines, normalize whitespace)
                    value_str = value_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                    value_str = ' '.join(value_str.split())  # Normalize whitespace
                    
                    # Skip values that are too long (might break the graph)
                    if len(value_str) > 500:
                        value_str = value_str[:500] + "..."
                    
                    # Create predicate from column name
                    predicate = col.lower().replace('_', ' ').replace('-', ' ')
                    # Limit predicate length
                    if len(predicate) > 100:
                        predicate = predicate[:100]
                    
                    # Fact: "Entity has property with value"
                    if collect_fact(primary_entity, predicate, value_str):
                        facts_added += 1
                        row_facts += 1
                    else:
                        facts_skipped += 1
                
                # Step 4: Create relationships between columns in the same row
                # If we have multiple entity columns, create relationships
                if len(entity_columns) > 1:
                    for i in range(len(entity_columns) - 1):
                        entity1_val = row[entity_columns[i]]
                        entity2_val = row[entity_columns[i + 1]]
                        
                        if pd.isna(entity1_val) or pd.isna(entity2_val):
                            continue
                        
                        entity1 = str(entity1_val).strip()
                        entity2 = str(entity2_val).strip()
                        
                        # Skip invalid entities
                        if (not entity1 or not entity2 or 
                            entity1.lower() in ['nan', 'none', 'null', ''] or
                            entity2.lower() in ['nan', 'none', 'null', '']):
                            continue
                        
                        # Clean entity names
                        entity1 = entity1.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                        entity1 = ' '.join(entity1.split())
                        entity2 = entity2.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                        entity2 = ' '.join(entity2.split())
                        
                        if len(entity1) > 200 or len(entity2) > 200:
                            continue
                        
                        # Create relationship fact
                        relationship_pred = f"related_to_{entity_columns[i + 1].lower()}"
                        if len(relationship_pred) > 100:
                            relationship_pred = relationship_pred[:100]
                        
                        if collect_fact(entity1, relationship_pred, entity2):
                            facts_added += 1
                            relationship_facts += 1
                        else:
                            facts_skipped += 1
            
            # Step 5: Add summary facts about the CSV structure
            summary_facts = [
                (file_name, "has_total_rows", str(len(df))),
                (file_name, "has_total_columns", str(len(df.columns))),
                (file_name, "is_csv_file", "true"),
            ]
            
            for subject, predicate, obj in summary_facts:
                if collect_fact(subject, predicate, obj):
                    facts_added += 1
                else:
                    facts_skipped += 1
            
            # Batch add all collected facts to graph at once (much faster)
            if len(facts_to_add) > 0:
                print(f"📊 Collected {len(facts_to_add)} facts, now adding to graph...")
                batch_add_facts_to_graph()
            else:
                print("⚠️  No facts collected to add")
            
            # PERFORMANCE: Skip metadata operations entirely for CSV files
            # Adding metadata (source_doc, agent) requires iterating through graph for each fact
            # This is extremely slow for large CSVs. Facts are still added to the graph correctly.
            # If metadata is needed, it can be added later in a batch operation.
            
            # Save graph once at the end (only if we added facts)
            if facts_added > 0:
                from knowledge import save_knowledge_graph
                print(f"💾 Saving knowledge graph with {facts_added} new facts...")
                save_knowledge_graph()
                print(f"✅ CSV processing complete! Added {facts_added} facts, skipped {facts_skipped} duplicates")
            else:
                print(f"⚠️  No new facts added (all were duplicates or invalid)")
            
            return {
                'success': True,
                'facts_added': facts_added,
                'facts_skipped': facts_skipped,
                'column_facts': column_facts,
                'row_facts': row_facts,
                'relationship_facts': relationship_facts,
                'total_rows': len(df),
                'rows_processed': rows_to_process,
                'total_columns': len(df.columns),
                'entity_columns': entity_columns,
                'note': f'Processed {rows_to_process} of {len(df)} rows' if rows_to_process < len(df) else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'facts_added': 0,
                'facts_skipped': 0
            }

