"""
Document Text Extraction Module
================================

This module handles extracting raw text content from various file formats.
The extracted text is then passed to knowledge.py for fact extraction.

ARCHITECTURE:
This module now uses a multi-agent architecture where specialized agents
handle different file types:
- PDFAgent: Processes PDF files
- DOCXAgent: Processes DOCX/DOC files
- TXTAgent: Processes plain text files (TXT, MD, etc.)
- CSVAgent: Processes CSV files
- XLSXAgent: Processes Excel XLSX files

Supported Formats:
- PDF: Uses PyPDF2 (via PDFAgent)
- DOCX/DOC: Uses python-docx (via DOCXAgent)
- TXT/MD: Direct file read with encoding detection (via TXTAgent)
- CSV: Uses pandas (via CSVAgent)
- XLSX: Uses openpyxl (via XLSXAgent)

Flow:
1. User uploads file → api_server.py receives it
2. api_server.py calls handle_file_upload() → AgentCoordinator routes to appropriate agent
3. Agent extracts text → knowledge.add_to_graph() → extracts facts

Key Functions:
- handle_file_upload(file_paths): Main entry point - processes multiple files
- process_uploaded_file(file): Processes single file, returns text
- Legacy functions maintained for backward compatibility

Author: Research Brain Team
Last Updated: 2025-01-15
"""

import os
from datetime import datetime
from knowledge import add_to_graph

# Import the multi-agent system
from agents import AgentCoordinator

# ============================================================================
# GLOBAL STATE
# ============================================================================

last_extracted_text = ""  # Last extracted text (for debugging)
processed_files = []      # List of processed file names

# Initialize the agent coordinator (singleton pattern)
_coordinator = None

def get_coordinator():
    """Get or create the agent coordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = AgentCoordinator()
    return _coordinator

# ============================================================================
# LEGACY FUNCTIONS (maintained for backward compatibility)
# ============================================================================

def extract_text_from_pdf(file_path):
    """Legacy function - delegates to PDFAgent."""
    coordinator = get_coordinator()
    agent = coordinator.get_agent_for_file(file_path)
    if agent and agent.agent_name == "PDF Agent":
        try:
            return agent.extract_text(file_path)
        except Exception as e:
            return f"Error reading PDF: {e}"
    return f"Error: PDF agent not available"

def extract_text_from_docx(file_path):
    """Legacy function - delegates to DOCXAgent."""
    coordinator = get_coordinator()
    agent = coordinator.get_agent_for_file(file_path)
    if agent and agent.agent_name == "DOCX Agent":
        try:
            return agent.extract_text(file_path)
        except Exception as e:
            return f"Error reading DOCX: {e}"
    return f"Error: DOCX agent not available"

def extract_text_from_txt(file_path):
    """Legacy function - delegates to TXTAgent."""
    coordinator = get_coordinator()
    agent = coordinator.get_agent_for_file(file_path)
    if agent and agent.agent_name == "TXT Agent":
        try:
            return agent.extract_text(file_path)
        except Exception as e:
            return f"Error reading TXT: {e}"
    return f"Error: TXT agent not available"

def extract_text_from_csv(file_path):
    """Legacy function - delegates to CSVAgent."""
    coordinator = get_coordinator()
    agent = coordinator.get_agent_for_file(file_path)
    if agent and agent.agent_name == "CSV Agent":
        try:
            return agent.extract_text(file_path)
        except Exception as e:
            return f"Error reading CSV: {e}"
    return f"Error: CSV agent not available"

# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def update_extracted_text(text):
    """Update the global last extracted text."""
    global last_extracted_text
    last_extracted_text = text

def show_extracted_text():
    """Show the last extracted text (for debugging)."""
    global last_extracted_text
    if not last_extracted_text:
        return " No file has been processed yet.\n\nUpload a file and process it to see the extracted text here."
    preview = last_extracted_text[:1000]
    if len(last_extracted_text) > 1000:
        preview += "\n\n... (truncated, showing first 1000 characters)"
    return f" **Last Extracted Text:**\n\n{preview}"

def process_uploaded_file(file, original_filename=None):
    """
    Process a single file using the multi-agent system.
    
    Args:
        file: File path (string) or file object
        original_filename: Original filename if different from file path
        
    Returns:
        Processing result message
    """
    if file is None:
        return "No file uploaded."
    
    # Handle both string paths and file objects
    if isinstance(file, str):
        file_path = file
    else:
        file_path = file.name if hasattr(file, 'name') else str(file)
    
    # Get the coordinator and process the file
    coordinator = get_coordinator()
    
    # Check if file type is supported
    if not coordinator.can_process(file_path):
        file_extension = os.path.splitext(file_path)[1].lower()
        supported = ', '.join(sorted(coordinator.get_supported_extensions()))
        return f" Unsupported file type: {file_extension}\n\nSupported formats: {supported}"
    
    # Process file with appropriate agent
    result = coordinator.process_file(file_path, original_filename)
    
    # Check if processing succeeded
    if not result['success']:
        error_msg = result.get('error', 'Unknown error')
        return f" Error processing file: {error_msg}"
    
    # Extract text and metadata
    extracted_text = result['text']
    metadata = result['metadata']
    agent_name = result.get('agent', 'Unknown Agent')
    
    # Get the agent instance for enhanced processing
    agent = coordinator.get_agent_for_file(file_path)
    
    # Update global state
    update_extracted_text(extracted_text)
    
    # Prepare display information
    display_name = original_filename if original_filename else os.path.basename(file_path)
    file_extension = os.path.splitext(file_path)[1].lower()
    preview = extracted_text[:300] + "..." if len(extracted_text) > 300 else extracted_text
    
    # Debug: Check if text was extracted
    if not extracted_text or len(extracted_text.strip()) < 10:
        print(f"⚠️  WARNING: Extracted text is too short or empty for {display_name}")
        print(f"   Text length: {len(extracted_text) if extracted_text else 0}")
        print(f"   Agent: {agent_name}")
        return f"⚠️  Warning: Could not extract meaningful text from {display_name}. File may be empty or in an unsupported format."
    
    # Add to knowledge graph (CSV uses same path as other files: extract text -> add_to_graph)
    filename = original_filename if original_filename else os.path.basename(file_path)
    timestamp = datetime.now().isoformat()
    csv_result_data = None
    kg_result = add_to_graph(extracted_text, source_document=filename, uploaded_at=timestamp, agent_name=agent_name)
    
    # Build result message with agent information
    file_size = len(extracted_text)
    result_msg = f" Successfully processed {display_name}!\n\n"
    result_msg += f"📊 File stats:\n"
    result_msg += f"• Size: {file_size:,} characters\n"
    result_msg += f"• Type: {file_extension.upper()}\n"
    result_msg += f"• Agent: {agent_name}\n"
    
    # Add format-specific metadata if available
    if 'pages' in metadata:
        result_msg += f"• Pages: {metadata['pages']}\n"
    if 'rows' in metadata:
        result_msg += f"• Rows: {metadata['rows']}\n"
    if 'columns' in metadata:
        result_msg += f"• Columns: {metadata['columns']}\n"
    if 'sheets' in metadata:
        result_msg += f"• Sheets: {len(metadata['sheets'])}\n"
    
    result_msg += f"\n Text preview:\n{preview}\n\n{kg_result}"
    
    # Return structured result for API response
    result_data = {
        'message': result_msg,
        'agent_name': agent_name,
        'metadata': metadata,
    }
    
    # Add CSV-specific stats if available
    if agent_name == "CSV Agent" and 'csv_result_data' in locals() and csv_result_data:
        result_data['csv_stats'] = {
            'column_facts': csv_result_data.get('column_facts', 0),
            'row_facts': csv_result_data.get('row_facts', 0),
            'relationship_facts': csv_result_data.get('relationship_facts', 0),
            'facts_added': csv_result_data.get('facts_added', 0),
            'entity_columns': csv_result_data.get('entity_columns', [])
        }
        result_data['rows_processed'] = csv_result_data.get('rows_processed', csv_result_data.get('total_rows', 0))
        result_data['total_columns'] = csv_result_data.get('total_columns', 0)
        result_data['facts_added'] = csv_result_data.get('facts_added', 0)
    
    return result_data

def handle_file_upload(files, original_filenames=None):
    """
    Process multiple files using the multi-agent system.
    
    Args:
        files: List of file paths (strings) or file objects
        original_filenames: Optional dict mapping file paths to original filenames
                          {file_path: original_filename}
    
    Returns:
        Summary message with processing results
    """
    global processed_files
    if not files or len(files) == 0:
        return "Please select at least one file to process."
    
    # Create mapping if not provided
    if original_filenames is None:
        original_filenames = {}
    
    # Get coordinator
    coordinator = get_coordinator()
    
    results = []
    new_processed = []
    # Only skip duplicates within this batch (same request). Re-uploading the same
    # filename in a new request should always be processed (e.g. after reset or to refresh).
    seen_in_this_batch = set()
    
    for file in files:
        if file is None:
            continue
        
        try:
            # Determine file path and name
            if isinstance(file, str):
                file_path = file
                file_name = original_filenames.get(file_path, os.path.basename(file))
            else:
                file_path = file.name if hasattr(file, 'name') else str(file)
                file_name = original_filenames.get(file_path, os.path.basename(file_path))
            
            # Skip only if same filename appears twice in this upload (duplicate in batch)
            if file_name in seen_in_this_batch:
                results.append(f"SKIP: {file_name} - Duplicate in this upload, skipping")
                continue
            seen_in_this_batch.add(file_name)
            
            # Process file
            result = process_uploaded_file(file, original_filename=file_name)
            
            # Check if processing was successful
            # result can be a string or dict (if enhanced processing returns structured data)
            if isinstance(result, dict):
                result_message = result.get('message', '')
                if result_message.startswith(" Successfully"):
                    results.append(f"SUCCESS: {file_name} - {result_message}")
                    file_result = {
                        'name': file_name,
                        'filename': file_name,
                        'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                        'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'agent': result.get('agent_name'),
                        'agent_name': result.get('agent_name'),
                        'metadata': result.get('metadata', {}),
                    }
                    # Add CSV-specific stats if available
                    if result.get('csv_stats'):
                        file_result['csv_stats'] = result['csv_stats']
                        file_result['rows_processed'] = result.get('rows_processed')
                        file_result['total_columns'] = result.get('total_columns')
                        file_result['facts_added'] = result.get('facts_added')
                    new_processed.append(file_result)
                else:
                    results.append(f"ERROR: {file_name} - {result_message}")
            elif isinstance(result, str) and result.startswith(" Successfully"):
                results.append(f"SUCCESS: {file_name} - {result}")
                new_processed.append({
                    'name': file_name,
                    'filename': file_name,
                    'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                    'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                })
            else:
                error_msg = result.get('message', str(result)) if isinstance(result, dict) else result
                results.append(f"ERROR: {file_name} - {error_msg}")
                
        except Exception as e:
            file_name = original_filenames.get(file_path, os.path.basename(file)) if isinstance(file, str) else original_filenames.get(file_path, os.path.basename(file.name)) if hasattr(file, 'name') else str(file)
            results.append(f"ERROR: {file_name} - Error: {e}")
    
    # Update processed files list
    processed_files.extend(new_processed)
    
    # Build summary
    total_files = len(files)
    successful = len([r for r in results if r.startswith("SUCCESS")])
    skipped = len([r for r in results if r.startswith("SKIP")])
    failed = len([r for r in results if r.startswith("ERROR")])
    
    summary = f"**Upload Summary:**\n"
    summary += f"• Total files: {total_files}\n"
    summary += f"• Successfully processed: {successful}\n"
    summary += f"• Skipped (duplicates in batch): {skipped}\n"
    summary += f"• Failed: {failed}\n\n"
    
    # Add supported formats info
    supported_formats = ', '.join(sorted(coordinator.get_supported_extensions()))
    summary += f"**Supported formats:** {supported_formats}\n\n"
    
    summary += "**File Results:**\n"
    for result in results:
        summary += f"{result}\n"
    
    # Return both summary string and structured data for API
    return {
        'summary': summary,
        'file_results': new_processed,
        'total_files': total_files,
        'successful': successful,
        'skipped': skipped,
        'failed': failed
    }

def show_processed_files():
    """Show list of processed files."""
    global processed_files
    if not processed_files:
        coordinator = get_coordinator()
        supported = ', '.join(sorted(coordinator.get_supported_extensions()))
        return f"**No files processed yet.**\n\n**Start building your knowledge base:**\n1. Select one or more files ({supported})\n2. Click 'Process Files' to extract knowledge\n3. View your processed files here\n4. Upload more files to expand your knowledge base!"
    
    result = f"**Processed Files ({len(processed_files)}):**\n\n"
    for i, file_info in enumerate(processed_files, 1):
        result += f"**{i}. {file_info['name']}**\n"
        result += f"   • Size: {file_info['size']:,} bytes\n"
        result += f"   • Processed: {file_info['processed_at']}\n\n"
    return result

def clear_processed_files():
    """Clear the processed files list."""
    global processed_files
    processed_files = []
    return "Processed files list cleared. You can now re-upload previously processed files."

def simple_test():
    """Test function."""
    return " Event handler is working! Button clicked successfully!"

def get_supported_formats():
    """
    Get list of supported file formats.
    
    Returns:
        List of supported file extensions
    """
    coordinator = get_coordinator()
    return coordinator.get_supported_extensions()

def get_agent_info():
    """
    Get information about all registered agents.
    
    Returns:
        Dictionary mapping agent names to their supported extensions
    """
    coordinator = get_coordinator()
    return coordinator.get_agent_info()
