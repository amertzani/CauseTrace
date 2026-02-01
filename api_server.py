"""
FastAPI Backend Server - Research Brain
========================================

This is the MAIN BACKEND ENTRY POINT. It provides REST API endpoints
for the React frontend to interact with the knowledge management system.

Architecture:
- Receives HTTP requests from frontend (React app)
- Processes requests using knowledge.py, file_processing.py, documents_store.py
- Returns JSON responses

Key Endpoints:
- POST /api/knowledge/upload: Upload and process documents
- POST /api/knowledge/facts: Create a new fact
- GET /api/knowledge/facts: Get all facts
- DELETE /api/knowledge/facts/{id}: Delete a fact
- GET /api/documents: Get all uploaded documents
- GET /api/export: Export all knowledge as JSON
- POST /api/knowledge/import: Import knowledge from JSON

Connection:
- Frontend connects to: http://localhost:8001 (default)
- API docs available at: http://localhost:8001/docs

To run:
    python api_server.py
    
Or use the convenience scripts:
    start_api.bat (Windows)
    launch_app.sh (macOS/Linux) or start_backend_simple.sh for backend only
    
The scripts automatically activate the virtual environment if available.

Author: Research Brain Team
Last Updated: 2025-01-15
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import tempfile
import json
import re
import asyncio
from datetime import datetime

# Import your existing modules
from responses import respond as rqa_respond
from knowledge import (
    add_to_graph as kb_add_to_graph,
    show_graph_contents as kb_show_graph_contents,
    visualize_knowledge_graph as kb_visualize_knowledge_graph,
    save_knowledge_graph as kb_save_knowledge_graph,
    load_knowledge_graph as kb_load_knowledge_graph,
    delete_all_knowledge as kb_delete_all_knowledge,
    graph as kb_graph,
    import_knowledge_from_json_file as kb_import_json
)
from file_processing import handle_file_upload as fp_handle_file_upload
from documents_store import add_document, get_all_documents, delete_document as ds_delete_document, cleanup_documents_without_facts, delete_all_documents as ds_delete_all_documents, save_documents as ds_save_documents, DOCUMENTS_FILE as DOCUMENTS_FILE_PATH
from knowledge import create_comprehensive_backup as kb_create_comprehensive_backup, KNOWLEDGE_FILE as KG_FILE, BACKUP_FILE as KG_BACKUP_FILE

from contextlib import asynccontextmanager

# Load knowledge graph on startup using lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: clear cache and uploaded sources on every restart (clean slate)
    print("Initializing server (clearing cache and uploaded sources on restart)...")
    try:
        # 1. Remove persistence files so no old data can be loaded (bulletproof)
        from causal_graph import CAUSAL_STORE_PATH
        for path, name in [
            (KG_FILE, "knowledge_graph.pkl"),
            (KG_BACKUP_FILE, "knowledge_backup.json"),
            (DOCUMENTS_FILE_PATH, "documents_store.json"),
            (CAUSAL_STORE_PATH, "causal_graphs_store.json"),
        ]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"🧹 Removed {name}")
                except Exception as e:
                    print(f"⚠️  Could not remove {path}: {e}")

        # 2. Clear in-memory knowledge graph and write empty file (so graph is empty and file exists)
        kb_delete_all_knowledge()

        # 3. Write empty documents store (file was removed above)
        ds_save_documents([])

        # 4. Recreate empty data-driven causal graph store (file was removed above)
        try:
            from causal_graph import clear_data_driven_causal_graphs
            clear_data_driven_causal_graphs()
            print("🧹 Cleared data-driven causal graph store")
        except Exception as e:
            print(f"⚠️  Could not clear causal graph store: {e}")
            import traceback
            traceback.print_exc()

        print("✅ Server started with clean state (no persisted documents or facts)")

        # Pre-load LLM model in background to avoid timeout on first request
        print("🔄 Pre-loading LLM model for research assistant (this may take 1-2 minutes)...")
        import asyncio
        from responses import load_llm_model
        
        # Start pre-loading in background (don't block startup)
        def preload_llm_sync():
            try:
                result = load_llm_model()
                if result:
                    print("✅ LLM model pre-loaded successfully")
                else:
                    print("⚠️  LLM model not available, will use rule-based responses")
            except Exception as e:
                print(f"⚠️  Failed to pre-load LLM: {e}")
                print("   Will use rule-based responses")
        
        # Run in background thread (non-blocking)
        import threading
        preload_thread = threading.Thread(target=preload_llm_sync, daemon=True)
        preload_thread.start()
        print("   (Model loading in background, server is ready)")
        
    except Exception as e:
        print(f"⚠️  Warning during knowledge graph initialization: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with empty graph...")
    yield
    # Shutdown (if needed)
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(title="NesyX API", description="Backend API for NesyX Knowledge Graph System", lifespan=lifespan)

# Configure CORS - Allow all origins (you can restrict this to specific domains later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Request/Response Models
# ==========================================================

class ChatMessage(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []

class AddKnowledgeRequest(BaseModel):
    text: str

class AddFactRequest(BaseModel):
    subject: str
    predicate: str
    object: str
    source: Optional[str] = "manual"
    details: Optional[str] = None

class DeleteKnowledgeRequest(BaseModel):
    keyword: Optional[str] = None
    count: Optional[int] = None


class DoWhyEffectRequest(BaseModel):
    document_name: str
    treatment: str
    outcome: str
    method: Optional[str] = "backdoor.linear_regression"

# ==========================================================
# API Endpoints
# ==========================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "NesyX API",
        "facts_count": len(kb_graph)
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "facts": len(kb_graph)}

@app.post("/api/chat")
async def chat_endpoint(request: ChatMessage):
    """Chat endpoint - ask questions about the knowledge base"""
    import asyncio
    try:
        # Run the response generation in a thread pool to avoid blocking
        # and set a timeout to prevent hanging
        loop = asyncio.get_event_loop()
        
        # Check if LLM is still loading and wait a bit if needed
        from responses import LLM_PIPELINE, load_llm_model, USE_LLM, LLM_AVAILABLE
        if USE_LLM and LLM_AVAILABLE and LLM_PIPELINE is None:
            # Model not loaded yet, try to load it (with timeout)
            print("⏳ LLM not loaded yet, loading now...")
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, load_llm_model),
                    timeout=90.0  # Give 90 seconds for model loading
                )
            except asyncio.TimeoutError:
                print("⚠️  LLM loading timed out, using rule-based responses")
        
        # Generate response with timeout
        # Adjust timeout based on device (CPU is much slower)
        from responses import LLM_DEVICE
        if LLM_DEVICE == "cpu":
            timeout_seconds = 20.0  # Shorter timeout for CPU
        else:
            timeout_seconds = 45.0  # Longer timeout for GPU
        
        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(None, rqa_respond, request.message, request.history),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            print(f"⏱️  Response generation timed out after {timeout_seconds}s on {LLM_DEVICE}")
            # Return a helpful message instead of raising an error
            response = "I'm sorry, the response is taking too long. This might be because the LLM is running on CPU, which is slower. Please try a simpler question or wait a moment and try again."
        return {
            "response": response,
            "status": "success"
        }
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, 
            detail="Request timed out. The LLM is taking too long to respond. Try disabling LLM with USE_LLM=false or ask a simpler question."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/api/knowledge/add")
async def add_knowledge_endpoint(request: AddKnowledgeRequest):
    """Add knowledge to the graph from text"""
    try:
        # Get current fact count before adding
        fact_count_before = len(kb_graph)
        
        # Add knowledge to graph
        result = kb_add_to_graph(request.text)
        # add_to_graph already saves, but ensure it's saved
        kb_save_knowledge_graph()
        
        # Verify save worked
        if os.path.exists(KG_FILE):
            file_size = os.path.getsize(KG_FILE)
            print(f"✅ Knowledge saved - file size: {file_size} bytes, facts in graph: {len(kb_graph)}")
        
        # Extract extraction method from result message
        extraction_method = "regex"
        if "TRIPLEX" in result.upper():
            extraction_method = "triplex"
        elif "FALLBACK" in result.upper():
            extraction_method = "regex (triplex fallback)"
        
        # Get newly added facts (those added after the operation)
        # Extract the last fact added (most recent)
        facts_list = []
        for i, (s, p, o) in enumerate(kb_graph):
            subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
            predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
            object_val = str(o)
            facts_list.append({
                "id": i + 1,
                "subject": subject,
                "predicate": predicate,
                "object": object_val,
                "source": "manual"
            })
        
        # Return the last fact added (most recent)
        new_fact = facts_list[-1] if facts_list else None
        
        return {
            "message": result,
            "status": "success",
            "total_facts": len(kb_graph),
            "fact": new_fact,  # Return the created fact for frontend
            "extraction_method": extraction_method  # Indicate which method was used
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding knowledge: {str(e)}")

@app.post("/api/knowledge/facts")
async def create_fact_endpoint(request: AddFactRequest):
    """Create a structured fact directly (subject, predicate, object)"""
    try:
        import rdflib
        from urllib.parse import quote
        from knowledge import fact_exists as kb_fact_exists
        
        # Check if fact already exists
        if kb_fact_exists(request.subject, request.predicate, str(request.object)):
            print(f"⚠️  POST /api/knowledge/facts: Duplicate fact detected - {request.subject} {request.predicate} {request.object}")
            return {
                "message": "Fact already exists in knowledge graph",
                "status": "duplicate",
                "fact": {
                    "subject": request.subject,
                    "predicate": request.predicate,
                    "object": str(request.object)
                },
                "total_facts": len(kb_graph)
            }
        
        # For structured facts, add directly to graph with proper URI encoding
        # Replace spaces with underscores in URIs to avoid RDFLib warnings
        subject_clean = request.subject.strip().replace(' ', '_')
        predicate_clean = request.predicate.strip().replace(' ', '_')
        object_value = str(request.object).strip()
        
        # Create URIs (encode spaces to avoid RDFLib warnings)
        subject_uri = rdflib.URIRef(f"urn:{quote(subject_clean, safe='')}")
        predicate_uri = rdflib.URIRef(f"urn:{quote(predicate_clean, safe='')}")
        object_literal = rdflib.Literal(object_value)
        
        # Add directly to graph
        kb_graph.add((subject_uri, predicate_uri, object_literal))
        
        # Add details if provided
        if request.details and request.details.strip():
            from knowledge import add_fact_details as kb_add_fact_details
            kb_add_fact_details(request.subject, request.predicate, object_value, request.details)
        
        # Add source document and timestamp (manual for directly created facts)
        from datetime import datetime
        from knowledge import add_fact_source_document as kb_add_fact_source_document
        timestamp = datetime.now().isoformat()
        kb_add_fact_source_document(request.subject, request.predicate, object_value, "manual", timestamp)
        
        # Save to disk
        save_result = kb_save_knowledge_graph()
        
        # Verify the fact was added
        fact_count = len(kb_graph)
        print(f"✅ POST /api/knowledge/facts: Added fact - {request.subject} {request.predicate} {request.object}")
        if request.details:
            print(f"✅ Added details: {request.details[:50]}...")
        print(f"✅ Save result: {save_result}")
        print(f"✅ Total facts in graph: {fact_count}")
        
        # Verify file was written
        if os.path.exists(KG_FILE):
            file_size = os.path.getsize(KG_FILE)
            print(f"✅ Knowledge file size: {file_size} bytes")
        
        # Get details for the response
        from knowledge import get_fact_details as kb_get_fact_details
        details = kb_get_fact_details(request.subject, request.predicate, object_value)
        
        # Create the fact object - use the actual index in the graph
        new_fact = {
            "id": str(fact_count),  # Use current count as ID (string format)
            "subject": request.subject,  # Return original subject (with spaces)
            "predicate": request.predicate,  # Return original predicate (with spaces)
            "object": object_value,  # Return original object
            "source": request.source,
            "details": details if details else None
        }
        
        return {
            "message": f"✅ Added fact successfully. Total facts: {fact_count}",
            "status": "success",
            "total_facts": fact_count,
            "fact": new_fact
        }
    except Exception as e:
        print(f"❌ Error creating fact: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error creating fact: {str(e)}")

@app.get("/api/knowledge/triplex-status")
async def triplex_status_endpoint():
    """Get Triplex model status and availability"""
    try:
        from knowledge import TRIPLEX_AVAILABLE, USE_TRIPLEX, TRIPLEX_MODEL, TRIPLEX_DEVICE
        
        status = {
            "available": TRIPLEX_AVAILABLE,
            "enabled": USE_TRIPLEX,
            "loaded": TRIPLEX_MODEL is not None,
            "device": TRIPLEX_DEVICE if TRIPLEX_AVAILABLE else "N/A"
        }
        
        if TRIPLEX_AVAILABLE and USE_TRIPLEX:
            status["message"] = "Triplex is available and enabled. LLM extraction will be used."
        elif TRIPLEX_AVAILABLE and not USE_TRIPLEX:
            status["message"] = "Triplex is available but disabled. Set USE_TRIPLEX=true to enable."
        else:
            status["message"] = "Triplex is not available. Using regex-based extraction."
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting Triplex status: {str(e)}")

@app.post("/api/knowledge/upload")
async def upload_file_endpoint(files: List[UploadFile] = File(..., description="Files to upload (use form field name 'files')")):
    """Upload and process files (PDF, DOCX, TXT, CSV)"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided. Send multipart form with field 'files'.")
    tmp_paths = []  # Initialize outside try block so finally can access it
    try:
        facts_before = len(kb_graph)
        file_info_list = []
        
        # Map temporary file paths to original filenames
        temp_to_original = {}
        
        for file in files:
            # Save uploaded file temporarily
            suffix = os.path.splitext(file.filename)[1] if file.filename else ""
            original_filename = file.filename or 'unknown'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
                tmp_paths.append(tmp_path)
                temp_to_original[tmp_path] = original_filename
                file_info_list.append({
                    'name': original_filename,
                    'size': len(content),
                    'type': suffix.lstrip('.') or 'unknown'
                })
        
        try:
            # Process all files in a thread so the event loop stays responsive
            # (handle_file_upload / add_to_graph can be slow for large files)
            upload_result = await asyncio.to_thread(
                fp_handle_file_upload, tmp_paths, original_filenames=temp_to_original
            )
            
            # Handle both string (legacy) and dict (new) return formats
            if isinstance(upload_result, dict):
                result = upload_result.get('summary', '')
                file_results = upload_result.get('file_results', [])
            else:
                result = upload_result
                file_results = []
            
            # Data-driven causal discovery for CSV files (causal-learn)
            for tmp_path in tmp_paths:
                orig = temp_to_original.get(tmp_path, '')
                if orig.lower().endswith('.csv') and os.path.exists(tmp_path):
                    try:
                        from causal_graph import (
                            discover_causal_structure_from_csv,
                            save_data_driven_causal_graph,
                        )
                        graph_data = discover_causal_structure_from_csv(tmp_path)
                        if "error" not in graph_data:
                            save_data_driven_causal_graph(orig, graph_data)
                            print(f"✅ Data-driven causal graph saved for {orig} (select it in Causal Graph → Source)")
                        else:
                            print(f"⚠️  Causal discovery for {orig}: {graph_data.get('error')}")
                    except Exception as e:
                        err = str(e)
                        print(f"⚠️  Causal discovery failed for {orig}: {err}")
                        if "causallearn" in err.lower() or "causal-learn" in err.lower():
                            print("   → Install with: pip install causal-learn")
            
            # IMPORTANT: Ensure graph is saved to disk
            # add_to_graph already saves, but let's make sure it's persisted
            kb_save_knowledge_graph()
            
            # CRITICAL: Reload from disk to get the actual saved facts
            # The in-memory graph might be out of sync if there were multiple processes
            # or if the graph was cleared on startup
            kb_load_knowledge_graph()
            
            facts_after = len(kb_graph)
            facts_extracted = facts_after - facts_before
            
            
            # CRITICAL: If facts_extracted is 0 but we processed files, check the result message
            # The result message from add_to_graph contains the actual count
            if facts_extracted == 0 and result:
                # Try to extract the actual count from the result message
                import re
                total_match = re.search(r'Total facts stored: (\d+)', result)
                if total_match:
                    total_facts = int(total_match.group(1))
                    facts_extracted = max(0, total_facts - facts_before)
                    print(f"⚠️  Adjusted facts_extracted from result message: {facts_extracted}")
            
            # Parse result message to extract added/skipped counts and extraction method
            import re
            added_match = re.search(r'Added (\d+) new triples', result)
            skipped_match = re.search(r'skipped (\d+) duplicates', result)
            added_count = int(added_match.group(1)) if added_match else facts_extracted
            skipped_count = int(skipped_match.group(1)) if skipped_match else 0
            
            # Extract extraction method from result (result is always str here)
            extraction_method = "regex"
            result_upper = (result or "").upper()
            if "TRIPLEX" in result_upper:
                extraction_method = "triplex"
            elif "FALLBACK" in result_upper:
                extraction_method = "regex (triplex fallback)"
            
            print(f"✅ Upload processed {len(files)} file(s)")
            print(f"   Extraction method: {extraction_method}")
            print(f"   Facts before: {facts_before}, after: {facts_after}, extracted: {facts_extracted}")
            print(f"   Added: {added_count}, Skipped duplicates: {skipped_count}")
            print(f"   Graph now has {len(kb_graph)} total facts")
            
            # Verify facts are actually in the graph
            if len(kb_graph) > 0:
                sample_fact = list(kb_graph)[0]
                print(f"   Sample fact from graph: {sample_fact}")
            else:
                print("   ⚠️  WARNING: Graph is empty after processing!")
            
            # Save document metadata - Count facts per document by checking source
            # This ensures each document gets the correct fact count
            processed_docs = []
            
            # Count facts per document using helper function
            from documents_store import count_facts_for_document
            
            # Always try to save documents, even if total facts_extracted is 0
            # because individual files might have facts
            for file_info in file_info_list:
                document_name = file_info['name']
                
                # Count facts that have this document as source
                facts_for_this_doc = count_facts_for_document(document_name)
                
                # Save document if it has facts
                if facts_for_this_doc > 0:
                    doc = add_document(
                        name=document_name,
                        size=file_info['size'],
                        file_type=file_info['type'],
                        facts_extracted=facts_for_this_doc
                    )
                    if doc:
                        processed_docs.append(doc)
                        print(f"✅ Saved document {document_name} with {facts_for_this_doc} facts")
                else:
                    print(f"⚠️  Document {document_name} has 0 facts - not saved")
            
            # If no documents were saved and no facts were extracted overall, clean up
            if len(processed_docs) == 0 and facts_extracted == 0:
                # No facts extracted from any file - PERMANENTLY REMOVE any existing documents with these names
                print(f"⚠️  No facts extracted from {len(files)} file(s) - REMOVING documents")
                for file_info in file_info_list:
                    from documents_store import load_documents, save_documents
                    docs = load_documents()
                    original_count = len(docs)
                    # PERMANENTLY REMOVE documents with this name
                    docs = [d for d in docs if d.get('name') != file_info['name']]
                    removed_count = original_count - len(docs)
                    if removed_count > 0:
                        save_documents(docs)
                        print(f"   🗑️  PERMANENTLY removed {file_info['name']} (no facts extracted)")
            
            # Final save to ensure everything is persisted to disk
            kb_save_knowledge_graph()
            
            # CRITICAL: Reload one more time to ensure in-memory graph matches disk
            # This is necessary because the graph might have been cleared on startup
            kb_load_knowledge_graph()
            
            # Verify final state
            final_fact_count = len(kb_graph)
            if os.path.exists(KG_FILE):
                file_size = os.path.getsize(KG_FILE)
                print(f"✅ Final save - file size: {file_size} bytes, facts in graph: {final_fact_count}")
                
            # Update total_facts in response to reflect actual graph state
            if final_fact_count > 0:
                print(f"✅ Upload complete: Graph now has {final_fact_count} facts in memory")
            
            print(f"✅ Upload processed {len(files)} file(s), added {added_count} new facts, skipped {skipped_count} duplicates")
            
            # Get final fact count after all saves and reloads
            final_total = len(kb_graph)
            
            return {
                "message": result,
                "files_processed": len(files),
                "status": "success",
                "total_facts": final_total,  # Use final count after reload
                "facts_extracted": added_count,  # Use actual added count
                "facts_skipped": skipped_count,   # Add skipped duplicates count
                "extraction_method": extraction_method,  # Indicate which method was used
                "documents": processed_docs,
                "file_results": file_results  # Include detailed file results with CSV stats
            }
        finally:
            # Clean up temporary files
            if tmp_paths:  # Only clean up if tmp_paths was initialized
                for tmp_path in tmp_paths:
                    if os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except Exception as cleanup_error:
                            print(f"⚠️  Warning: Could not delete temp file {tmp_path}: {cleanup_error}")
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error uploading files: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error uploading files: {error_msg}")

@app.post("/api/process")
async def process_documents_endpoint(request: Dict[str, Any]):
    """Process already uploaded documents by their IDs
    
    Note: This endpoint is for re-processing existing documents.
    For new uploads, use /api/knowledge/upload which uploads and processes in one step.
    """
    try:
        document_ids = request.get("document_ids", [])
        if not document_ids:
            raise HTTPException(status_code=400, detail="No document IDs provided")
        
        # Get documents from store
        from documents_store import get_all_documents, get_document_by_id
        all_docs = get_all_documents()
        
        # Find documents to process
        docs_to_process = []
        for doc_id in document_ids:
            doc = get_document_by_id(doc_id) if hasattr(get_all_documents, '__call__') else None
            if not doc:
                # Try to find by ID in all docs
                doc = next((d for d in all_docs if d.get('id') == doc_id), None)
            if doc:
                docs_to_process.append(doc)
        
        if not docs_to_process:
            return {
                "message": "No documents found with provided IDs",
                "status": "error",
                "documents_processed": 0
            }
        
        # Note: Re-processing requires the original files, which may not be available
        # This endpoint mainly exists for API compatibility
        # In practice, users should re-upload files to process them
        return {
            "message": f"Found {len(docs_to_process)} document(s). Re-processing requires re-uploading the files. Please use /api/knowledge/upload to upload and process files.",
            "status": "info",
            "documents_found": len(docs_to_process),
            "documents_processed": 0
        }
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error processing documents: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing documents: {error_msg}")

@app.get("/api/knowledge/graph")
async def get_graph_endpoint():
    """Get knowledge graph visualization"""
    try:
        graph_html = kb_visualize_knowledge_graph()
        return {
            "graph_html": graph_html,
            "total_facts": len(kb_graph),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting graph: {str(e)}")

@app.get("/api/knowledge/causal-graph")
async def get_causal_graph_endpoint(
    include_inferred: bool = Query(True, description="Include inferred causal relationships"),
    source: Optional[str] = Query("kb", description="Source: 'kb' (knowledge base) or 'dataset'"),
    document_name: Optional[str] = Query(None, description="Document name for source=dataset"),
):
    """Get causal graph: from knowledge base (default) or data-driven from a CSV dataset"""
    try:
        if source == "dataset" and document_name:
            from causal_graph import get_data_driven_causal_graph
            causal_data = get_data_driven_causal_graph(document_name)
            if not causal_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data-driven causal graph found for document: {document_name}",
                )
            return {
                "nodes": causal_data.get("nodes", []),
                "edges": causal_data.get("edges", []),
                "stats": causal_data.get("stats", {}),
                "status": "success",
                "source": "dataset",
                "document_name": document_name,
            }
        # Default: from knowledge base
        from causal_graph import get_causal_graph_data
        print(f"📊 Causal graph requested (include_inferred={include_inferred})")
        causal_data = get_causal_graph_data(include_inferred=include_inferred)
        print(f"✅ Causal graph generated: {len(causal_data['nodes'])} nodes, {len(causal_data['edges'])} edges")
        return {
            "nodes": causal_data["nodes"],
            "edges": causal_data["edges"],
            "stats": causal_data["stats"],
            "status": "success",
            "source": "kb",
        }
    except HTTPException:
        raise
    except ImportError as e:
        print(f"❌ Import error in causal graph: {e}")
        raise HTTPException(status_code=500, detail=f"Error importing causal graph module: {str(e)}")
    except Exception as e:
        print(f"❌ Error generating causal graph: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating causal graph: {str(e)}")


@app.get("/api/knowledge/causal-graph/sources")
async def get_causal_graph_sources_endpoint():
    """List available causal graph sources: KB + document names with data-driven graphs"""
    try:
        from causal_graph import list_data_driven_causal_graph_sources
        data_driven = list_data_driven_causal_graph_sources()
        sources = [
            {"id": "kb", "label": "Causal graph from knowledge base", "type": "kb"},
            *[{"id": name, "label": f"Dataset: {name}", "type": "dataset"} for name in data_driven],
        ]
        print(f"✅ Causal graph sources: {len(sources)} total ({len(data_driven)} dataset(s))")
        return {"sources": sources, "status": "success"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️  Causal graph sources failed: {e} — returning KB only")
        return {
            "sources": [{"id": "kb", "label": "Causal graph from knowledge base", "type": "kb"}],
            "status": "success",
        }


@app.get("/api/knowledge/causal-graph/export")
async def export_causal_graph_endpoint(
    source: str = Query("all", description="Export: 'kb' (knowledge base only), 'all' (KB + all datasets), or 'dataset' (single dataset)"),
    document_name: Optional[str] = Query(None, description="Required when source=dataset: document name with data-driven graph"),
    include_inferred: bool = Query(True, description="Include inferred causal relationships (for KB)"),
):
    """Export causal graph(s) as JSON: KB only, all sources (KB + datasets), or one dataset."""
    try:
        from datetime import datetime
        from causal_graph import (
            get_causal_graph_data,
            list_data_driven_causal_graph_sources,
            get_data_driven_causal_graph,
        )
        exported_at = datetime.now().isoformat()

        if source == "kb":
            causal_data = get_causal_graph_data(include_inferred=include_inferred)
            export_data = {
                "metadata": {
                    "version": "1.0",
                    "exported_at": exported_at,
                    "source": "kb",
                    "include_inferred": include_inferred,
                },
                "nodes": causal_data.get("nodes", []),
                "edges": causal_data.get("edges", []),
                "stats": causal_data.get("stats", {}),
            }
            print(f"✅ GET /api/knowledge/causal-graph/export: KB — {len(export_data['nodes'])} nodes, {len(export_data['edges'])} edges")
            return export_data

        if source == "dataset":
            if not document_name:
                raise HTTPException(status_code=400, detail="document_name required when source=dataset")
            graph_data = get_data_driven_causal_graph(document_name)
            if not graph_data:
                raise HTTPException(status_code=404, detail=f"No data-driven graph found for: {document_name}")
            export_data = {
                "metadata": {
                    "version": "1.0",
                    "exported_at": exported_at,
                    "source": "dataset",
                    "document_name": document_name,
                },
                "nodes": graph_data.get("nodes", []),
                "edges": graph_data.get("edges", []),
                "stats": graph_data.get("stats", {}),
            }
            if graph_data.get("data_columns") is not None and graph_data.get("data_rows") is not None:
                export_data["data_columns"] = graph_data["data_columns"]
                export_data["data_rows"] = graph_data["data_rows"]
            print(f"✅ GET /api/knowledge/causal-graph/export: dataset {document_name} — {len(export_data['nodes'])} nodes, {len(export_data['edges'])} edges")
            return export_data

        # source == "all": KB + all data-driven datasets
        kb_data = get_causal_graph_data(include_inferred=include_inferred)
        datasets = list_data_driven_causal_graph_sources()
        export_data = {
            "metadata": {
                "version": "1.0",
                "exported_at": exported_at,
                "sources": ["kb"] + datasets,
                "include_inferred": include_inferred,
            },
            "kb": {
                "nodes": kb_data.get("nodes", []),
                "edges": kb_data.get("edges", []),
                "stats": kb_data.get("stats", {}),
            },
            "datasets": {},
        }
        for doc_name in datasets:
            g = get_data_driven_causal_graph(doc_name)
            if g:
                entry = {
                    "nodes": g.get("nodes", []),
                    "edges": g.get("edges", []),
                    "stats": g.get("stats", {}),
                }
                if g.get("data_columns") is not None and g.get("data_rows") is not None:
                    entry["data_columns"] = g["data_columns"]
                    entry["data_rows"] = g["data_rows"]
                export_data["datasets"][doc_name] = entry
        total_nodes = len(export_data["kb"]["nodes"]) + sum(len(v.get("nodes", [])) for v in export_data["datasets"].values())
        total_edges = len(export_data["kb"]["edges"]) + sum(len(v.get("edges", [])) for v in export_data["datasets"].values())
        print(f"✅ GET /api/knowledge/causal-graph/export: all — {total_nodes} nodes, {total_edges} edges (kb + {len(datasets)} dataset(s))")
        return export_data
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error exporting causal graph: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error exporting causal graph: {str(e)}")


@app.post("/api/knowledge/causal-graph/dowhy")
async def dowhy_effect_estimation_endpoint(body: DoWhyEffectRequest):
    """Run DoWhy causal effect estimation on a data-driven causal graph (dataset only)."""
    try:
        try:
            import dowhy  # noqa: F401
        except ModuleNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="DoWhy is not installed. In your backend environment run: pip install dowhy (or pip install -r requirements.txt), then restart the backend.",
            )
        from causal_graph import run_dowhy_effect_estimation
        result = run_dowhy_effect_estimation(
            document_name=body.document_name,
            treatment=body.treatment,
            outcome=body.outcome,
            method=body.method or "backdoor.linear_regression",
        )
        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "DoWhy estimation failed"),
            )
        return {
            "success": True,
            "treatment": result["treatment"],
            "outcome": result["outcome"],
            "estimate_value": result.get("estimate_value"),
            "estimate": result.get("estimate"),
            "refutation": result.get("refutation"),
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/knowledge/contents")
async def get_contents_endpoint():
    """Get all knowledge graph contents as text"""
    try:
        contents = kb_show_graph_contents()
        return {
            "contents": contents,
            "total_facts": len(kb_graph),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting contents: {str(e)}")

@app.get("/api/knowledge/facts")
async def get_facts_endpoint(
    include_inferred: bool = True,
    min_confidence: float = 0.0
):
    """Get all knowledge graph facts as structured JSON array"""
    try:
        # CRITICAL: Always reload from disk to get the latest saved facts
        # This ensures we have facts that were saved after upload, even if server was restarted
        # The in-memory graph might be empty if server was restarted (cleared on startup)
        load_result = kb_load_knowledge_graph()
        
        # Debug: If graph is empty but file exists, something is wrong
        import os
        if len(kb_graph) == 0 and os.path.exists(KG_FILE):
            file_size = os.path.getsize(KG_FILE)
            if file_size > 1000:  # File has data but graph is empty
                print(f"⚠️  WARNING: Graph file is {file_size} bytes but graph is empty!")
                # Try reloading again
                kb_load_knowledge_graph()
        
        facts = []
        from urllib.parse import unquote, quote
        import rdflib
        
        # OPTIMIZED: Build lookup maps in a single pass instead of calling functions for each fact
        # This reduces O(n*m) complexity to O(n) where n = total triples, m = facts
        # Build fact_id_uri -> metadata map first
        # ENHANCED: Now supports multiple sources per fact
        metadata_map = {}  # fact_id_uri -> {details, source_documents: [(source, timestamp), ...]}
        
        # Pass 1: Collect all metadata triples (O(n))
        # Metadata triples use fact_id_uri format: urn:fact:subject|predicate|object
        for s, p, o in kb_graph:
            predicate_str = str(p)
            # Check if this is a metadata triple (has fact_id_uri as subject)
            if 'urn:fact:' in str(s):
                fact_id_uri = str(s)
                
                if 'has_details' in predicate_str:
                    if fact_id_uri not in metadata_map:
                        metadata_map[fact_id_uri] = {'source_documents': []}
                    metadata_map[fact_id_uri]['details'] = str(o)
                elif 'source_document' in predicate_str:
                    if fact_id_uri not in metadata_map:
                        metadata_map[fact_id_uri] = {'source_documents': []}
                    # Source entry format: "source_document|uploaded_at" or just "source_document"
                    source_entry = str(o)
                    if '|' in source_entry:
                        parts = source_entry.split('|', 1)
                        if len(parts) == 2:
                            source_doc = parts[0]
                            timestamp = parts[1]
                            metadata_map[fact_id_uri]['source_documents'].append((source_doc, timestamp))
                    else:
                        # Legacy format or separate triples - will be matched with timestamp below
                        metadata_map[fact_id_uri]['source_documents'].append((source_entry, None))
                elif 'uploaded_at' in predicate_str:
                    if fact_id_uri not in metadata_map:
                        metadata_map[fact_id_uri] = {'source_documents': []}
                    timestamp = str(o)
                    # Try to match with existing source entries that don't have timestamps
                    if 'source_documents' in metadata_map[fact_id_uri]:
                        # Update entries without timestamps
                        updated_sources = []
                        for source_doc, existing_timestamp in metadata_map[fact_id_uri]['source_documents']:
                            if existing_timestamp is None:
                                updated_sources.append((source_doc, timestamp))
                            else:
                                updated_sources.append((source_doc, existing_timestamp))
                        # If no existing sources, add a new entry with this timestamp
                        if not updated_sources:
                            updated_sources.append(("", timestamp))
                        metadata_map[fact_id_uri]['source_documents'] = updated_sources
                elif 'is_inferred' in predicate_str:
                    if fact_id_uri not in metadata_map:
                        metadata_map[fact_id_uri] = {'source_documents': []}
                    # Store inferred status (convert "true"/"false" string to boolean)
                    metadata_map[fact_id_uri]['is_inferred'] = str(o).lower() == 'true'
                elif 'confidence' in predicate_str:
                    if fact_id_uri not in metadata_map:
                        metadata_map[fact_id_uri] = {'source_documents': []}
                    # Store confidence score (convert string to float)
                    try:
                        metadata_map[fact_id_uri]['confidence'] = float(str(o))
                    except (ValueError, TypeError):
                        metadata_map[fact_id_uri]['confidence'] = 0.7  # Default confidence
                elif 'processed_by_agent' in predicate_str:
                    if fact_id_uri not in metadata_map:
                        metadata_map[fact_id_uri] = {'source_documents': []}
                    # Store agent name
                    metadata_map[fact_id_uri]['agent'] = str(o).strip()
        
        # Pass 2: Collect facts and match with metadata using fact_id URI (O(n))
        fact_index = 0
        for s, p, o in kb_graph:
            # Skip metadata triples
            predicate_str = str(p)
            if ('fact_subject' in predicate_str or 'fact_predicate' in predicate_str or 
                'fact_object' in predicate_str or 'has_details' in predicate_str or 
                'source_document' in predicate_str or 'uploaded_at' in predicate_str or
                'is_inferred' in predicate_str or 'confidence' in predicate_str or
                'processed_by_agent' in predicate_str):
                continue
            
            fact_index += 1
            
            # Extract subject from URI (format: urn:entity:subject or urn:subject)
            # Handle both urn:entity:subject and urn:subject formats
            subject_uri_str = str(s)
            if 'urn:entity:' in subject_uri_str:
                subject = subject_uri_str.split('urn:entity:')[-1]
            elif 'urn:' in subject_uri_str:
                subject = subject_uri_str.split('urn:')[-1]
            else:
                subject = subject_uri_str
            subject = unquote(subject).replace('_', ' ')
            
            # Extract predicate from URI (format: urn:predicate:predicate or urn:predicate)
            # Handle both urn:predicate:predicate and urn:predicate formats
            predicate_uri_str = str(p)
            if 'urn:predicate:' in predicate_uri_str:
                predicate = predicate_uri_str.split('urn:predicate:')[-1]
            elif 'urn:' in predicate_uri_str:
                predicate = predicate_uri_str.split('urn:')[-1]
            else:
                predicate = predicate_uri_str
            predicate = unquote(predicate).replace('_', ' ')
            
            # Object is already a literal
            object_val = str(o)
            
            # Build fact_id URI the same way add_fact_source_document does (for lookup)
            # Format: subject|predicate|object -> urn:fact:subject|predicate|object
            fact_id = f"{subject}|{predicate}|{object_val}"
            fact_id_clean = fact_id.strip().replace(' ', '_')
            fact_id_uri = f"urn:fact:{quote(fact_id_clean, safe='')}"
            
            # Get metadata from lookup map (O(1) lookup)
            metadata = metadata_map.get(fact_id_uri, {})
            
            # Debug: Log if metadata not found (for troubleshooting)
            if not metadata and len(metadata_map) > 0:
                # Try to find a matching fact_id_uri (case-insensitive, partial match)
                matching_uri = None
                for stored_uri in metadata_map.keys():
                    if fact_id_uri.lower() in stored_uri.lower() or stored_uri.lower() in fact_id_uri.lower():
                        matching_uri = stored_uri
                        break
                if matching_uri:
                    print(f"⚠️  Fact ID URI mismatch: expected '{fact_id_uri}', found '{matching_uri}'")
                    metadata = metadata_map.get(matching_uri, {})
            
            # Get all sources for this fact
            source_documents = metadata.get('source_documents', [])
            
            # Format sources: if multiple, use the first one for backward compatibility
            # but also include all sources in a new field
            primary_source = source_documents[0][0] if source_documents and source_documents[0][0] else None
            primary_timestamp = source_documents[0][1] if source_documents and source_documents[0][1] else None
            
            # Format all sources as a list
            all_sources = []
            for source_doc, timestamp in source_documents:
                if source_doc:  # Only include entries with actual source documents
                    all_sources.append({
                        "document": source_doc,
                        "uploadedAt": timestamp if timestamp else None
                    })
            
            # Determine type from is_inferred status
            # Handle both boolean and string "true"/"false" values
            is_inferred_val = metadata.get('is_inferred', False)
            if isinstance(is_inferred_val, str):
                is_inferred = is_inferred_val.lower() == 'true'
            else:
                is_inferred = bool(is_inferred_val)
            fact_type = "inferred" if is_inferred else "original"
            
            # Get confidence score
            confidence = metadata.get('confidence', 0.7)  # Default confidence if not found
            
            # Get agent name
            agent_name = metadata.get('agent', '')  # Agent that processed the file
            
            # Apply filters
            if not include_inferred and is_inferred:
                continue  # Skip inferred facts if filter is enabled
            if confidence < min_confidence:
                continue  # Skip facts below confidence threshold
            
            facts.append({
                "id": str(fact_index),
                "subject": subject,
                "predicate": predicate,
                "object": object_val,
                "source": "knowledge_graph",
                "details": metadata.get('details') if metadata.get('details') else None,
                "sourceDocument": primary_source,  # Backward compatibility: first source
                "uploadedAt": primary_timestamp,  # Backward compatibility: first timestamp
                "sourceDocuments": all_sources if all_sources else None,  # New: all sources
                "isInferred": is_inferred,  # Backward compatibility: marks if fact is inferred (boolean)
                "type": fact_type,  # Primary field: type of fact ("original" or "inferred")
                "confidence": confidence,  # New: confidence score (0.0 to 1.0)
                "agent": agent_name if agent_name else None  # Agent that processed the file
            })
        
        print(f"✅ GET /api/knowledge/facts: Returning {len(facts)} facts")
        if len(facts) > 0:
            print(f"   Sample fact: {facts[0]}")
        else:
            print("   ⚠️  No facts in graph!")
            # Debug: Check if file exists
            if os.path.exists(KG_FILE):
                file_size = os.path.getsize(KG_FILE)
                print(f"   📁 knowledge_graph.pkl exists ({file_size} bytes) but graph is empty!")
                # If file exists but no facts, try to see what's in the graph
                all_triples = list(kb_graph)
                print(f"   📊 Total triples in graph: {len(all_triples)}")
                if len(all_triples) > 0:
                    print(f"   📊 Sample triple: {all_triples[0]}")
        
        # CRITICAL: Return facts in the format the frontend expects
        # Frontend expects: { success: true, data: { facts: [...] } }
        # But FastAPI returns directly, so we need to ensure the response has the right structure
        response = {
            "facts": facts,
            "total_facts": len(facts),  # Use len(facts) not len(kb_graph) since we filter metadata
            "status": "success"
        }
        return response
    except Exception as e:
        print(f"❌ Error getting facts: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting facts: {str(e)}")

@app.get("/api/documents")
async def get_documents_endpoint(include_all: bool = False):
    """Get all uploaded documents
    
    Args:
        include_all: If True, return all documents. If False (default), only return documents that contributed facts.
    """
    try:
        all_documents = get_all_documents()
        
        # Update fact counts for existing documents to ensure accuracy
        # This syncs document fact counts with actual facts in the graph
        from documents_store import count_facts_for_document, save_documents
        updated_docs = []
        for doc in all_documents:
            doc_name = doc.get('name', '')
            if doc_name:
                actual_facts = count_facts_for_document(doc_name)
                if actual_facts != doc.get('facts_extracted', 0):
                    doc['facts_extracted'] = actual_facts
                    print(f"🔄 Updated {doc_name}: fact count → {actual_facts}")
                updated_docs.append(doc)
        
        if updated_docs != all_documents:
            # Save updated documents if counts changed
            save_documents(updated_docs)
            all_documents = updated_docs
        
        # Filter: ONLY return documents that have contributed facts (facts_extracted > 0)
        # This ensures we NEVER show documents without facts
        if not include_all:
            documents = [doc for doc in all_documents if doc.get('facts_extracted', 0) > 0]
            print(f"✅ GET /api/documents: Returning {len(documents)} documents with facts (out of {len(all_documents)} total)")
        else:
            documents = all_documents
            print(f"✅ GET /api/documents: Returning {len(documents)} documents (all)")
        
        return {
            "documents": documents,
            "total_documents": len(documents),
            "total_all_documents": len(all_documents),
            "status": "success"
        }
    except Exception as e:
        print(f"❌ Error getting documents: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting documents: {str(e)}")

@app.delete("/api/documents/{document_id}")
async def delete_document_endpoint(document_id: str):
    """Delete a document by ID"""
    try:
        success = ds_delete_document(document_id)
        if success:
            return {
                "message": "Document deleted successfully",
                "status": "success"
            }
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.delete("/api/knowledge/delete")
async def delete_knowledge_endpoint(request: DeleteKnowledgeRequest):
    """Delete knowledge from the graph"""
    try:
        if request.keyword:
            result = kb_delete_all_knowledge()  # You may want to implement keyword-based deletion
        elif request.count:
            from knowledge import delete_recent_knowledge
            result = delete_recent_knowledge(request.count)
        else:
            result = kb_delete_all_knowledge()
        
        kb_save_knowledge_graph()
        return {
            "message": result,
            "status": "success",
            "total_facts": len(kb_graph)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting knowledge: {str(e)}")

@app.delete("/api/knowledge/facts/{fact_id}")
async def delete_fact_endpoint(fact_id: str):
    """Delete a specific fact by ID (subject, predicate, object)"""
    try:
        import rdflib
        from urllib.parse import quote
        from knowledge import fact_exists as kb_fact_exists
        
        # Parse fact_id - it should be in format "subject|predicate|object"
        # Or we can accept it as a JSON string
        try:
            import json
            from urllib.parse import unquote
            # Decode URL encoding first
            decoded_id = unquote(fact_id)
            fact_data = json.loads(decoded_id)
            subject = fact_data.get('subject')
            predicate = fact_data.get('predicate')
            object_val = fact_data.get('object')
        except json.JSONDecodeError:
            # Try parsing as pipe-separated
            parts = fact_id.split('|')
            if len(parts) == 3:
                subject, predicate, object_val = parts
            else:
                # Try to find fact by searching all facts
                # This is a fallback - ideally fact_id should be structured
                raise HTTPException(status_code=400, detail="Invalid fact ID format. Expected JSON or 'subject|predicate|object'")
        
        if not subject or not predicate or object_val is None:
            raise HTTPException(status_code=400, detail="Missing subject, predicate, or object")
        
        # Check if fact exists
        if not kb_fact_exists(subject, predicate, str(object_val)):
            raise HTTPException(status_code=404, detail="Fact not found in knowledge graph")
        
        # Create URI-encoded triple to match how it's stored
        subject_clean = str(subject).strip().replace(' ', '_')
        predicate_clean = str(predicate).strip().replace(' ', '_')
        object_value = str(object_val).strip()
        
        subject_uri = rdflib.URIRef(f"urn:{quote(subject_clean, safe='')}")
        predicate_uri = rdflib.URIRef(f"urn:{quote(predicate_clean, safe='')}")
        object_literal = rdflib.Literal(object_value)
        
        # Remove details first (if any)
        from knowledge import remove_fact_details as kb_remove_fact_details
        kb_remove_fact_details(subject, predicate, object_value)
        
        # Remove from graph
        if (subject_uri, predicate_uri, object_literal) in kb_graph:
            kb_graph.remove((subject_uri, predicate_uri, object_literal))
            kb_save_knowledge_graph()
            
            print(f"✅ DELETE /api/knowledge/facts/{fact_id}: Deleted fact - {subject} {predicate} {object_val}")
            print(f"✅ Graph now has {len(kb_graph)} facts")
            
            return {
                "message": "Fact deleted successfully",
                "status": "success",
                "deleted_fact": {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_val
                },
                "total_facts": len(kb_graph)
            }
        else:
            raise HTTPException(status_code=404, detail="Fact not found in knowledge graph")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting fact: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting fact: {str(e)}")

@app.get("/api/export")
async def export_knowledge_endpoint(
    include_inferred: bool = True,
    min_confidence: float = 0.0
):
    """Export all knowledge graph facts as JSON"""
    try:
        from datetime import datetime
        from urllib.parse import unquote
        
        # Reload graph to ensure we have latest facts
        kb_load_knowledge_graph()
        
        # Import get_fact_details function
        from knowledge import get_fact_details as kb_get_fact_details
        
        # Extract facts from graph
        facts = []
        # Import get_fact_source_document function
        from knowledge import get_fact_source_document as kb_get_fact_source_document
        
        for i, (s, p, o) in enumerate(kb_graph):
            # Skip metadata triples (those with special predicates for details, source document, timestamp)
            predicate_str = str(p)
            if ('fact_subject' in predicate_str or 'fact_predicate' in predicate_str or 
                'fact_object' in predicate_str or 'has_details' in predicate_str or 
                'source_document' in predicate_str or 'uploaded_at' in predicate_str or
                'is_inferred' in predicate_str):
                continue
            
            # Extract subject from URI (urn:subject -> subject)
            subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
            # Decode URL encoding and replace underscores back to spaces
            subject = unquote(subject).replace('_', ' ')
            
            # Extract predicate from URI
            predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
            predicate = unquote(predicate).replace('_', ' ')
            
            # Object is already a literal, just get the string value
            object_val = str(o)
            
            # Get details for this fact
            details = kb_get_fact_details(subject, predicate, object_val)
            
            # Get all source documents and timestamps
            all_sources = kb_get_fact_source_document(subject, predicate, object_val)
            
            # Get inferred status
            from knowledge import get_fact_is_inferred as kb_get_fact_is_inferred
            is_inferred_val = kb_get_fact_is_inferred(subject, predicate, object_val)
            # Handle None (not set) as False (original)
            is_inferred = bool(is_inferred_val) if is_inferred_val is not None else False
            
            # Get confidence score
            from knowledge import get_fact_confidence as kb_get_fact_confidence
            confidence = kb_get_fact_confidence(subject, predicate, object_val)
            
            # Determine type from is_inferred status
            fact_type = "inferred" if is_inferred else "original"
            
            # Apply filters
            if not include_inferred and is_inferred:
                continue  # Skip inferred facts if filter is enabled
            if confidence < min_confidence:
                continue  # Skip facts below confidence threshold
            
            # Format sources
            primary_source = all_sources[0][0] if all_sources and all_sources[0][0] else None
            primary_timestamp = all_sources[0][1] if all_sources and all_sources[0][1] else None
            
            # Format all sources as a list
            source_docs_list = []
            for source_doc, timestamp in all_sources:
                if source_doc:
                    source_docs_list.append({
                        "document": source_doc,
                        "uploadedAt": timestamp if timestamp else None
                    })
            
            facts.append({
                "id": str(i + 1),
                "subject": subject,
                "predicate": predicate,
                "object": object_val,
                "source": "knowledge_graph",
                "details": details if details else None,
                "sourceDocument": primary_source,  # Backward compatibility
                "uploadedAt": primary_timestamp,  # Backward compatibility
                "sourceDocuments": source_docs_list if source_docs_list else None,  # All sources
                "isInferred": is_inferred,  # Backward compatibility: marks if fact is inferred (boolean)
                "type": fact_type,  # Primary field: type of fact ("original" or "inferred")
                "confidence": confidence  # New: confidence score (0.0 to 1.0)
            })
        
        # Create export response with metadata
        export_data = {
            "facts": facts,
            "metadata": {
                "version": "1.2.3",
                "totalFacts": len(facts),
                "lastUpdated": datetime.now().isoformat()
            }
        }
        
        print(f"✅ GET /api/export: Exporting {len(facts)} facts")
        return export_data
    except Exception as e:
        print(f"❌ Error exporting knowledge: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error exporting knowledge: {str(e)}")

@app.post("/api/knowledge/import")
async def import_json_endpoint(file: UploadFile = File(...)):
    """Import knowledge from JSON file"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            facts_before = len(kb_graph)
            result = kb_import_json(tmp_path)
            kb_save_knowledge_graph()
            facts_after = len(kb_graph)
            facts_added = facts_after - facts_before
            
            # Parse the result message to extract added/skipped counts
            import re
            added_match = re.search(r'Imported (\d+)', result)
            skipped_match = re.search(r'skipped (\d+)', result)
            added_count = int(added_match.group(1)) if added_match else facts_added
            skipped_count = int(skipped_match.group(1)) if skipped_match else 0
            
            print(f"✅ POST /api/knowledge/import: Added {added_count} facts, skipped {skipped_count} duplicates")
            
            return {
                "message": result,
                "status": "success",
                "total_facts": len(kb_graph),
                "facts_added": added_count,
                "facts_skipped": skipped_count
            }
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing JSON: {str(e)}")

@app.get("/api/knowledge/stats")
async def get_stats_endpoint():
    """Get knowledge graph statistics"""
    try:
        return {
            "total_facts": len(kb_graph),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/api/knowledge/save")
async def save_knowledge_endpoint():
    """Manually trigger knowledge graph save"""
    try:
        result = kb_save_knowledge_graph()
        return {
            "message": result,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving knowledge: {str(e)}")

# ==========================================================
# Simulator Endpoints
# ==========================================================

# In-memory storage for experiments (could be replaced with database)
experiments_store: Dict[str, Any] = {}
experiments_file = "experiments_store.json"

def load_experiments():
    """Load experiments from file"""
    global experiments_store
    try:
        if os.path.exists(experiments_file):
            with open(experiments_file, 'r') as f:
                experiments_store = json.load(f)
    except Exception as e:
        print(f"Error loading experiments: {e}")
        experiments_store = {}

def save_experiments():
    """Save experiments to file"""
    try:
        with open(experiments_file, 'w') as f:
            json.dump(experiments_store, f, indent=2)
    except Exception as e:
        print(f"Error saving experiments: {e}")

# Load experiments on startup
load_experiments()

class ExperimentRequest(BaseModel):
    name: str
    scenarioDescription: str  # Natural language scenario description
    inputData: Optional[str] = None  # Optional extra data not in knowledge base
    scenarioType: str  # "hypothesis" | "prediction" | "what_if" | "validation"
    parameters: Optional[Dict[str, Any]] = {}

class ScenarioRequest(BaseModel):
    scenarioType: str
    inputData: str
    knowledgeGraph: Optional[Dict[str, Any]] = None

def run_experiment_logic(config: ExperimentRequest) -> Dict[str, Any]:
    """
    Core logic for running experiments based on knowledge graph.
    Analyzes natural language scenario description against the knowledge graph.
    """
    try:
        # Parse optional input data if provided
        additional_data = None
        if config.inputData and config.inputData.strip():
            try:
                if config.inputData.strip().startswith('{') or config.inputData.strip().startswith('['):
                    additional_data = json.loads(config.inputData.strip())
                else:
                    additional_data = {"text": config.inputData.strip()}
            except json.JSONDecodeError:
                additional_data = {"text": config.inputData.strip()}
        
        # Extract keywords and entities from natural language scenario description
        scenario_text = config.scenarioDescription.lower()
        
        # Simple keyword extraction - split by common words and punctuation
        # Remove common stop words for better matching
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'if', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
        words = re.findall(r'\b\w+\b', scenario_text)
        search_terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Also extract potential entity names (capitalized words or quoted strings)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', config.scenarioDescription)
        search_terms.extend([w.lower() for w in capitalized_words])
        
        # Extract quoted strings
        quoted_strings = re.findall(r'"([^"]+)"', config.scenarioDescription)
        search_terms.extend([s.lower() for s in quoted_strings])
        
        # Remove duplicates and keep only meaningful terms
        search_terms = list(set([term for term in search_terms if len(term) > 2]))[:20]  # Limit to 20 terms
        
        # Get relevant facts from knowledge graph
        relevant_facts = []
        fact_scores = {}  # Track relevance scores
        
        # Search knowledge graph for relevant facts
        for subject, predicate, obj in kb_graph:
            subject_str = str(subject).split(':')[-1] if ':' in str(subject) else str(subject)
            predicate_str = str(predicate).split(':')[-1] if ':' in str(predicate) else str(predicate)
            object_str = str(obj)
            
            # Calculate relevance score based on term matches
            score = 0
            fact_text = f"{subject_str} {predicate_str} {object_str}".lower()
            
            for term in search_terms:
                if term in fact_text:
                    # Higher score for exact matches in subject/object
                    if term in subject_str.lower() or term in object_str.lower():
                        score += 3
                    elif term in predicate_str.lower():
                        score += 2
                    else:
                        score += 1
            
            if score > 0:
                fact_id = f"{subject_str}|{predicate_str}|{object_str}"
                if fact_id not in fact_scores or fact_scores[fact_id] < score:
                    relevant_facts.append({
                        "subject": subject_str,
                        "predicate": predicate_str,
                        "object": object_str,
                        "relevance_score": score
                    })
                    fact_scores[fact_id] = score
        
        # Sort by relevance score (highest first)
        relevant_facts.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        relevant_facts = relevant_facts[:30]  # Limit to top 30 most relevant
        
        # Extract entities mentioned in scenario
        mentioned_entities = list(set([f["subject"] for f in relevant_facts[:15]] + [f["object"] for f in relevant_facts[:15]]))
        
        # Analyze based on scenario type
        analysis = {
            "scenario_type": config.scenarioType,
            "scenario_description": config.scenarioDescription,
            "relevant_facts_count": len(relevant_facts),
            "relevant_facts": relevant_facts,
            "mentioned_entities": mentioned_entities[:10],
            "search_terms_used": search_terms[:10],
            "additional_data": additional_data,
            "analysis": {}
        }
        
        if config.scenarioType == "hypothesis":
            # Test if hypothesis is supported by facts
            conclusion = "Hypothesis is supported by knowledge graph" if len(relevant_facts) >= 3 else \
                        "Hypothesis has partial support" if len(relevant_facts) > 0 else \
                        "Hypothesis needs more evidence in knowledge graph"
            
            analysis["analysis"] = {
                "hypothesis": config.scenarioDescription,
                "supporting_facts_count": len(relevant_facts),
                "conclusion": conclusion,
                "confidence": min(1.0, len(relevant_facts) / 5.0),
                "key_evidence": [f"{f['subject']} {f['predicate']} {f['object']}" for f in relevant_facts[:5]]
            }
        elif config.scenarioType == "prediction":
            # Make predictions based on patterns
            predicted_outcomes = []
            if len(relevant_facts) > 0:
                # Look for patterns in relationships
                relationships = {}
                for fact in relevant_facts[:10]:
                    pred = fact["predicate"]
                    if pred not in relationships:
                        relationships[pred] = []
                    relationships[pred].append(f"{fact['subject']} → {fact['object']}")
                
                for pred, examples in list(relationships.items())[:3]:
                    predicted_outcomes.append(f"Based on '{pred}' relationships: {', '.join(examples[:2])}")
            
            analysis["analysis"] = {
                "scenario": config.scenarioDescription,
                "predicted_outcomes": predicted_outcomes if predicted_outcomes else ["Insufficient data for prediction"],
                "confidence": min(0.9, len(relevant_facts) / 10.0),
                "basis": f"Analysis based on {len(relevant_facts)} relevant facts from knowledge graph"
            }
        elif config.scenarioType == "what_if":
            # Explore what-if scenarios
            affected_entities = list(set([f["subject"] for f in relevant_facts[:15]] + [f["object"] for f in relevant_facts[:15]]))
            potential_impacts = []
            
            if len(relevant_facts) > 0:
                # Group by relationships
                impact_groups = {}
                for fact in relevant_facts[:10]:
                    pred = fact["predicate"]
                    if pred not in impact_groups:
                        impact_groups[pred] = []
                    impact_groups[pred].append(fact["object"])
                
                for pred, entities in list(impact_groups.items())[:3]:
                    potential_impacts.append(f"'{pred}' may affect: {', '.join(set(entities)[:3])}")
            
            analysis["analysis"] = {
                "scenario": config.scenarioDescription,
                "affected_entities": affected_entities[:10],
                "potential_impacts": potential_impacts if potential_impacts else ["No clear impacts identified"],
                "related_facts_count": len(relevant_facts)
            }
        elif config.scenarioType == "validation":
            # Validate against knowledge graph
            validation_result = len(relevant_facts) >= 2
            validation_details = []
            
            if validation_result:
                validation_details.append(f"Scenario is validated by {len(relevant_facts)} matching facts")
                validation_details.append(f"Key entities found: {', '.join(mentioned_entities[:5])}")
            else:
                validation_details.append(f"Scenario has limited support ({len(relevant_facts)} matching facts)")
                if len(relevant_facts) == 0:
                    validation_details.append("No matching data found in knowledge graph")
            
            analysis["analysis"] = {
                "scenario": config.scenarioDescription,
                "matches_found": len(relevant_facts),
                "is_valid": validation_result,
                "validation_details": validation_details,
                "confidence": min(1.0, len(relevant_facts) / 3.0)
            }
        
        return {
            "status": "success",
            "results": analysis,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/simulator/experiment")
async def run_experiment_endpoint(request: ExperimentRequest):
    """Run an experiment based on knowledge graph"""
    try:
        # Run the experiment logic
        result = run_experiment_logic(request)
        
        # Store the experiment
        experiment_id = f"exp_{int(datetime.now().timestamp() * 1000)}"
        experiment_data = {
            "id": experiment_id,
            "config": request.dict(),
            "status": "completed" if result.get("status") == "success" else "failed",
            "results": result.get("results"),
            "error": result.get("error"),
            "timestamp": result.get("timestamp", datetime.now().isoformat())
        }
        
        experiments_store[experiment_id] = experiment_data
        save_experiments()
        
        return {
            "id": experiment_id,
            "results": result.get("results"),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running experiment: {str(e)}")

@app.post("/api/simulator/scenario")
async def test_scenario_endpoint(request: ScenarioRequest):
    """Test a scenario against the knowledge graph"""
    try:
        # Create a temporary experiment config
        temp_config = ExperimentRequest(
            name=f"Scenario Test - {request.scenarioType}",
            scenarioDescription=request.inputData,  # Use inputData as scenario description
            inputData=None,  # No additional data for quick tests
            scenarioType=request.scenarioType,
            parameters={}
        )
        
        # Run the scenario logic
        result = run_experiment_logic(temp_config)
        
        return {
            "results": result.get("results"),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing scenario: {str(e)}")

@app.get("/api/simulator/experiments")
async def get_experiments_endpoint():
    """Get all stored experiments"""
    try:
        # Convert dict to list
        experiments_list = list(experiments_store.values())
        # Sort by timestamp (newest first)
        experiments_list.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return {
            "experiments": experiments_list,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting experiments: {str(e)}")

@app.delete("/api/simulator/experiments/{experiment_id}")
async def delete_experiment_endpoint(experiment_id: str):
    """Delete an experiment"""
    try:
        if experiment_id in experiments_store:
            del experiments_store[experiment_id]
            save_experiments()
            return {
                "message": "Experiment deleted",
                "status": "success"
            }
        else:
            raise HTTPException(status_code=404, detail="Experiment not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting experiment: {str(e)}")
@app.post("/api/reset")
async def reset_endpoint():
    """
    Reset the app: erase all documents, knowledge base, and knowledge graph.
    Equivalent to restarting the app with a clean state.
    """
    try:
        docs_deleted = ds_delete_all_documents()
        kb_result = kb_delete_all_knowledge()
        return {
            "message": "App reset successfully. All documents, knowledge base, and knowledge graph have been erased.",
            "status": "success",
            "documents_deleted": docs_deleted,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting app: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 8001 (frontend expects this port)
    port = int(os.getenv("API_PORT", 8001))
    host = os.getenv("API_HOST", "0.0.0.0")  # Bind to all interfaces for external access
    
    # Check if the requested port is available (use same host as uvicorn to avoid bind failures)
    import socket
    def is_port_available(check_port):
        """Check if a port is available by trying to bind to it on the same host uvicorn will use."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, check_port))
            sock.close()
            return True
        except OSError:
            return False

    # Try requested port first, then 8002, 8003, 8004 so a second run (e.g. Code Runner) doesn't crash
    for attempt_port in [port, 8002, 8003, 8004]:
        if is_port_available(attempt_port):
            port = attempt_port
            if attempt_port != int(os.getenv("API_PORT", 8001)):
                print(f"⚠️  Port 8001 busy; using port {port} instead. Frontend: http://localhost:{port}")
            break
    else:
        print(f"⚠️  Ports 8001–8004 busy. Using {port} anyway (may fail). Stop other backends or set API_PORT.")
    
    print(f"Starting NesyX API server on http://{host}:{port}")
    print(f"API documentation available at http://localhost:{port}/docs")
    print(f"Frontend should connect to: http://localhost:{port}")
    
    uvicorn.run(app, host=host, port=port)

