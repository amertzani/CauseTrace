# Multi-Agent File Processing Architecture

## Overview

The file processing system has been refactored to use a **multi-agent architecture** where specialized agents handle different file types. This provides better separation of concerns, extensibility, and maintainability.

## What Changed

### New Structure

```
agents/
├── __init__.py          # Module exports
├── base_agent.py        # Abstract base class
├── pdf_agent.py         # PDF file processing
├── docx_agent.py        # DOCX/DOC file processing
├── txt_agent.py         # TXT/MD file processing
├── csv_agent.py         # CSV file processing
├── xlsx_agent.py        # Excel XLSX file processing
├── coordinator.py       # Routes files to agents
└── README.md            # Detailed documentation
```

### Key Components

1. **BaseAgent**: Abstract base class defining the agent interface
2. **Specialized Agents**: One agent per file type (PDF, DOCX, TXT, CSV, XLSX)
3. **AgentCoordinator**: Routes files to appropriate agents based on extension

## Benefits

### 1. **Separation of Concerns**
- Each agent handles only one file type
- Changes to PDF processing don't affect CSV processing
- Easier to debug and maintain

### 2. **Extensibility**
- Adding a new file type = adding a new agent
- No need to modify existing code
- Simple registration process

### 3. **Format-Specific Optimizations**
- **CSVAgent**: Intelligent separator detection, data summaries
- **TXTAgent**: Automatic encoding detection
- **PDFAgent**: Page-by-page extraction with metadata
- **DOCXAgent**: Table and paragraph extraction
- **XLSXAgent**: Multi-sheet support

### 4. **Better Error Handling**
- Isolated error handling per file type
- Detailed error messages with context
- Graceful fallbacks

### 5. **Rich Metadata**
- Each agent extracts format-specific metadata:
  - PDF: Pages, title, author
  - DOCX: Paragraphs, tables, document properties
  - CSV: Rows, columns, data types
  - XLSX: Sheets, dimensions
  - TXT: Lines, encoding

## Backward Compatibility

✅ **All existing code continues to work!**

The `file_processing.py` module maintains the same function signatures:
- `handle_file_upload()` - Still works the same way
- `process_uploaded_file()` - Still works the same way
- Legacy functions (`extract_text_from_pdf()`, etc.) - Still available

The multi-agent system is used internally but doesn't break existing code.

## Usage Examples

### Direct Agent Usage

```python
from agents import AgentCoordinator

coordinator = AgentCoordinator()

# Process a file
result = coordinator.process_file('document.pdf')
if result['success']:
    print(f"Text: {result['text'][:100]}...")
    print(f"Agent: {result['agent']}")
    print(f"Metadata: {result['metadata']}")
```

### Through file_processing.py (Existing API)

```python
from file_processing import handle_file_upload

# Works exactly as before
result = handle_file_upload(['file1.pdf', 'file2.docx', 'file3.csv'])
```

## Supported File Types

| Format | Extension | Agent | Status |
|--------|-----------|-------|--------|
| PDF | `.pdf` | PDFAgent | ✅ Active |
| Word | `.docx`, `.doc` | DOCXAgent | ✅ Active |
| Text | `.txt`, `.text` | TXTAgent | ✅ Active |
| Markdown | `.md`, `.markdown` | TXTAgent | ✅ Active |
| CSV | `.csv` | CSVAgent | ✅ Active |
| Excel | `.xlsx`, `.xlsm` | XLSXAgent | ⚠️ Requires `openpyxl` |

## Installation

New dependencies added to `requirements.txt`:
- `openpyxl` - For Excel XLSX support
- `chardet` - For encoding detection in text files

Install with:
```bash
pip install openpyxl chardet
```

Or install all requirements (two-stage to avoid pip ResolutionTooDeep):
```bash
pip install -r requirements-core.txt
BLIS_ARCH=generic pip install -r requirements-ml.txt
```
Or run: `./install_requirements.sh`

## Adding New File Types

To add support for a new file type (e.g., PPTX):

1. **Create the agent** (`agents/pptx_agent.py`):
```python
from .base_agent import BaseAgent

class PPTXAgent(BaseAgent):
    def __init__(self):
        super().__init__("PPTX Agent")
        self.supported_extensions = ['.pptx']
    
    def can_process(self, file_path: str) -> bool:
        return file_path.endswith('.pptx')
    
    def extract_text(self, file_path: str) -> str:
        # Your extraction logic
        pass
```

2. **Register in coordinator** (`agents/coordinator.py`):
```python
from .pptx_agent import PPTXAgent

def __init__(self):
    self.agents: List[BaseAgent] = [
        # ... existing agents
        PPTXAgent(),  # Add here
    ]
```

3. **Export in __init__.py**:
```python
from .pptx_agent import PPTXAgent
__all__ = [..., 'PPTXAgent']
```

That's it! The new file type is now supported.

## Testing

Test the system:
```bash
cd /Users/s20/Enesy
python3 -c "from agents import AgentCoordinator; c = AgentCoordinator(); print(c.get_supported_extensions())"
```

Expected output:
```
['.md', '.pdf', '.doc', '.text', '.markdown', '.docx', '.xlsx', '.xlsm', '.txt', '.csv']
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│           AgentCoordinator                      │
│  (Routes files to appropriate agents)           │
└──────────────┬──────────────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
   ┌───▼───┐      ┌───▼───┐
   │ PDF   │      │ DOCX  │
   │ Agent │      │ Agent │
   └───────┘      └───────┘
       │               │
   ┌───▼───┐      ┌───▼───┐
   │ TXT   │      │ CSV   │
   │ Agent │      │ Agent │
   └───────┘      └───────┘
       │               │
   ┌───▼───┐      ┌───▼───┐
   │ XLSX  │      │ ...   │
   │ Agent │      │       │
   └───────┘      └───────┘
```

## Migration Notes

✅ **No migration needed!** The system is backward compatible.

Existing code using `file_processing.py` will automatically use the new multi-agent system without any changes.

## Performance

- **Fast routing**: Extension-based lookup (O(1))
- **Lazy initialization**: Agents created only when needed
- **Efficient processing**: Format-specific optimizations
- **Parallel-ready**: Architecture supports future parallel processing

## Future Enhancements

Potential improvements:
1. **Parallel processing**: Process multiple files simultaneously
2. **Caching**: Cache extracted text for repeated processing
3. **Streaming**: Process large files in chunks
4. **More agents**: PPTX, ODT, RTF, etc.
5. **Agent metrics**: Track processing times, success rates

## Questions?

See `agents/README.md` for detailed documentation on:
- Agent implementation details
- Advanced usage examples
- Error handling patterns
- Best practices

