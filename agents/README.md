# Multi-Agent File Processing System

This directory contains a multi-agent architecture for processing different file types. Each agent specializes in handling a specific file format, providing optimized extraction and processing.

## Architecture

### Base Agent (`base_agent.py`)
Abstract base class that all specialized agents inherit from. Defines the interface:
- `can_process(file_path)`: Check if agent can handle the file
- `extract_text(file_path)`: Extract text content
- `extract_metadata(file_path)`: Extract format-specific metadata
- `process(file_path)`: Main processing method

### Specialized Agents

1. **PDFAgent** (`pdf_agent.py`)
   - Handles: `.pdf`
   - Uses: PyPDF2
   - Features: Page-by-page extraction, PDF metadata extraction

2. **DOCXAgent** (`docx_agent.py`)
   - Handles: `.docx`, `.doc`
   - Uses: python-docx
   - Features: Paragraph and table extraction, document properties

3. **TXTAgent** (`txt_agent.py`)
   - Handles: `.txt`, `.text`, `.md`, `.markdown`
   - Uses: Automatic encoding detection (chardet)
   - Features: Multi-encoding support, line counting

4. **CSVAgent** (`csv_agent.py`)
   - Handles: `.csv`
   - Uses: pandas
   - Features: Intelligent separator detection, data summary, structured formatting

5. **XLSXAgent** (`xlsx_agent.py`)
   - Handles: `.xlsx`, `.xlsm`
   - Uses: openpyxl
   - Features: Multi-sheet support, Excel metadata extraction

### Coordinator (`coordinator.py`)
Routes files to the appropriate agent based on file extension. Provides:
- Automatic agent selection
- Unified processing interface
- Support for batch processing

## Usage

### Basic Usage

```python
from agents import AgentCoordinator

# Initialize coordinator
coordinator = AgentCoordinator()

# Process a single file
result = coordinator.process_file('document.pdf', original_filename='my_document.pdf')

if result['success']:
    text = result['text']
    metadata = result['metadata']
    agent_name = result['agent']
    print(f"Processed by {agent_name}")
    print(f"Extracted {len(text)} characters")
else:
    print(f"Error: {result['error']}")
```

### Batch Processing

```python
# Process multiple files
files = ['file1.pdf', 'file2.docx', 'file3.csv']
results = coordinator.process_files(files)

for result in results:
    if result['success']:
        print(f"✓ {result['metadata']['file_name']} - {result['agent']}")
    else:
        print(f"✗ {result['metadata']['file_name']} - {result['error']}")
```

### Check Supported Formats

```python
# Get all supported extensions
extensions = coordinator.get_supported_extensions()
print(f"Supported: {', '.join(extensions)}")

# Get agent information
agent_info = coordinator.get_agent_info()
for agent_name, extensions in agent_info.items():
    print(f"{agent_name}: {', '.join(extensions)}")
```

## Integration

The multi-agent system is integrated into `file_processing.py`:

```python
from file_processing import handle_file_upload, process_uploaded_file

# Process files (uses agents internally)
result = handle_file_upload(['file.pdf', 'file.docx'])
```

## Adding New Agents

To add a new agent:

1. Create a new agent class inheriting from `BaseAgent`:

```python
from .base_agent import BaseAgent

class MyNewAgent(BaseAgent):
    def __init__(self):
        super().__init__("My New Agent")
        self.supported_extensions = ['.myext']
    
    def can_process(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_extensions
    
    def extract_text(self, file_path: str) -> str:
        # Your extraction logic here
        pass
```

2. Register it in `coordinator.py`:

```python
from .my_new_agent import MyNewAgent

def __init__(self):
    self.agents: List[BaseAgent] = [
        PDFAgent(),
        DOCXAgent(),
        # ... other agents
        MyNewAgent(),  # Add your agent here
    ]
```

3. Update `__init__.py` to export the new agent.

## Dependencies

- **PDFAgent**: `PyPDF2` (already in requirements.txt)
- **DOCXAgent**: `python-docx` (already in requirements.txt)
- **TXTAgent**: `chardet` (added to requirements.txt)
- **CSVAgent**: `pandas` (already in requirements.txt)
- **XLSXAgent**: `openpyxl` (added to requirements.txt)

## Benefits

1. **Separation of Concerns**: Each agent handles one file type
2. **Extensibility**: Easy to add new file types
3. **Maintainability**: Changes to one agent don't affect others
4. **Optimization**: Format-specific optimizations per agent
5. **Error Handling**: Isolated error handling per file type
6. **Metadata**: Rich metadata extraction per format

## Error Handling

Each agent handles errors gracefully:
- Returns structured error messages
- Continues processing other files if one fails
- Provides detailed error information in metadata

## Performance

- Agents are initialized once (singleton coordinator)
- Fast extension-based routing
- Format-specific optimizations (e.g., CSV uses pandas efficiently)

