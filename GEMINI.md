# Lightweight Local Assistant - Development Context

This project provides a secure, private bridge between cloud-based LLMs (like Gemini) and a user's local machine. It serves a dual purpose:
1. A **standalone Python library** for building autonomous local AI workflows.
2. A **Gemini CLI extension** (via MCP) that allows Gemini to interactively perform local tasks.

## Architecture: "Cloud Brain, Local Hands"

The core philosophy of this project is to delegate sensitive file processing and command execution to a local AI model (via Ollama), while a more powerful cloud model acts as the orchestrator.

### The Python Library (`lightweight_local_assistant`)

- **`LocalAssistant` (in `agent.py`)**: The primary class. It implements a "Think-Act-Observe" reasoning loop. It dynamically injects context limits, handles self-reflection to determine if a task is complete, and supports a structured **Planning Mode** (`use_plan=True`) for complex tasks.
- **`tools.py`**: Contains OS-agnostic implementations of the core capabilities:
  - `run_shell_command`: Executes commands via `asyncio.create_subprocess_shell`.
  - `_read_local_file`: Efficiently reads text and PDF files, supporting pagination and tailing to manage context windows.
  - `write_file_tool`: Safely writes output to disk.
- **`models.py`**: Handles discovery of local Ollama models and their context limits.
- **`security.py`**: Manages sandbox detection and the cleanup of orphaned subprocesses.

### The Gemini CLI Extension (`mcp_server.py`)

- This is a thin wrapper built using `FastMCP`.
- It imports the library's functions and exposes them as tools to the Gemini CLI.
- **Security Constraint:** When run as an MCP server for Gemini CLI, the script enforces a strict macOS sandboxing check (`libsandbox.1.dylib`). It will refuse to start if it is not running within a secure `seatbelt` environment to prevent the cloud agent from arbitrarily exploring the host filesystem.

## Development & Testing

### Installation for Development
1. Clone the repository.
2. Create a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
3. Install in editable mode with dev dependencies: `pip install -e ".[dev]"`

### Testing Strategy
The test suite (in the `test/` directory) uses `pytest` and `unittest` paradigms. It heavily utilizes `unittest.mock.patch` and `AsyncMock` to isolate the `LocalAssistant` logic from the actual Ollama network calls and local file system side-effects.

**To run the tests:**
```bash
pytest
```

**Key Test Coverage Areas:**
- **Agent Loop**: Verification of turn limits, missing tool handling, and self-correction reflection.
- **Planning Mode**: Verification of the 2-phase (plan -> execute) workflow and the checklist update logic.
- **V2 Features**: Context guard truncation, batch operations, and PDF reading.
- **Security**: Verification of the macOS sandbox check.

## Future Roadmap Ideas

1. **Intelligent Summarization Pass:** Add an optional tool for the local agent to summarize massive files before returning context to the cloud orchestrator.
2. **Local RAG Integration:** Connect to a local Vector DB to enable semantic project search rather than relying purely on shell `grep`/`find`.
3. **Safe Mode:** Create a strictly read-only version of the `ask` loop that strips write/execute permissions entirely.
4. **Dynamic Routing:** Allow the assistant to switch local models on the fly (e.g., using a coding model for refactoring and a general model for summarization).
