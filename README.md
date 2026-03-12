# Lightweight Local Assistant

**A powerful Python library for local AI automation, with an optional Gemini CLI extension.**

Now, anywhere you build with Python, you can drop in an autonomous local assistant agent. 

The `lightweight_local_assistant` is an asynchronous Python framework that provides a "Think-Act-Observe" reasoning loop for local LLMs (via Ollama). It empowers your local models to iteratively interact with your file system, execute shell commands, and read documents to accomplish complex tasks. 

**Use Cases:**
*   **Automated Refactoring:** Write a Python script that points the assistant at a directory and asks it to update outdated API calls across all files.
*   **Data Extraction:** Create a script that uses the assistant to read through hundreds of local PDFs and compile a summary report.
*   **Local RAG Pipelines:** Use the assistant as the intelligent "worker" node in your custom local data processing pipelines.
*   **Interactive CLI:** Use the provided MCP extension to bring local context into the cloud-based Gemini CLI.

Privacy is never guaranteed. If you use the optional Gemini CLI extension or send the local assistant's output to an external API, sensitive data could be sent to the cloud.

---

## ⚠️ Security Warning & Sandboxing

**CRITICAL: When used as a standard Python library, the `LocalAssistant` is ENTIRELY UNSANDBOXED.** 

It executes shell commands with the full permissions of the user running the script. A poorly phrased prompt or an LLM hallucination could result in the agent executing destructive commands (e.g., `rm -rf`, overwriting critical files, or exposing secrets). 

**You should ONLY run your Python programs utilizing this library inside a suitable sandbox.**

### How to Sandbox Your Python Scripts

#### macOS (Seatbelt)
macOS includes a native, highly configurable sandboxing mechanism called `seatbelt` (accessed via `sandbox-exec`). This is the same mechanism used by the Gemini CLI extension.

1.  Create a profile file named `sandbox.sb`:
    ```lisp
    (version 1)
    (deny default)
    (allow file-read*)
    ; ONLY allow writing to the current directory and its subdirectories
    (allow file-write* (subpath (param "PWD")))
    (allow process-exec*)
    (allow process-fork)
    (allow network-bind)
    (allow network-outbound)
    ```
2.  Run your script using `sandbox-exec`:
    ```bash
    sandbox-exec -f sandbox.sb -D PWD="$PWD" python3 your_script.py
    ```

#### Linux (Bubblewrap / Docker)
*   **Bubblewrap (`bwrap`):** A lightweight unprivileged sandboxing tool.
    ```bash
    bwrap --ro-bind / / --bind "$PWD" "$PWD" --unshare-all --share-net python3 your_script.py
    ```
    *(This mounts the entire system as read-only, except for the current directory which is writable).*
*   **Docker:** Run your script inside a minimal Docker container, mounting only the necessary directories as volumes.

#### Windows (Windows Sandbox / WSL)
*   **Windows Sandbox:** A lightweight, isolated desktop environment. You can map a specific folder to the sandbox and run your Python script there. The environment is destroyed when closed.
*   **WSL 2 (Windows Subsystem for Linux):** While not a strict security sandbox against the WSL instance itself, running the script inside WSL prevents it from directly manipulating your core Windows system files. You can further sandbox inside WSL using Docker.

---

## 🚀 Why use this?
*   **Flexibility:** Use it as a library in your own Python automation scripts to build custom local AI agents.
*   **Privacy:** Summarize sensitive PDFs or analyze private code entirely on your local machine.
*   **Power:** Gives local LLMs "hands" to run shell commands (`grep`, `find`, `ls`), write code, and read files iteratively.
*   **Cost & Speed:** Offloads heavy processing (like searching through large directories) to your local hardware.

## 🛠️ Prerequisites
1.  **[Ollama](https://ollama.com/):** Must be installed and running to provide the local AI models.
2.  **macOS (Optional but Recommended):** Required for the strict security sandbox (`libsandbox`) if you choose to use the Gemini CLI extension.

## 📦 Installation (Python Library)

You can easily install `lightweight_local_assistant` into your preferred Python environment (Virtualenv, Conda, or Mamba).

1.  **Start Ollama** and pull a capable model (we recommend `qwen3-coder:30b` if you have at least 32 GB of RAM, or `llama3` for smaller setups):
    ```bash
    ollama pull qwen3-coder:30b
    ```

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Local-Assistant-for-Gemini-CLI.git
    cd Local-Assistant-for-Gemini-CLI
    ```

3.  **Activate your environment:**
    *   **Virtualenv:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    *   **Conda:**
        ```bash
        conda create -n lightweight-local-assistant-env python=3.10
        conda activate lightweight-local-assistant-env
        ```
    *   **Mamba:**
        ```bash
        mamba create -n lightweight-local-assistant-env python=3.10
        mamba activate lightweight-local-assistant-env
        ```

4.  **Install the package:**
    Install it in standard mode or editable mode (if you plan to modify the code):
    ```bash
    pip install .
    # OR for editable mode:
    pip install -e .
    ```

## 💡 Quickstart & API Reference

### 1. Initialization

The `LocalAssistant` class is your primary entry point.

```python
from lightweight_local_assistant import LocalAssistant

# Basic initialization (uses default model qwen3-coder:30b)
assistant = LocalAssistant()

# Initialize with a specific model and Ollama URL
assistant = LocalAssistant(
    model="llama3", 
    ollama_url="http://localhost:11434"
)
```

### 2. The `ask` Method

The `.ask()` method is how you send tasks to the agent. It triggers the iterative Think-Act-Observe loop.

```python
import asyncio
from lightweight_local_assistant import LocalAssistant

async def main():
    assistant = LocalAssistant()
    
    # Example: Simple Shell Task
    print("--- Shell Example ---")
    response = await assistant.ask("What is the current git branch?")
    print(response)

    # Example: Reading and Summarizing
    print("\n--- Reading Example ---")
    response = await assistant.ask("Read README.md and tell me the required dependencies.")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

**Parameters for `ask()`:**
*   `prompt` (str): The task or question for the assistant.
*   `local_file_context` (list[str], optional): A list of filenames to hint to the model that it should prioritize reading these files.
*   `use_plan` (bool, default `False`): If `True`, forces the agent into Planning Mode. It will first create a `.gemini/local_plan.md` checklist using discovery tools, and then execute it step-by-step. Use this for complex refactoring or multi-file operations.
*   `num_ctx` (int, default `32768`): The context window size to request from Ollama. Increase this if you expect the agent to read very large files.
*   `max_turns` (int, optional): Limits the maximum number of tool-call iterations the agent can make before forcing a final answer. Defaults to 20 for direct execution and 40 for planning mode.

### 3. Utility Functions

The library also exposes the underlying tools if you wish to use them directly in your scripts:

```python
from lightweight_local_assistant import run_shell_command, read_file_tool, write_file_tool

async def utils_demo():
    # Execute a shell command safely with timeout protection
    output = await run_shell_command("ls -la")
    print(output)
    
    # Read specific lines from a file
    content = await read_file_tool("my_file.txt", offset=10, limit=5)
    
    # Read specific pages from a PDF
    pdf_content = await read_file_tool("document.pdf", pages=[1, 3])
    
    # Write content to a file (creates directories if needed)
    await write_file_tool("output/summary.txt", "This is the summary.")
```

## ✨ Key Features
*   **Intelligent Planning:** For complex tasks, the assistant generates a `.gemini/local_plan.md` checklist and executes it iteratively.
*   **Batch Operations:** Built-in tools support multiple paths or commands in a single turn, significantly reducing latency.
*   **Context Guard:** Automatically detects large outputs (like massive log files) and truncates them with segmentation instructions to prevent context overflow.
*   **Self-Reflection:** The agent evaluates its own work against your prompt and extends its turn limit if it realizes a task is incomplete.
*   **Native Document Support:** Reads both plain text and PDF files natively, with support for pagination and reading file "tails".

---

## 🎁 Bonus: Gemini CLI Extension

If you use the [Gemini CLI](https://github.com/google/gemini-cli), you can use this library as a secure MCP extension. This creates a "Cloud Brain, Local Hands" architecture where Gemini orchestrates high-level goals and delegates local file operations to your machine.

**⚠️ macOS Only:** The Gemini CLI extension is strictly limited to macOS. This is because it relies on the macOS `seatbelt` sandboxing mechanism (via Gemini's `-s` flag) to ensure the local agent cannot access files outside of the current working directory. The extension will refuse to start on other operating systems or if sandboxing is bypassed.

### Installing the Extension
Run the provided installer script. It will set up an isolated virtual environment and register the tool with Gemini CLI.
```bash
./install_extension.sh
```

### Using the Extension
**CRITICAL:** You must use the `-s` (sandbox) flag when starting Gemini CLI. The extension will refuse to run without macOS sandboxing enabled.
```bash
gemini -s "Use the local assistant to summarize 'contract_draft.pdf' and save it to 'summary.md'"
```

### Uninstalling the Extension
1.  Unregister from Gemini CLI: `gemini mcp remove --scope user lightweight-local-assistant`
2.  Delete the installation directory: `rm -rf ~/.gemini-lightweight-local-assistant`

---

## 🏗️ Architecture
The core logic resides in `lightweight_local_assistant/agent.py`. It implements a robust `while` loop that interfaces with Ollama:
1.  **System Prompt Injection:** Injects dynamic context about the available tools, the project directory, and the model's native context limit.
2.  **Tool Execution:** Parses the LLM's requested function calls (e.g., `read_file`, `run_shell_command`), executes them locally, and returns the results.
3.  **Reflection:** Prompts the model to verify if the original goal has been achieved before returning a final response.

## 📜 License
MIT
