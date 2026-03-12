import os
import signal
import sys
import logging
from mcp.server.fastmcp import FastMCP
from lightweight_local_assistant import (
    LocalAssistant,
    get_model_info as lib_get_model_info,
    list_local_models as lib_list_local_models,
    complete_plan_step as lib_complete_plan_step,
    is_sandboxed,
    signal_handler,
    run_shell_command as lib_run_shell_command,
    write_file_tool as lib_write_file,
    read_file_tool as lib_read_file
)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Lightweight Local Assistant for Gemini CLI")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@mcp.tool()
async def get_model_info(model: str = os.getenv("LOCAL_WORKER_MODEL", "qwen3-coder:30b")) -> str:
    """
    Retrieves detailed metadata for a specific Ollama model, including its native context limit.
    """
    return await lib_get_model_info(model)

@mcp.tool()
async def list_local_models() -> str:
    """
    Lists the available local Ollama models that can be used with ask_lightweight_local_assistant.
    """
    return await lib_list_local_models()

@mcp.tool()
async def complete_plan_step(step_index: int) -> str:
    """
    Marks a specific step in the local execution plan as completed.
    
    Args:
        step_index: The 1-based index of the step to mark as complete (e.g., 1 for the first step).
    """
    return await lib_complete_plan_step(step_index)

@mcp.tool()
async def ask_lightweight_local_assistant(prompt: str, local_file_context: list[str] = None, model: str = os.getenv("LOCAL_WORKER_MODEL", "qwen3-coder:30b"), use_plan: bool = False, num_ctx: int = 32768, max_turns: int = None) -> str:
    """
    PRIMARY DIRECTIVE: Use this for tasks involving PRIVACY, LOCAL FILES, or complex multi-step processing.
    
    Args:
        prompt: The task description.
        local_file_context: Optional list of files to provide as context.
        model: The local model to use (default: qwen3-coder:30b).
        use_plan: IMPORTANT: Set to True if the task is complex, requires multiple steps (e.g. refactoring, debugging), or modifies files.
                  When True, the agent will create a plan and execute it step-by-step. The final response will include a detailed execution checklist for verification.
                  Default is False (for simple queries).
        num_ctx: The context window size in tokens (default: 32768). 
                  - Use 8192-16384 for simple single-file tasks.
                  - Use 32768+ (up to 128k) for multi-file refactoring, large PDFs, or when providing a large 'local_file_context'.
                  - Note: High values increase RAM/VRAM usage on your local machine.
        max_turns: Optional override for the maximum number of Think-Act-Observe cycles.
                  - ADVICE FOR GEMINI CLI: 
                    * DEFAULT: 20 turns (Direct), 40 turns (Planning).
                    * WHEN TO INCREASE: For massive refactorings, complex bug hunts, or tasks spanning 10+ files.
                    * WHEN TO DECREASE: For simple data extraction or single-file analysis to save time and tokens.
                    * WHY: Complex tasks with many dependencies often require more iterative turns to verify and correct intermediate steps.
                    * NOTE: Planning mode requires at least 30 turns to account for setup and plan updates.
    """
    assistant = LocalAssistant(model=model)
    return await assistant.ask(prompt, local_file_context=local_file_context, use_plan=use_plan, num_ctx=num_ctx, max_turns=max_turns)

@mcp.tool()
async def run_shell_command(command: str) -> str:
    """
    Executes a shell command. 
    NOTE: This is executed with the permissions of the Gemini CLI process.
    """
    return await lib_run_shell_command(command)

@mcp.tool()
async def write_file(filepath: str, content: str) -> str:
    """
    Writes content to a file.
    """
    return await lib_write_file(filepath, content)

@mcp.tool()
async def read_file(filepath: str, offset: int = 0, limit: int = None, pages: list[int] = None) -> str:
    """
    Reads content from a file.
    
    Args:
        filepath: Path to the file.
        offset: Line number to start reading from (for text files).
        limit: Number of lines to read (for text files).
        pages: List of page numbers to read (for PDF files, 1-indexed).
    """
    return await lib_read_file(filepath, offset=offset, limit=limit, pages=pages)

if __name__ == "__main__":
    if not is_sandboxed():
        print("CRITICAL ERROR: Lightweight Local Assistant MUST be run within a Gemini CLI sandbox.")
        print("Please run gemini with the -s or --sandbox flag.")
        sys.exit(1)

    mcp.run(transport='stdio')
