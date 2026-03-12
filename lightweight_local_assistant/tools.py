import os
import logging
import asyncio
from itertools import islice
from pypdf import PdfReader
from .security import active_subprocesses, PLAN_FILE

logger = logging.getLogger(__name__)

def _read_local_file(filepath: str, offset: int = 0, limit: int = None, pages: list[int] = None, tail: int = None) -> tuple[str, int, str]:
    """
    Helper to read text or PDF files locally.
    Returns: (content_string, total_units, unit_name)
    """
    if not os.path.exists(filepath):
        return f"[Error: File {filepath} not found]", 0, "unknown"

    if filepath.lower().endswith(".pdf"):
        try:
            reader = PdfReader(filepath)
            total_pages = len(reader.pages)
            text = ""
            
            # If tail is specified for PDF, get the last N pages
            if tail:
                target_pages = range(max(1, total_pages - tail + 1), total_pages + 1)
            else:
                target_pages = pages if pages else range(1, total_pages + 1)
            
            for p_num in target_pages:
                if 1 <= p_num <= total_pages:
                    if total_pages > 1:
                        text += f"--- Page {p_num} ---\n"
                    text += reader.pages[p_num - 1].extract_text() + "\n"
            return text, total_pages, "pages"
        except Exception as e:
            return f"[Error reading PDF {filepath}: {str(e)}]", 0, "pages"
    else:
        try:
            # Efficient buffered line count
            total_lines = 0
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    total_lines += chunk.count(b'\n')
            
            # If the file doesn't end in a newline but has content, it's still a line
            if total_lines == 0 and os.path.getsize(filepath) > 0:
                total_lines = 1

            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                if tail:
                    # For tail, we calculate the offset
                    start_line = max(0, total_lines - tail)
                    gen = islice(f, start_line, None)
                else:
                    stop = (offset + limit) if limit else None
                    gen = islice(f, offset, stop)
                return "".join(gen), total_lines, "lines"
        except Exception as e:
            return f"[Error reading file {filepath}: {str(e)}]", 0, "lines"

async def run_shell_command(command: str) -> str:
    """
    Executes a shell command. 
    NOTE: This is executed with the permissions of the current process.
    """
    logger.info(f"Executing command: {command}")
    
    try:
        # Create subprocess in the current event loop
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        active_subprocesses.add(process)
        
        try:
            # Use asyncio.wait_for for timeout
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
            output = f"STDOUT:\n{stdout.decode()}\nSTDERR:\n{stderr.decode()}"
            if process.returncode != 0:
                output += f"\nReturn Code: {process.returncode}"
            return output
        except asyncio.TimeoutError:
            process.terminate()
            await process.wait()
            return "Error: Command timed out after 30 seconds."
        finally:
            active_subprocesses.discard(process)
            
    except asyncio.CancelledError:
        logger.info(f"Shell command cancelled: {command}")
        # Terminate process if cancelled
        for p in list(active_subprocesses):
            if p.returncode is None:
                p.terminate()
        raise
    except Exception as e:
        logger.error(f"Execution Error: {e}")
        return f"Error executing command: {str(e)}"

async def complete_plan_step(step_index: int) -> str:
    """
    Marks a specific step in the local execution plan as completed.
    
    Args:
        step_index: The 1-based index of the step to mark as complete (e.g., 1 for the first step).
    """
    if not os.path.exists(PLAN_FILE):
        return f"Error: Plan file {PLAN_FILE} not found. Are you in Planning Mode?"
        
    try:
        with open(PLAN_FILE, "r") as f:
            lines = f.readlines()
            
        count = 0
        updated = False
        for i, line in enumerate(lines):
            # Matches "- [ ]" or "- [x]"
            if "- [" in line and ("] " in line):
                count += 1
                if count == step_index:
                    if "- [ ]" in line:
                        lines[i] = line.replace("- [ ]", "- [x]")
                        updated = True
                    else:
                        return f"Step {step_index} is already marked as complete."
                    break
        
        if not updated:
            return f"Error: Could not find step {step_index} in the plan."
            
        with open(PLAN_FILE, "w") as f:
            f.writelines(lines)
            
        return f"Successfully marked step {step_index} as complete in {PLAN_FILE}."
    except Exception as e:
        logger.error(f"Error updating plan: {e}")
        return f"Error updating plan: {str(e)}"

async def write_file_tool(filepath: str, content: str) -> str:
    """
    Writes content to a file.
    """
    logger.info(f"Writing to {filepath}")
    try:
        # Create subdirectories if they don't exist
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        logger.error(f"File Write Error: {e}")
        return f"Error writing file: {str(e)}"

async def read_file_tool(filepath: str, offset: int = 0, limit: int = None, pages: list[int] = None) -> str:
    """
    Reads content from a file.
    
    Args:
        filepath: Path to the file.
        offset: Line number to start reading from (for text files).
        limit: Number of lines to read (for text files).
        pages: List of page numbers to read (for PDF files, 1-indexed).
    """
    logger.info(f"Reading from {filepath} (offset={offset}, limit={limit}, pages={pages})")
    if not os.path.exists(filepath):
        return f"Error: File '{filepath}' not found."
        
    try:
        res, total, unit = _read_local_file(filepath, offset=offset, limit=limit, pages=pages)
        return res
    except Exception as e:
        logger.error(f"File Read Error: {e}")
        return f"Error reading file: {str(e)}"
