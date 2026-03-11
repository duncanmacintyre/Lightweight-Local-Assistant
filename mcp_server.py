import os
import json
import logging
import subprocess
import ctypes
import sys
import ollama
from itertools import islice
from pypdf import PdfReader
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Local Assistant for Gemini CLI")

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_WORKER_MODEL = os.getenv("LOCAL_WORKER_MODEL", "qwen3-coder:30b")
PLAN_FILE = ".gemini/local_plan.md"

def is_sandboxed() -> bool:
    """Checks if the process is running within a macOS sandbox (seatbelt)."""
    try:
        # libsandbox is available on macOS
        libsandbox = ctypes.CDLL("/usr/lib/libsandbox.1.dylib")
        # sandbox_check(pid, entity, flags) returns 1 if sandboxed
        return libsandbox.sandbox_check(os.getpid(), None, 0) == 1
    except Exception:
        return False

def _read_local_file(filepath: str, offset: int = 0, limit: int = None, pages: list[int] = None) -> str:
    """Helper to read text or PDF files locally with support for partial reading."""
    if not os.path.exists(filepath):
        return f"[Error: File {filepath} not found]"

    if filepath.lower().endswith(".pdf"):
        try:
            reader = PdfReader(filepath)
            total_pages = len(reader.pages)
            text = ""
            
            # If pages are specified, use them (1-indexed for user convenience)
            target_pages = pages if pages else range(1, total_pages + 1)
            
            for p_num in target_pages:
                if 1 <= p_num <= total_pages:
                    # Show header if specifically requested pages OR if multi-page document
                    if pages or total_pages > 1:
                        text += f"--- Page {p_num} ---\n"
                    text += reader.pages[p_num - 1].extract_text() + "\n"
                else:
                    text += f"[Warning: Page {p_num} out of range (Total pages: {total_pages})]\n"
            return text
        except Exception as e:
            return f"[Error reading PDF {filepath}: {str(e)}]"
    else:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                # Use offset/limit for text files (lines) efficiently with islice
                stop = (offset + limit) if limit else None
                gen = islice(f, offset, stop)
                return "".join(gen)
        except Exception as e:
            return f"[Error reading file {filepath}: {str(e)}]"

@mcp.tool()
async def list_local_models() -> str:
    """
    Lists the available local Ollama models that can be used with ask_local_assistant.
    """
    try:
        models_info = ollama.list()
        # The ollama library returns a ModelList object with a list of Model objects
        # Each Model object has a 'model' attribute for its name.
        models = []
        if hasattr(models_info, 'models'):
            models = [m.model for m in models_info.models]
        elif isinstance(models_info, dict) and 'models' in models_info:
            models = [m.get('name') or m.get('model') for m in models_info['models']]
            
        if not models:
            return "No local models found in Ollama."
        return "Available local models:\n- " + "\n- ".join(models)
    except Exception as e:
        logger.error(f"Error listing Ollama models: {e}")
        return f"Error listing models: {str(e)}"

@mcp.tool()
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

@mcp.tool()
async def ask_local_assistant(prompt: str, local_file_context: list[str] = None, model: str = LOCAL_WORKER_MODEL, use_plan: bool = False, num_ctx: int = 32768) -> str:
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
    """
    logger.info(f"Local Agent: Initializing iterative loop with model {model} (Planning Mode: {use_plan}, Context Window: {num_ctx})")
    
    # Define available tools
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'read_file',
                'description': 'Read content from a file.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'filepath': {'type': 'string', 'description': 'Path to the file to read.'},
                        'offset': {'type': 'integer', 'description': 'Line number to start reading from (text files).'},
                        'limit': {'type': 'integer', 'description': 'Number of lines to read (text files).'},
                        'pages': {
                            'type': 'array', 
                            'items': {'type': 'integer'}, 
                            'description': 'List of page numbers to read (PDF files, 1-indexed).'
                        },
                    },
                    'required': ['filepath'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'write_file',
                'description': 'Write content to a file.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'filepath': {'type': 'string', 'description': 'Path to the file to save.'},
                        'content': {'type': 'string', 'description': 'Content to write.'},
                    },
                    'required': ['filepath', 'content'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'complete_plan_step',
                'description': 'Mark a step in the execution plan as completed.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'step_index': {'type': 'integer', 'description': 'The 1-based index of the step to mark as complete.'},
                    },
                    'required': ['step_index'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'run_shell_command',
                'description': 'Execute a shell command (zsh).',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'command': {'type': 'string', 'description': 'The zsh command to execute.'},
                    },
                    'required': ['command'],
                },
            },
        },
    ]

    async def execute_tool(func_name, args):
        if func_name == "read_file":
            return _read_local_file(
                args.get('filepath', ""),
                offset=args.get('offset', 0),
                limit=args.get('limit'),
                pages=args.get('pages')
            )
        elif func_name == "write_file":
            fp = args.get('filepath', "")
            try:
                if os.path.dirname(fp):
                    os.makedirs(os.path.dirname(fp), exist_ok=True)
                with open(fp, "w") as f:
                    f.write(args.get('content', ""))
                return f"Successfully wrote to {fp}"
            except Exception as e:
                return f"Error writing file: {str(e)}"
        elif func_name == "complete_plan_step":
            return await complete_plan_step(args.get('step_index', 0))
        elif func_name == "run_shell_command":
            return await run_shell_command(args.get('command', ""))
        return f"Unknown tool: {func_name}"

    # --- PHASE 1: FORCED PLANNING (If enabled) ---
    if use_plan:
        logger.info("Local Agent: Entering Planning Phase...")
        
        # 1. Clear old plan
        if os.path.exists(PLAN_FILE):
            os.remove(PLAN_FILE)
            
        # 2. Planning Loop
        plan_system_msg = (
            "You are a Senior Technical Planner. Your goal is to create a detailed, step-by-step implementation plan "
            "for the user's request. To create a grounded and accurate plan, you should first explore the project "
            "using 'run_shell_command' (e.g., 'ls -R', 'grep') and 'read_file'.\n\n"
            f"Once you understand the context, you MUST call the 'write_file' tool to save your plan to the EXACT path: '{PLAN_FILE}'.\n"
            "Format your plan as a Markdown checklist:\n"
            "- [ ] Step 1: <Description>\n"
            "- [ ] Step 2: <Description>\n\n"
            "Do NOT execute implementation steps yet. Only use discovery tools to inform your plan. "
            "After writing the plan file, stop."
        )
        
        plan_messages = [{'role': 'system', 'content': plan_system_msg}, {'role': 'user', 'content': prompt}]
        
        plan_created = False
        attempted_paths = []
        planning_tools = [t for t in tools if t['function']['name'] in ['write_file', 'read_file', 'run_shell_command']]

        for i in range(10): # Max 10 turns to explore + write a plan
            logger.info(f"Planning Turn {i+1}/10")
            response = ollama.chat(
                model=model, 
                messages=plan_messages, 
                tools=planning_tools,
                options={'num_ctx': num_ctx}
            )
            msg = response.get('message', {})
            logger.info(f"Planning Response: {msg}")
            plan_messages.append(msg)
            
            tool_calls = msg.get('tool_calls')
            if not tool_calls:
                if plan_created:
                    break
                else:
                    plan_messages.append({'role': 'system', 'content': f"You must write the plan to '{PLAN_FILE}' using 'write_file' before finishing."})
                    continue

            for tc in tool_calls:
                func_name = tc['function']['name']
                args = json.loads(tc['function']['arguments']) if isinstance(tc['function']['arguments'], str) else tc['function']['arguments']
                
                result = await execute_tool(func_name, args)
                plan_messages.append({'role': 'tool', 'content': result, 'name': func_name})
                
                if func_name == "write_file":
                    fp = args.get('filepath', "")
                    attempted_paths.append(fp)
                    if fp.endswith("local_plan.md"):
                        plan_created = True
            
            if plan_created and not any(tc['function']['name'] != 'write_file' for tc in tool_calls):
                 # If we just wrote the plan and didn't do anything else, we might be done
                 pass 

        if not plan_created:
            return f"Error: Agent failed to generate a plan file ({PLAN_FILE}) in Planning Mode. Attempted paths: {attempted_paths}"
        
        logger.info("Local Agent: Plan created. Entering Execution Phase.")

    # --- PHASE 2: EXECUTION LOOP (Main) ---
    
    base_system_msg = (
        "IDENTITY & CONTEXT:\n"
        "You are the 'Local Assistant', a secure autonomous reasoning agent running on the user's computer. "
        "You act as the local 'hands' for a Cloud-based Brain (Gemini). You process sensitive data "
        "locally to ensure privacy. You have access to the current project directory.\n\n"
        
        "PRIORITY 1: PRECISION & ADHERENCE\n"
        "Always prioritize the user's specific request. If the user asks for extraction (e.g., 'Who are the authors?'), "
        "provide ONLY that information. Do NOT provide high-level summaries unless explicitly asked.\n\n"

        "CAPABILITIES & TOOLS:\n"
        "- 'run_shell_command': Run zsh commands. You have access to standard macOS utilities: grep, rg, sed, awk, find, ls, cat, etc.\n"
        "  * Use 'rg' (ripgrep) for high-performance text searching.\n"
        "  * Use 'awk' or 'column' for processing structured/tabular text.\n"
        "- 'read_file': Read text OR PDF files. PDFs are automatically converted to text for you.\n"
        "  * Use 'offset' and 'limit' (lines) for text files to avoid context overload.\n"
        "  * Use 'pages' (list of ints) for PDFs. For metadata like authors/titles, usually only Page 1 is needed.\n"
        "- 'write_file': Save results or summaries to a file.\n\n"
        
        "CONSTRAINTS:\n"
        "- Stay focused on the task.\n"
        "- If you are stuck after several attempts, report the specific technical blocker.\n"
        "- When finished, provide the specific answer requested.\n"
        "- Anonymize PII (Personally Identifiable Information) in your final response unless the user explicitly requested that specific data.\n\n"
    )

    if use_plan:
        execution_instructions = (
            "OPERATING MODE: PLAN EXECUTION\n"
            "You are executing a pre-defined plan. The current plan status will be provided to you every turn.\n"
            "YOUR JOB:\n"
            "1. Review the 'CURRENT PLAN STATE'.\n"
            "2. Execute the next unchecked step ([ ]).\n"
            "3. VERY IMPORTANT: After finishing the work for a step, you MUST call 'complete_plan_step' to mark it as [x].\n"
            "   - You MUST pass the 1-based index of the step you just completed.\n"
            "   - Do NOT try to edit the plan file manually using 'write_file'. Use 'complete_plan_step'.\n"
            "4. Do not deviate from the plan."
        )
        system_msg = base_system_msg + execution_instructions
    else:
        system_msg = base_system_msg + (
            "OPERATING MODE: DIRECT EXECUTION\n"
            "You MUST solve tasks iteratively using a 'Think-Act-Observe' cycle:\n"
            "1. THOUGHT: First, explain your reasoning and plan in the text response.\n"
            "2. ACTION: Then, call exactly ONE tool to execute a step of your plan.\n"
            "3. OBSERVATION: Review the tool's output. If it failed, diagnose why and try a different approach.\n\n"
        )
    
    messages = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': prompt}]
    if local_file_context:
        ctx_msg = f"Available files: {', '.join(local_file_context)}."
        messages.insert(1, {'role': 'system', 'content': ctx_msg})

    # Increase turns for planning mode as it involves overhead steps (update plan)
    MAX_TURNS = 30 if use_plan else 15
    turn_count = 0
    has_reflected = False
    plan_nudge_count = 0
    last_tool_was_work = False
    
    try:
        while turn_count < MAX_TURNS:
            turn_count += 1
            remaining = MAX_TURNS - turn_count
            logger.info(f"Local Agent Loop: Turn {turn_count}/{MAX_TURNS}")
            
            # --- CONTEXT INJECTION (The "Python-Led" Logic) ---
            loop_system_msg = f"REMINDER: {prompt}\n"
            loop_system_msg += f"PROGRESS: Turn {turn_count}/{MAX_TURNS} ({remaining} remaining).\n"
            
            if use_plan and os.path.exists(PLAN_FILE):
                try:
                    with open(PLAN_FILE, 'r') as f:
                        plan_content = f.read()
                    loop_system_msg += f"\n=== CURRENT PLAN STATE ({PLAN_FILE}) ===\n{plan_content}\n===============================\n"
                    
                    if last_tool_was_work:
                        loop_system_msg += "The previous action appeared successful. If a plan step is now complete, you should update the plan using 'complete_plan_step' before proceeding.\n"
                    else:
                        loop_system_msg += "INSTRUCTION: Execute the next [ ] step. Then call 'complete_plan_step' with the correct index to mark it [x].\n"

                    # Detect if plan is fully complete to nudge finishing
                    if "- [ ]" not in plan_content and "- [x]" in plan_content:
                        loop_system_msg += "Plan appears complete. You can now provide your final response.\n"
                except Exception as e:
                    loop_system_msg += f"\n[Warning: Could not read plan file: {e}]\n"
            else:
                loop_system_msg += " Focus on this task.\n"
            
            current_messages = messages + [{'role': 'system', 'content': loop_system_msg}]
            
            response = ollama.chat(
                model=model, 
                messages=current_messages, 
                tools=tools,
                options={'num_ctx': num_ctx}
            )
            assistant_msg = response.get('message', {})
            messages.append(assistant_msg)
            
            tool_calls = assistant_msg.get('tool_calls')
            if not tool_calls:
                # --- REFLECTION STEP ---
                if not has_reflected:
                    logger.info("Local Agent: Performing self-reflection check...")
                    
                    reflection_prompt = (
                        f"Self-Correction Check: Review your work. Did you fully answer the user's request: '{prompt}'? "
                        "If you need to do more work (e.g. read more files, run more commands), answer 'incomplete'. "
                        "Respond ONLY in JSON format: {\"status\": \"complete\"} OR {\"status\": \"incomplete\", \"reason\": \"<what is missing>\"}."
                    )
                    
                    # Create a temporary message history for the reflection (don't pollute the main history yet)
                    reflection_messages = messages + [{'role': 'system', 'content': reflection_prompt}]
                    
                    try:
                        # Force JSON mode for the reflection
                        ref_response = ollama.chat(
                            model=model, 
                            messages=reflection_messages, 
                            format='json',
                            options={'num_ctx': num_ctx}
                        )
                        ref_content = ref_response.get('message', {}).get('content', '{}')
                        ref_data = json.loads(ref_content)
                        
                        if ref_data.get('status') == 'incomplete':
                            reason = ref_data.get('reason', 'No reason provided')
                            logger.info(f"Local Agent Reflection: Incomplete ({reason}). Extending turns.")
                            
                            has_reflected = True
                            MAX_TURNS += 5
                            
                            # Inject the realization back into the main history so the agent acts on it
                            messages.append({
                                'role': 'system', 
                                'content': f"SELF-CORRECTION: You acknowledged the task is incomplete because: '{reason}'. "
                                           f"You have been granted 5 extra turns. Please continue working to resolve this."
                            })
                            continue # Resume the loop to allow more tool calls
                            
                        else:
                            logger.info("Local Agent Reflection: Task complete.")
                            has_reflected = True
                            
                    except Exception as e:
                        logger.warning(f"Reflection failed (parsing error or model limitation): {e}. Assuming complete.")
                
                # --- PLAN VALIDATION NUDGE ---
                if use_plan and os.path.exists(PLAN_FILE):
                    try:
                        with open(PLAN_FILE, 'r') as f:
                            plan_content = f.read()
                        if "- [ ]" in plan_content and plan_nudge_count < 2:
                            plan_nudge_count += 1
                            logger.info(f"Local Agent: Intercepting finish. {plan_content.count('- [ ]')} steps still unchecked. Nudging...")
                            messages.append({
                                'role': 'system', 
                                'content': "You have attempted to finish, but some steps in the plan are still marked as incomplete [ ]. "
                                           "If these steps are finished, please update the plan using 'complete_plan_step'. "
                                           "If they are not needed or you have a reason to skip them, please explain. "
                                           "Otherwise, continue with the remaining work."
                            })
                            continue # Re-run the loop for another turn
                    except Exception as e:
                        logger.warning(f"Plan validation failed: {e}")

                # No more tools requested and reflection/plan checks passed
                final_response = assistant_msg.get('content', "Task completed.")
                
                if use_plan and os.path.exists(PLAN_FILE):
                    try:
                        with open(PLAN_FILE, 'r') as f:
                            plan_content = f.read()
                        
                        # Check if plan was actually updated (basic check for [x])
                        if "- [x]" not in plan_content and "- [ ]" in plan_content:
                             final_response += "\n\n[Warning: Agent executed actions but failed to update the plan checklist.]"
                        
                        final_response += f"\n\n--- EXECUTION PLAN ---\n{plan_content}\n----------------------"
                        
                        # Cleanup
                        if os.path.exists(PLAN_FILE):
                            os.remove(PLAN_FILE)
                        # Try to remove the directory if empty
                        try:
                            os.rmdir(os.path.dirname(PLAN_FILE))
                        except (OSError, FileNotFoundError):
                            pass # Directory not empty or already gone
                            
                    except Exception as e:
                        logger.warning(f"Could not read/cleanup plan file: {e}")
                
                return final_response

            for tool_call in tool_calls:
                func_name = tool_call['function']['name']
                args = tool_call['function'].get('arguments', {})
                
                # Track if this turn involved actual work vs just bureaucracy
                if func_name in ["run_shell_command", "write_file"]:
                    last_tool_was_work = True
                else:
                    last_tool_was_work = False
                
                if not isinstance(args, dict):
                    try:
                        args = json.loads(str(args))
                    except:
                        args = {}
                
                result = await execute_tool(func_name, args)
                messages.append({'role': 'tool', 'content': result, 'name': func_name})
        
        return f"Local Agent reached the maximum turn limit ({MAX_TURNS}) without finishing the task."
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Local Agent Error: {e}\n{error_trace}")
        return f"Error in local agent: {str(e)}"

@mcp.tool()
async def run_shell_command(command: str) -> str:
    """
    Executes a shell command. 
    NOTE: This is executed with the permissions of the Gemini CLI process.
    """
    logger.info(f"Executing command: {command}")
    
    try:
        result = subprocess.run(
            ["zsh", "-c", command],
            capture_output=True,
            text=True,
            timeout=30 # Safety timeout
        )
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\nReturn Code: {result.returncode}"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        logger.error(f"Execution Error: {e}")
        return f"Error executing command: {str(e)}"

@mcp.tool()
async def write_file(filepath: str, content: str) -> str:
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
    logger.info(f"Reading from {filepath} (offset={offset}, limit={limit}, pages={pages})")
    if not os.path.exists(filepath):
        return f"Error: File '{filepath}' not found."
        
    try:
        return _read_local_file(filepath, offset=offset, limit=limit, pages=pages)
    except Exception as e:
        logger.error(f"File Read Error: {e}")
        return f"Error reading file: {str(e)}"

if __name__ == "__main__":
    if not is_sandboxed():
        print("CRITICAL ERROR: Local Assistant MUST be run within a Gemini CLI sandbox.")
        print("Please run gemini with the -s or --sandbox flag.")
        sys.exit(1)

    mcp.run(transport='stdio')