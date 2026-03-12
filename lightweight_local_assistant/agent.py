import os
import json
import logging
import asyncio
import ollama
from .models import get_model_info, LOCAL_WORKER_MODEL, OLLAMA_BASE_URL
from .tools import _read_local_file, run_shell_command, complete_plan_step, write_file_tool
from .security import cleanup_resources, PLAN_FILE

logger = logging.getLogger(__name__)

class LocalAssistant:
    def __init__(self, model: str = LOCAL_WORKER_MODEL, ollama_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.ollama_url = ollama_url
        self.client = ollama.AsyncClient(host=self.ollama_url)

    async def ask(self, prompt: str, local_file_context: list[str] = None, use_plan: bool = False, num_ctx: int = 32768, max_turns: int = None) -> str:
        """
        Interacts with the local assistant to perform a task.
        """
        logger.info(f"Lightweight Local Assistant: Initializing iterative loop with model {self.model} (Planning Mode: {use_plan}, Context Window: {num_ctx}, Max Turns: {max_turns})")
        
        # Set default turn limits
        if use_plan:
            if max_turns is not None and max_turns < 30:
                return f"Warning: Planning mode requires at least 30 turns for reliable execution. You specified {max_turns}. Please increase 'max_turns' or use Direct Execution mode."
            MAX_TURNS = max_turns if max_turns is not None else 40
        else:
            MAX_TURNS = max_turns if max_turns is not None else 20

        # Discovery Phase: Get model context limit
        try:
            raw_info = await get_model_info(self.model)
            model_meta = json.loads(raw_info)
            native_ctx = model_meta.get('context_length')
            model_ctx_msg = f"LOCAL MODEL CAPABILITIES: Model '{self.model}' reports a native context limit of {native_ctx} tokens.\n"
        except Exception:
            model_ctx_msg = ""

        # Define available tools
        tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'read_file',
                    'description': 'Read content from one or more files.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'filepaths': {
                                'type': 'array', 
                                'items': {'type': 'string'}, 
                                'description': 'List of paths to the files to read.'
                            },
                            'offset': {'type': 'integer', 'description': 'Line number to start reading from (text files).'},
                            'limit': {'type': 'integer', 'description': 'Number of lines to read (text files).'},
                            'tail': {'type': 'integer', 'description': 'Number of lines/pages to read from the END of the file.'},
                            'pages': {
                                'type': 'array', 
                                'items': {'type': 'integer'}, 
                                'description': 'List of page numbers to read (PDF files, 1-indexed).'
                            },
                        },
                        'required': ['filepaths'],
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
                    'description': 'Execute one or more shell commands.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'command': {'type': 'string', 'description': 'A single shell command to execute.'},
                            'commands': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': 'A list of shell commands to execute sequentially.'
                            },
                        },
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'request_clarification',
                    'description': 'Ask the user for missing information or to resolve an ambiguity.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'question': {'type': 'string', 'description': 'The question to ask the user.'},
                        },
                        'required': ['question'],
                    },
                },
            },
        ]

        async def execute_tool(func_name, args):
            # A conservative estimation: 3.5 characters per token. 
            # num_ctx is in tokens.
            safe_char_limit = int(num_ctx * 0.25 * 3.5)
            # Cap at a reasonable maximum to prevent massive strings being returned even if num_ctx is huge
            absolute_max_chars = 64000
            char_limit = min(safe_char_limit, absolute_max_chars)

            def truncate_with_meta(text, unit_name, total_units):
                if len(text) > char_limit:
                    truncated = text[:char_limit]
                    footer = (
                        f"\n\n[WARNING: Output truncated due to context limits ({len(text)} chars > {char_limit} limit). "
                        f"This file has {total_units} {unit_name} total. "
                        f"Use 'offset'/'limit' or 'pages' to read the next segment.]"
                    )
                    return truncated + footer
                return text

            if func_name == "read_file":
                fps = args.get('filepaths')
                if not fps:
                    fps = [args.get('filepath', "")]
                
                results = []
                for fp in fps:
                    res, total, unit = _read_local_file(
                        fp,
                        offset=args.get('offset', 0),
                        limit=args.get('limit'),
                        pages=args.get('pages'),
                        tail=args.get('tail')
                    )
                    res = truncate_with_meta(res, unit, total)
                    # Wrap each file's output with a header including total units
                    results.append(f"--- FILE: {fp} ({total} {unit} total) ---\n{res}\n")
                return "\n".join(results)

            elif func_name == "write_file":
                fp = args.get('filepath', "")
                res = await write_file_tool(fp, args.get('content', ""))
                return res
            elif func_name == "complete_plan_step":
                return await complete_plan_step(args.get('step_index', 0))
            elif func_name == "run_shell_command":
                cmds = args.get('commands')
                if not cmds:
                    cmds = [args.get('command', "")]
                
                results = []
                for cmd in cmds:
                    if not cmd: continue
                    res = await run_shell_command(cmd)
                    # Count lines for metadata in shell command context
                    total_lines = res.count('\n') + 1
                    res = truncate_with_meta(res, "lines", total_lines)
                    if len(cmds) > 1:
                        results.append(f"--- COMMAND: {cmd} ---\n{res}\n")
                    else:
                        results.append(res)
                return "\n".join(results)
            elif func_name == "request_clarification":
                return f"CLARIFICATION_REQUIRED: {args.get('question', 'What do you need?')}"
            return f"Unknown tool: {func_name}"

        try:
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
                planning_tools = [t for t in tools if t['function']['name'] in ['write_file', 'read_file', 'run_shell_command', 'request_clarification']]

                for i in range(10): # Max 10 turns to explore + write a plan
                    logger.info(f"Planning Turn {i+1}/10")
                    response = await self.client.chat(
                        model=self.model, 
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
                        
                        if result.startswith("CLARIFICATION_REQUIRED:"):
                            return result

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
                "You are the 'Lightweight Local Assistant', a secure autonomous reasoning agent running on the user's computer. "
                "You act as the local 'hands' for a Cloud-based Brain (Gemini). You process sensitive data "
                "locally to ensure privacy. You have access to the current project directory.\n\n"
                
                f"{model_ctx_msg}"
                "PRIORITY 1: PRECISION & ADHERENCE\n"
                "Always prioritize the user's specific request. If the user asks for extraction (e.g., 'Who are the authors?'), "
                "provide ONLY that information. Do NOT provide high-level summaries unless explicitly asked.\n\n"

                "CAPABILITIES & TOOLS:\n"
                "- 'run_shell_command': Run one or more shell commands. Use standard OS utilities appropriate for the current platform (e.g., grep, find, ls, dir, findstr, etc.).\n"
                "  * You can pass a list of 'commands' to execute multiple operations in a single turn.\n"
                "  * EFFICIENT FILTERING: Use 'grep', 'sed', 'awk', or 'column' to extract only the necessary data locally. This is faster and prevents hitting the Context Guard's truncation limits.\n"
                "- 'read_file': Read text OR PDF files. \n"
                "  * You can pass a list of 'filepaths' to read multiple files at once.\n"
                "  * Use 'offset' and 'limit' (lines) for text files to avoid context overload.\n"
                "  * Use 'tail' (int) to read the last N lines/pages. This is great for logs or the end of documents.\n"
                "  * Use 'pages' (list of ints) for PDFs. For metadata, usually only Page 1 is needed.\n"
                "  * WARNING: Large outputs will be automatically truncated. Use segmentation to read long files.\n"
                "- 'write_file': Save results or summaries to a file.\n"
                "- 'get_model_info': Inspect local model details (context limit, etc.) if needed.\n"
                "- 'request_clarification': If you are stuck because the user's request is ambiguous or missing info, "
                "use this to ask the user a specific question. This will pause your execution.\n\n"
                
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

            turn_count = 0
            has_reflected = False
            plan_nudge_count = 0
            last_tool_was_work = False
            
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
                
                response = await self.client.chat(
                    model=self.model, 
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
                        logger.info("Lightweight Local Assistant: Performing self-reflection check...")
                        
                        reflection_prompt = (
                            f"Self-Correction Check: Review your work. Did you fully answer the user's request: '{prompt}'? "
                            "If you need to do more work (e.g. read more files, run more commands), answer 'incomplete'. "
                            "Respond ONLY in JSON format: {\"status\": \"complete\"} OR {\"status\": \"incomplete\", \"reason\": \"<what is missing>\"}."
                        )
                        
                        # Create a temporary message history for the reflection (don't pollute the main history yet)
                        reflection_messages = messages + [{'role': 'system', 'content': reflection_prompt}]
                        
                        try:
                            # Force JSON mode for the reflection
                            ref_response = await self.client.chat(
                                model=self.model, 
                                messages=reflection_messages, 
                                format='json',
                                options={'num_ctx': num_ctx}
                            )
                            ref_content = ref_response.get('message', {}).get('content', '{}')
                            ref_data = json.loads(ref_content)
                            
                            if ref_data.get('status') == 'incomplete':
                                reason = ref_data.get('reason', 'No reason provided')
                                logger.info(f"Lightweight Local Assistant Reflection: Incomplete ({reason}). Extending turns.")
                                
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
                                logger.info("Lightweight Local Assistant Reflection: Task complete.")
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
                                logger.info(f"Lightweight Local Assistant: Intercepting finish. {plan_content.count('- [ ]')} steps still unchecked. Nudging...")
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

                    # --- CLARIFICATION BREAK ---
                    if result.startswith("CLARIFICATION_REQUIRED:"):
                        return result
            
            return f"Lightweight Local Assistant reached the maximum turn limit ({MAX_TURNS}) without finishing the task."

        except asyncio.CancelledError:
            logger.info("ask was cancelled. Cleaning up...")
            cleanup_resources()
            raise
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Lightweight Local Assistant Error: {e}\n{error_trace}")
            return f"Error in local assistant: {str(e)}"
        finally:
            cleanup_resources()
