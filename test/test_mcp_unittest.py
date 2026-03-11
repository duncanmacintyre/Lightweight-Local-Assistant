import unittest
import os
import shutil
import tempfile
import sys
import json
from unittest.mock import MagicMock, patch, mock_open
import mcp_server

class TestMCPServer(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Create a temporary directory for each test
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    # --- Sandbox Detection Tests ---

    def test_is_sandboxed_true(self):
        with patch("ctypes.CDLL") as mock_cdll:
            mock_libsandbox = MagicMock()
            mock_libsandbox.sandbox_check.return_value = 1
            mock_cdll.return_value = mock_libsandbox
            from mcp_server import is_sandboxed
            self.assertTrue(is_sandboxed())

    def test_is_sandboxed_false(self):
        with patch("ctypes.CDLL") as mock_cdll:
            mock_libsandbox = MagicMock()
            mock_libsandbox.sandbox_check.return_value = 0
            mock_cdll.return_value = mock_libsandbox
            from mcp_server import is_sandboxed
            self.assertFalse(is_sandboxed())

    def test_is_sandboxed_exception(self):
        with patch("ctypes.CDLL", side_effect=Exception("Not macOS")):
            from mcp_server import is_sandboxed
            self.assertFalse(is_sandboxed())

    # --- File Reading Tests ---

    def test_read_local_file_text(self):
        test_file = os.path.join(self.test_dir, "hello.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("hello world")
        
        from mcp_server import _read_local_file as read_fn
        self.assertEqual(read_fn(test_file), "hello world")

    def test_read_local_file_text_partial(self):
        test_file = os.path.join(self.test_dir, "lines.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("line1\nline2\nline3\nline4\n")
        
        from mcp_server import _read_local_file as read_fn
        self.assertEqual(read_fn(test_file, offset=2), "line3\nline4\n")
        self.assertEqual(read_fn(test_file, offset=1, limit=2), "line2\nline3\n")

    @patch("mcp_server.PdfReader")
    def test_read_local_file_pdf_partial(self, mock_pdf_reader):
        test_pdf = os.path.join(self.test_dir, "test.pdf")
        with open(test_pdf, "w") as f:
            f.write("pdf-content")
        
        p1 = MagicMock()
        p1.extract_text.return_value = "page 1 text"
        p2 = MagicMock()
        p2.extract_text.return_value = "page 2 text"
        
        mock_pdf_reader.return_value.pages = [p1, p2]
        
        from mcp_server import _read_local_file as read_fn
        result = read_fn(test_pdf, pages=[2])
        self.assertIn("page 2 text", result)
        self.assertNotIn("page 1 text", result)
        self.assertIn("--- Page 2 ---", result)

    def test_read_local_file_not_found(self):
        from mcp_server import _read_local_file as read_fn
        result = read_fn("missing_file_xyz.txt")
        self.assertIn("not found", result)

    @patch("mcp_server.PdfReader")
    def test_read_local_file_pdf_corrupt(self, mock_pdf_reader):
        bad_pdf = os.path.join(self.test_dir, "corrupt.pdf")
        with open(bad_pdf, "w") as f:
            f.write("not-a-pdf")
        mock_pdf_reader.side_effect = Exception("Invalid PDF format")
        
        from mcp_server import _read_local_file as read_fn
        result = read_fn(bad_pdf)
        self.assertIn("Error reading PDF", result)

    # --- Tool Execution Tests ---

    async def test_write_file(self):
        from mcp_server import write_file
        target = os.path.join(self.test_dir, "new.txt")
        result = await write_file(target, "content")
        self.assertIn("Successfully wrote", result)
        with open(target, "r") as f:
            self.assertEqual(f.read(), "content")

    async def test_run_shell_command_success(self):
        from mcp_server import run_shell_command
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "ls output"
            mock_run.return_value.stderr = ""
            mock_run.return_value.returncode = 0
            
            result = await run_shell_command("ls")
            self.assertIn("ls output", result)
            args = mock_run.call_args[0][0]
            self.assertEqual(args, ["zsh", "-c", "ls"])

    async def test_run_shell_command_error(self):
        from mcp_server import run_shell_command
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = "permission denied"
            mock_run.return_value.returncode = 1
            
            result = await run_shell_command("invalid_cmd")
            self.assertIn("Return Code: 1", result)
            self.assertIn("permission denied", result)

    # --- Agent Loop Logic (Mocked Ollama) ---

    @patch("ollama.chat")
    async def test_ask_local_assistant_planning_mode(self, mock_chat):
        from mcp_server import ask_local_assistant
        
        mock_chat.side_effect = [
            # Phase 1: Planning Turn 1 (Write Plan)
            {
                'message': {
                    'role': 'assistant',
                    'tool_calls': [{
                        'function': {
                            'name': 'write_file',
                            'arguments': json.dumps({'filepath': '.gemini/local_plan.md', 'content': '- [ ] Step A'})
                        }
                    }]
                }
            },
            # Phase 1: Planning Turn 2 (Finished planning)
            {'message': {'role': 'assistant', 'content': 'Plan created.'}},
            # Phase 2: Execution Turn 1 (Execute)
            {'message': {'role': 'assistant', 'content': 'Executed.'}},
            # Phase 2: Reflection
            {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}},
            # Phase 2: Nudge Turn 1
            {'message': {'role': 'assistant', 'content': 'Acknowledged nudge 1.'}},
            # Phase 2: Nudge Turn 2
            {'message': {'role': 'assistant', 'content': 'Acknowledged nudge 2.'}}
        ]
        
        # We need to mock os.path.exists and open to simulate the plan file handling
        with patch("os.path.exists") as mock_exists, \
             patch("builtins.open", mock_open(read_data="- [ ] Step A")) as mock_file, \
             patch("os.remove") as mock_remove:
             
            # Return False for the first call (plan not created), then True for all subsequent calls
            mock_exists.side_effect = lambda path: path != ".gemini/local_plan.md" or mock_exists.call_count > 1
            
            result = await ask_local_assistant("Deep work", use_plan=True)
            
            # Verify Planning Prompt
            plan_msg = mock_chat.call_args_list[0][1]['messages'][0]['content']
            self.assertIn("Senior Technical Planner", plan_msg)
            self.assertIn("Do NOT execute implementation steps yet", plan_msg)
            self.assertEqual(mock_chat.call_args_list[0][1]['options'], {'num_ctx': 32768})
            
            # Verify Execution Prompt (Plan Injection)
            # Planning took 2 turns (one for tool call, one for content-only finish)
            exec_context_msg = mock_chat.call_args_list[2][1]['messages'][-1]['content']
            self.assertIn("CURRENT PLAN STATE", exec_context_msg)
            self.assertEqual(mock_chat.call_args_list[2][1]['options'], {'num_ctx': 32768})
            
            # Verify Final Output contains the plan
            self.assertIn("--- EXECUTION PLAN ---", result)
            self.assertIn("- [ ] Step A", result)

    @patch("ollama.chat")
    async def test_ask_local_assistant_planning_mode_warning(self, mock_chat):
        from mcp_server import ask_local_assistant
        
        mock_chat.side_effect = [
            # Phase 1: Planning Turn 1 (Write Plan)
            {
                'message': {
                    'role': 'assistant',
                    'tool_calls': [{
                        'function': {
                            'name': 'write_file',
                            'arguments': json.dumps({'filepath': '.gemini/local_plan.md', 'content': '- [ ] Step A'})
                        }
                    }]
                }
            },
            # Phase 1: Planning Turn 2 (Finished planning)
            {'message': {'role': 'assistant', 'content': 'Plan created.'}},
            # Phase 2: Execution Turn 1 (Execute - No update to plan)
            {'message': {'role': 'assistant', 'content': 'Executed without updating plan.'}},
            # Phase 2: Reflection
            {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}},
            # Phase 2: Nudge Turn 1
            {'message': {'role': 'assistant', 'content': 'Acknowledged nudge 1.'}},
            # Phase 2: Nudge Turn 2
            {'message': {'role': 'assistant', 'content': 'Acknowledged nudge 2.'}}
        ]
        
        with patch("os.path.exists") as mock_exists, \
             patch("builtins.open", mock_open(read_data="- [ ] Step A")) as mock_file, \
             patch("os.remove") as mock_remove:
             
            # Return False for the first call (plan not created), then True for all subsequent calls
            mock_exists.side_effect = lambda path: path != ".gemini/local_plan.md" or mock_exists.call_count > 1
            
            result = await ask_local_assistant("Deep work", use_plan=True)
            
            # Verify Warning
            self.assertIn("[Warning: Agent executed actions but failed to update the plan checklist.]", result)


    @patch("ollama.chat")
    async def test_ask_local_assistant_no_tool(self, mock_chat):
        from mcp_server import ask_local_assistant
        mock_chat.side_effect = [
            {'message': {'role': 'assistant', 'content': 'direct answer'}},
            {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}}
        ]
        result = await ask_local_assistant("What is 2+2?")
        self.assertEqual(result, "direct answer")

    @patch("ollama.chat")
    async def test_ask_local_assistant_multiple_turns(self, mock_chat):
        from mcp_server import ask_local_assistant
        
        test_file = os.path.join(self.test_dir, "a.txt")
        with open(test_file, "w") as f:
            f.write("source")
        
        mock_chat.side_effect = [
            {
                'message': {
                    'role': 'assistant',
                    'content': 'Thinking...',
                    'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'filepath': test_file}}}]
                }
            },
            {'message': {'role': 'assistant', 'content': 'Done!'}},
            {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}}
        ]
        
        result = await ask_local_assistant("Analyze a.txt")
        
        self.assertEqual(result, "Done!")
        self.assertEqual(mock_chat.call_count, 3)
        
        # Verify the tool call included the reminder
        # call_args_list[0] is the first call (tool request)
        last_call_messages = mock_chat.call_args_list[0][1]['messages']
        self.assertEqual(last_call_messages[-1]['role'], 'system')
        self.assertIn("REMINDER", last_call_messages[-1]['content'])

    @patch("ollama.chat")
    async def test_ask_local_assistant_turn_limit(self, mock_chat):
        from mcp_server import ask_local_assistant
        mock_chat.return_value = {
            'message': {
                'role': 'assistant',
                'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'filepath': 'dummy'}}}]
            }
        }
        
        result = await ask_local_assistant("Do something")
        self.assertIn("maximum turn limit", result)
        self.assertEqual(mock_chat.call_count, 15)

    @patch("ollama.chat")
    async def test_ask_local_assistant_agent_read_missing(self, mock_chat):
        from mcp_server import ask_local_assistant
        
        mock_chat.side_effect = [
            {
                'message': {
                    'role': 'assistant',
                    'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'filepath': 'gone.txt'}}}]
                }
            },
            {'message': {'content': 'The file is gone.'}},
            {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}}
        ]
        
        await ask_local_assistant("Read gone.txt")
        
        final_history = mock_chat.call_args_list[1][1]['messages']
        # Note: messages list grows, so we need to find the tool message in the history passed to the 2nd call
        # The history passed to chat() includes the reminder at the end.
        # But the tool message is appended to the messages list *before* chat() is called again.
        # So we look at the messages passed to the 2nd call.
        tool_msg = next(m for m in final_history if m['role'] == 'tool')
        self.assertIn("not found", tool_msg['content'])

    @patch("ollama.chat")
    async def test_ask_local_assistant_read_file_partial(self, mock_chat):
        from mcp_server import ask_local_assistant
        
        test_file = os.path.join(self.test_dir, "lines.txt")
        with open(test_file, "w") as f:
            f.write("line1\nline2\nline3\n")
        
        mock_chat.side_effect = [
            {
                'message': {
                    'role': 'assistant',
                    'tool_calls': [{
                        'function': {
                            'name': 'read_file', 
                            'arguments': {'filepath': test_file, 'offset': 1, 'limit': 1}
                        }
                    }]
                }
            },
            {'message': {'content': 'Read line 2'}},
            {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}}
        ]
        
        await ask_local_assistant("Read line 2 of lines.txt")
        
        final_history = mock_chat.call_args_list[1][1]['messages']
        tool_msg = next(m for m in final_history if m['role'] == 'tool')
        self.assertEqual(tool_msg['content'], "line2\n")

    # --- Model & Ollama Tool Tests ---

    @patch("ollama.list")
    async def test_list_local_models(self, mock_list):
        from mcp_server import list_local_models
        mock_list.return_value = {
            'models': [
                {'name': 'mistral-nemo:latest'},
                {'name': 'llama3:8b'}
            ]
        }
        
        result = await list_local_models()
        self.assertIn("Available local models:", result)
        self.assertIn("mistral-nemo:latest", result)
        self.assertIn("llama3:8b", result)

    @patch("ollama.list")
    async def test_list_local_models_empty(self, mock_list):
        from mcp_server import list_local_models
        mock_list.return_value = {'models': []}
        result = await list_local_models()
        self.assertIn("No local models found", result)

    @patch("ollama.list")
    async def test_list_local_models_error(self, mock_list):
        from mcp_server import list_local_models
        mock_list.side_effect = Exception("Ollama connection failed")
        result = await list_local_models()
        self.assertIn("Error listing models", result)

    @patch("ollama.chat")
    async def test_ask_local_assistant_custom_model(self, mock_chat):
        from mcp_server import ask_local_assistant
        mock_chat.side_effect = [
            {'message': {'content': 'using custom model'}},
            {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}}
        ]
        await ask_local_assistant("Hello", model="llama3")
        kwargs = mock_chat.call_args_list[0][1]
        self.assertEqual(kwargs['model'], "llama3")
        self.assertEqual(kwargs['options'], {'num_ctx': 32768})

if __name__ == "__main__":
    unittest.main()
