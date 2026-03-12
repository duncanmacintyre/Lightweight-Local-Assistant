import unittest
import os
import shutil
import tempfile
import sys
import json
import asyncio
from unittest.mock import MagicMock, patch, mock_open, AsyncMock
from lightweight_local_assistant import is_sandboxed, run_shell_command, write_file_tool, read_file_tool, LocalAssistant
from lightweight_local_assistant.tools import _read_local_file

class TestLocalAssistant(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # Create a temporary directory for each test
        self.test_dir = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Patch ollama.AsyncClient
        self.mock_client_patcher = patch("lightweight_local_assistant.agent.ollama.AsyncClient")
        self.mock_client_class = self.mock_client_patcher.start()
        self.mock_client = MagicMock()
        self.mock_client_class.return_value = self.mock_client
        
        self.mock_client.show = AsyncMock(return_value={"modelinfo": {"context_length": 32768}})
        self.mock_client.chat = AsyncMock()
        self.mock_client.list = AsyncMock()

    async def asyncTearDown(self):
        # Clean up the temporary directory
        os.chdir(self.old_cwd)
        shutil.rmtree(self.test_dir)
        self.mock_client_patcher.stop()

    # --- Sandbox Detection Tests ---

    def test_is_sandboxed_true(self):
        with patch("ctypes.CDLL") as mock_cdll:
            mock_libsandbox = MagicMock()
            mock_libsandbox.sandbox_check.return_value = 1
            mock_cdll.return_value = mock_libsandbox
            self.assertTrue(is_sandboxed())

    def test_is_sandboxed_false(self):
        with patch("ctypes.CDLL") as mock_cdll:
            mock_libsandbox = MagicMock()
            mock_libsandbox.sandbox_check.return_value = 0
            mock_cdll.return_value = mock_libsandbox
            self.assertFalse(is_sandboxed())

    def test_is_sandboxed_exception(self):
        with patch("ctypes.CDLL", side_effect=Exception("Not macOS")):
            self.assertFalse(is_sandboxed())

    # --- File Reading Tests ---

    def test_read_local_file_text(self):
        test_file = os.path.join(self.test_dir, "hello.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("hello world")
        
        content, total, unit = _read_local_file(test_file)
        self.assertEqual(content, "hello world")
        self.assertEqual(total, 1)
        self.assertEqual(unit, "lines")

    def test_read_local_file_text_partial(self):
        test_file = os.path.join(self.test_dir, "lines.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("line1\nline2\nline3\nline4\n")
        
        # Test offset only
        content, total, unit = _read_local_file(test_file, offset=2)
        self.assertEqual(content, "line3\nline4\n")
        self.assertEqual(total, 4)
        # Test offset and limit
        content, total, unit = _read_local_file(test_file, offset=1, limit=2)
        self.assertEqual(content, "line2\nline3\n")
        self.assertEqual(total, 4)

    @patch("lightweight_local_assistant.tools.PdfReader")
    def test_read_local_file_pdf_partial(self, mock_pdf_reader):
        test_pdf = os.path.join(self.test_dir, "test.pdf")
        with open(test_pdf, "w") as f:
            f.write("pdf-content")
        
        p1 = MagicMock()
        p1.extract_text.return_value = "page 1 text"
        p2 = MagicMock()
        p2.extract_text.return_value = "page 2 text"
        
        mock_pdf_reader.return_value.pages = [p1, p2]
        
        # Test reading only page 2
        content, total, unit = _read_local_file(test_pdf, pages=[2])
        self.assertIn("page 2 text", content)
        self.assertNotIn("page 1 text", content)
        self.assertIn("--- Page 2 ---", content)
        self.assertEqual(total, 2)
        self.assertEqual(unit, "pages")

    def test_read_local_file_not_found(self):
        content, total, unit = _read_local_file("missing_file_xyz.txt")
        self.assertIn("not found", content)

    @patch("lightweight_local_assistant.tools.PdfReader")
    def test_read_local_file_pdf_corrupt(self, mock_pdf_reader):
        bad_pdf = os.path.join(self.test_dir, "corrupt.pdf")
        with open(bad_pdf, "w") as f:
            f.write("not-a-pdf")
        mock_pdf_reader.side_effect = Exception("Invalid PDF format")
        
        content, total, unit = _read_local_file(bad_pdf)
        self.assertIn("Error reading PDF", content)

    # --- Tool Execution Tests ---

    async def test_write_file(self):
        target = os.path.join(self.test_dir, "new.txt")
        result = await write_file_tool(target, "content")
        self.assertIn("Successfully wrote", result)
        with open(target, "r") as f:
            self.assertEqual(f.read(), "content")

    async def test_run_shell_command_success(self):
        with patch("asyncio.create_subprocess_shell") as mock_exec:
            mock_process = MagicMock()
            mock_process.communicate = AsyncMock(return_value=(b"ls output", b""))
            mock_process.returncode = 0
            mock_exec.return_value = mock_process
            
            result = await run_shell_command("ls")
            self.assertIn("ls output", result)
            args = mock_exec.call_args[0]
            self.assertEqual(args, ("ls",))

    async def test_run_shell_command_error(self):
        with patch("asyncio.create_subprocess_shell") as mock_exec:
            mock_process = MagicMock()
            mock_process.communicate = AsyncMock(return_value=(b"", b"permission denied"))
            mock_process.returncode = 1
            mock_exec.return_value = mock_process
            
            result = await run_shell_command("invalid_cmd")
            self.assertIn("Return Code: 1", result)
            self.assertIn("permission denied", result)

    # --- Agent Loop Logic (Mocked Ollama) ---

    async def test_ask_lightweight_local_assistant_planning_mode(self):
        assistant = LocalAssistant()
        
        self.mock_client.chat.side_effect = [
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
            
            result = await assistant.ask("Deep work", use_plan=True)
            
            # Verify Planning Prompt
            plan_msg = self.mock_client.chat.call_args_list[0][1]['messages'][0]['content']
            self.assertIn("Senior Technical Planner", plan_msg)
            self.assertIn("Do NOT execute implementation steps yet", plan_msg)
            self.assertEqual(self.mock_client.chat.call_args_list[0][1]['options'], {'num_ctx': 32768})
            
            # Verify Execution Prompt (Plan Injection)
            # Planning took 2 turns (one for tool call, one for content-only finish)
            exec_context_msg = self.mock_client.chat.call_args_list[2][1]['messages'][-1]['content']
            self.assertIn("CURRENT PLAN STATE", exec_context_msg)
            self.assertEqual(self.mock_client.chat.call_args_list[2][1]['options'], {'num_ctx': 32768})
            
            # Verify Final Output contains the plan
            self.assertIn("--- EXECUTION PLAN ---", result)
            self.assertIn("- [ ] Step A", result)

    async def test_ask_lightweight_local_assistant_planning_mode_warning(self):
        assistant = LocalAssistant()
        
        self.mock_client.chat.side_effect = [
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
            
            result = await assistant.ask("Deep work", use_plan=True)
            
            # Verify Warning
            self.assertIn("[Warning: Agent executed actions but failed to update the plan checklist.]", result)


    async def test_ask_lightweight_local_assistant_no_tool(self):
        assistant = LocalAssistant()
        self.mock_client.chat.side_effect = [
            {'message': {'role': 'assistant', 'content': 'direct answer'}},
            {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}}
        ]
        result = await assistant.ask("What is 2+2?")
        self.assertEqual(result, "direct answer")

    async def test_ask_lightweight_local_assistant_multiple_turns(self):
        assistant = LocalAssistant()
        
        test_file = os.path.join(self.test_dir, "a.txt")
        with open(test_file, "w") as f:
            f.write("source")
        
        self.mock_client.chat.side_effect = [
            {
                'message': {
                    'role': 'assistant',
                    'content': 'Thinking...',
                    'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'filepaths': [test_file]}}}]
                }
            },
            {'message': {'role': 'assistant', 'content': 'Done!'}},
            {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}}
        ]
        
        result = await assistant.ask("Analyze a.txt")
        
        self.assertEqual(result, "Done!")
        self.assertEqual(self.mock_client.chat.call_count, 3)
        
        # Verify the tool call included the reminder
        last_call_messages = self.mock_client.chat.call_args_list[0][1]['messages']
        self.assertEqual(last_call_messages[-1]['role'], 'system')
        self.assertIn("REMINDER", last_call_messages[-1]['content'])

    async def test_ask_lightweight_local_assistant_turn_limit(self):
        assistant = LocalAssistant()
        self.mock_client.chat.return_value = {
            'message': {
                'role': 'assistant',
                'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'filepaths': ['dummy']}}}]
            }
        }

        result = await assistant.ask("Do something")
        self.assertIn("maximum turn limit", result)
        self.assertEqual(self.mock_client.chat.call_count, 20)

    async def test_ask_lightweight_local_assistant_agent_read_missing(self):
        assistant = LocalAssistant()
        
        self.mock_client.chat.side_effect = [
            {
                'message': {
                    'role': 'assistant',
                    'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'filepaths': ['gone.txt']}}}]
                }
            },
            {'message': {'content': 'The file is gone.'}},
            {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}}
        ]
        
        await assistant.ask("Read gone.txt")
        
        final_history = self.mock_client.chat.call_args_list[1][1]['messages']
        tool_msg = next(m for m in final_history if m['role'] == 'tool')
        self.assertIn("not found", tool_msg['content'])

    async def test_ask_lightweight_local_assistant_read_file_partial(self):
        assistant = LocalAssistant()
        
        test_file = os.path.join(self.test_dir, "lines.txt")
        with open(test_file, "w") as f:
            f.write("line1\nline2\nline3\n")
        
        self.mock_client.chat.side_effect = [
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
        
        await assistant.ask("Read line 2 of lines.txt")
        
        final_history = self.mock_client.chat.call_args_list[1][1]['messages']
        tool_msg = next(m for m in final_history if m['role'] == 'tool')
        expected_content = f"--- FILE: {test_file} (3 lines total) ---\nline2\n\n"
        self.assertEqual(tool_msg['content'], expected_content)

    async def test_list_local_models(self):
        from lightweight_local_assistant import list_local_models
        self.mock_client.list.return_value = MagicMock(
            models=[
                MagicMock(model='mistral-nemo:latest'),
                MagicMock(model='llama3:8b')
            ]
        )

        result = await list_local_models()
        self.assertIn("Available local models:", result)
        self.assertIn("mistral-nemo:latest", result)
        self.assertIn("llama3:8b", result)
    async def test_list_local_models_empty(self):
        from lightweight_local_assistant import list_local_models
        self.mock_client.list.return_value = MagicMock(models=[])
        result = await list_local_models()
        self.assertIn("No local models found", result)

    async def test_ask_lightweight_local_assistant_custom_model(self):
        assistant = LocalAssistant(model="llama3")
        self.mock_client.chat.side_effect = [
            {'message': {'content': 'using custom model'}},
            {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}}
        ]
        await assistant.ask("Hello")
        kwargs = self.mock_client.chat.call_args_list[0][1]
        self.assertEqual(kwargs['model'], "llama3")

if __name__ == "__main__":
    unittest.main()
