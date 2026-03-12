import os
import sys
import unittest
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock

# Add current directory to sys.path
sys.path.append(os.getcwd())

import mcp_server
from lightweight_local_assistant import LocalAssistant
from lightweight_local_assistant.tools import _read_local_file

class TestMCPServer(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.show = AsyncMock(return_value={"modelinfo": {"context_length": 32768}})
        self.mock_client.chat = AsyncMock()
        self.mock_client.list = AsyncMock()

    # --- Sandbox Detection Tests ---

    def test_is_sandboxed_true(self):
        with patch("ctypes.CDLL") as mock_cdll:
            mock_libsandbox = MagicMock()
            mock_libsandbox.sandbox_check.return_value = 1
            mock_cdll.return_value = mock_libsandbox
            self.assertTrue(mcp_server.is_sandboxed())

    def test_is_sandboxed_false(self):
        with patch("ctypes.CDLL") as mock_cdll:
            mock_libsandbox = MagicMock()
            mock_libsandbox.sandbox_check.return_value = 0
            mock_cdll.return_value = mock_libsandbox
            self.assertFalse(mcp_server.is_sandboxed())

    def test_is_sandboxed_exception(self):
        with patch("ctypes.CDLL", side_effect=Exception("Error")):
            self.assertFalse(mcp_server.is_sandboxed())

    # --- File Reading Tests ---

    def test_read_local_file_text(self):
        import io
        def mock_open_se(path, mode="r", *args, **kwargs):
            if "b" in mode:
                return io.BytesIO(b"hello")
            else:
                return io.StringIO("hello")

        with patch("os.path.exists", return_value=True):
            with patch("os.path.getsize", return_value=5):
                with patch("builtins.open", side_effect=mock_open_se):
                    result, total, unit = _read_local_file("test.txt")
                    self.assertEqual(result, "hello")
                    self.assertEqual(unit, "lines")

    def test_read_local_file_not_found(self):
        with patch("os.path.exists", return_value=False):
            result, total, unit = _read_local_file("missing.txt")
            self.assertIn("not found", result)

    def test_read_local_file_pdf_success(self):
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "pdf text"
        mock_reader.pages = [mock_page]

        with patch("os.path.exists", return_value=True):
            with patch("lightweight_local_assistant.tools.PdfReader", return_value=mock_reader):
                result, total, unit = _read_local_file("test.pdf")
                self.assertEqual(result, "pdf text\n")
                self.assertEqual(total, 1)
                self.assertEqual(unit, "pages")

    def test_read_local_file_pdf_error(self):
        with patch("os.path.exists", return_value=True):
            with patch("lightweight_local_assistant.tools.PdfReader", side_effect=Exception("Bad PDF")):
                result, total, unit = _read_local_file("test.pdf")
                self.assertIn("Error reading PDF", result)
    # --- Tool Execution Tests ---

    async def test_run_shell_command_success(self):
        with patch("asyncio.create_subprocess_shell") as mock_exec:
            mock_process = MagicMock()
            mock_process.communicate = AsyncMock(return_value=(b"out", b"err"))
            mock_process.returncode = 0
            mock_exec.return_value = mock_process
            
            result = await mcp_server.run_shell_command("ls")
            self.assertIn("STDOUT:\nout", result)

    async def test_run_shell_command_timeout(self):
        with patch("asyncio.create_subprocess_shell") as mock_exec:
            mock_process = MagicMock()
            mock_process.communicate = AsyncMock()
            mock_process.terminate = MagicMock()
            mock_process.wait = AsyncMock()
            mock_exec.return_value = mock_process
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                result = await mcp_server.run_shell_command("ls")
                self.assertIn("timed out", result)

    async def test_write_file_success(self):
        with patch("os.makedirs"):
            with patch("builtins.open", unittest.mock.mock_open()):
                result = await mcp_server.write_file("dir/file.txt", "data")
                self.assertIn("Successfully wrote", result)

    async def test_write_file_error(self):
        with patch("os.makedirs", side_effect=OSError("Perm Denied")):
            result = await mcp_server.write_file("dir/file.txt", "data")
            self.assertIn("Error writing file", result)

    # --- Agent Loop Logic ---

    @patch("lightweight_local_assistant.agent.ollama.AsyncClient")
    async def test_ask_lightweight_local_assistant_no_tool(self, mock_client_class):
        mock_client_class.return_value = self.mock_client
        self.mock_client.chat.side_effect = [
            {'message': {'role': 'assistant', 'content': 'hello'}},
            {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}}
        ]
        result = await mcp_server.ask_lightweight_local_assistant("hi")
        self.assertEqual(result, "hello")

    @patch("lightweight_local_assistant.agent.ollama.AsyncClient")
    async def test_ask_lightweight_local_assistant_all_tools(self, mock_client_class):
        mock_client_class.return_value = self.mock_client
        self.mock_client.chat.side_effect = [
            {'message': {'role': 'assistant', 'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'filepaths': ['f.txt']}}}]}},
            {'message': {'role': 'assistant', 'tool_calls': [{'function': {'name': 'write_file', 'arguments': {'filepath': 'w.txt', 'content': 'c'}}}]}},
            {'message': {'role': 'assistant', 'tool_calls': [{'function': {'name': 'run_shell_command', 'arguments': {'command': 'ls'}}}]}},
            {'message': {'role': 'assistant', 'content': 'Done'}},
            {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}}
        ]
        
        with patch("lightweight_local_assistant.agent._read_local_file", return_value=("read", 1, "lines")):
            with patch("lightweight_local_assistant.agent.write_file_tool", new_callable=AsyncMock, return_value="wrote"):
                with patch("lightweight_local_assistant.agent.run_shell_command", new_callable=AsyncMock, return_value="ran"):
                    with patch("lightweight_local_assistant.agent.get_model_info", new_callable=AsyncMock, return_value='{"context_length": 32000}'):
                        result = await mcp_server.ask_lightweight_local_assistant("hi")
                        self.assertEqual(result, "Done")

    @patch("lightweight_local_assistant.agent.ollama.AsyncClient")
    async def test_ask_lightweight_local_assistant_turn_limit(self, mock_client_class):
        mock_client_class.return_value = self.mock_client
        self.mock_client.chat.return_value = {
            'message': {'role': 'assistant', 'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'filepaths': ['dummy']}}}]}
        }
        result = await mcp_server.ask_lightweight_local_assistant("hi")
        self.assertIn("maximum turn limit", result)

    # --- Tool Wrapper Tests ---

    async def test_read_file_tool(self):
        with patch("lightweight_local_assistant.tools._read_local_file", return_value=("content", 1, "lines")):
            with patch("os.path.exists", return_value=True):
                result = await mcp_server.read_file("f.txt")
                self.assertIn("content", result)

    async def test_read_file_tool_missing(self):
        with patch("os.path.exists", return_value=False):
            result = await mcp_server.read_file("f.txt")
            self.assertIn("not found", result)

    @patch("lightweight_local_assistant.agent.ollama.AsyncClient")
    async def test_ask_lightweight_local_assistant_error(self, mock_client_class):
        mock_client_class.return_value = self.mock_client
        self.mock_client.chat.side_effect = Exception("Ollama Down")
        result = await mcp_server.ask_lightweight_local_assistant("hi")
        self.assertIn("Error in local assistant", result)

    # --- Model Listing ---

    @patch("lightweight_local_assistant.models.ollama.AsyncClient")
    async def test_list_local_models_success(self, mock_client_class):
        mock_client_class.return_value = self.mock_client
        self.mock_client.list.return_value = MagicMock(
            models=[MagicMock(model='m1'), MagicMock(model='m2')]
        )
        result = await mcp_server.list_local_models()
        self.assertIn("m1", result)
        self.assertIn("m2", result)

    @patch("lightweight_local_assistant.models.ollama.AsyncClient")
    async def test_list_local_models_empty(self, mock_client_class):
        mock_client_class.return_value = self.mock_client
        self.mock_client.list.return_value = MagicMock(models=[])
        result = await mcp_server.list_local_models()
        self.assertIn("No local models found", result)

    @patch("lightweight_local_assistant.models.ollama.AsyncClient")
    async def test_list_local_models_error(self, mock_client_class):
        mock_client_class.return_value = self.mock_client
        self.mock_client.list.side_effect = Exception("Failed")
        result = await mcp_server.list_local_models()
        self.assertIn("Error listing models", result)

if __name__ == "__main__":
    unittest.main()
