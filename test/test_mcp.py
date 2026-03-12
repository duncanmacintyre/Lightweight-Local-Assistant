import os
import pytest
import subprocess
import ctypes
import json
from unittest.mock import MagicMock, patch, mock_open, AsyncMock
from lightweight_local_assistant import is_sandboxed, run_shell_command, write_file_tool, read_file_tool, LocalAssistant
from lightweight_local_assistant.tools import _read_local_file

# --- Sandbox Detection Tests ---

def test_is_sandboxed_true():
    """Test is_sandboxed returns True when sandbox_check returns 1."""
    with patch("ctypes.CDLL") as mock_cdll:
        mock_libsandbox = MagicMock()
        mock_libsandbox.sandbox_check.return_value = 1
        mock_cdll.return_value = mock_libsandbox
        
        assert is_sandboxed() is True

def test_is_sandboxed_false():
    """Test is_sandboxed returns False when sandbox_check returns 0."""
    with patch("ctypes.CDLL") as mock_cdll:
        mock_libsandbox = MagicMock()
        mock_libsandbox.sandbox_check.return_value = 0
        mock_cdll.return_value = mock_libsandbox
        
        assert is_sandboxed() is False

def test_is_sandboxed_exception():
    """Test is_sandboxed returns False when an exception occurs."""
    with patch("ctypes.CDLL", side_effect=Exception("Not macOS")):
        assert is_sandboxed() is False

# --- File Reading Tests ---

def test_read_local_file_text(tmp_path):
    """Test reading a standard text file."""
    test_file = tmp_path / "hello.txt"
    test_file.write_text("hello world", encoding="utf-8")
    
    content, total, unit = _read_local_file(str(test_file))
    assert content == "hello world"
    assert total == 1
    assert unit == "lines"

def test_read_local_file_text_partial(tmp_path):
    """Test reading a text file with offset and limit."""
    test_file = tmp_path / "lines.txt"
    test_file.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")
    
    # Test offset only
    content, total, unit = _read_local_file(str(test_file), offset=2)
    assert content == "line3\nline4\n"
    assert total == 4
    # Test offset and limit
    content, total, unit = _read_local_file(str(test_file), offset=1, limit=2)
    assert content == "line2\nline3\n"
    assert total == 4

@patch("lightweight_local_assistant.tools.PdfReader")
def test_read_local_file_pdf_partial(mock_pdf_reader, tmp_path):
    """Test reading specific pages of a PDF."""
    test_pdf = tmp_path / "test.pdf"
    test_pdf.write_text("pdf-content")
    
    p1 = MagicMock()
    p1.extract_text.return_value = "page 1 text"
    p2 = MagicMock()
    p2.extract_text.return_value = "page 2 text"
    
    mock_pdf_reader.return_value.pages = [p1, p2]
    
    # Test reading only page 2
    content, total, unit = _read_local_file(str(test_pdf), pages=[2])
    assert "page 2 text" in content
    assert "page 1 text" not in content
    assert "--- Page 2 ---" in content
    assert total == 2
    assert unit == "pages"

def test_read_local_file_not_found():
    """Verify error message for non-existent files."""
    content, total, unit = _read_local_file("missing_file_xyz.txt")
    assert "not found" in content

@patch("lightweight_local_assistant.tools.PdfReader")
def test_read_local_file_pdf_corrupt(mock_pdf_reader, tmp_path):
    """Verify handling of corrupted PDF files."""
    bad_pdf = tmp_path / "corrupt.pdf"
    bad_pdf.write_text("not-a-pdf")
    mock_pdf_reader.side_effect = Exception("Invalid PDF format")
    
    content, total, unit = _read_local_file(str(bad_pdf))
    assert "Error reading PDF" in content

# --- Tool Execution Tests ---

@pytest.mark.asyncio
async def test_write_file(tmp_path):
    """Test the write_file tool."""
    target = tmp_path / "new.txt"
    result = await write_file_tool(str(target), "content")
    assert "Successfully wrote" in result
    assert target.read_text() == "content"

@pytest.mark.asyncio
async def test_run_shell_command_success():
    """Test the shell command runner."""
    
    with patch("asyncio.create_subprocess_shell") as mock_exec:
        mock_process = MagicMock()
        mock_process.communicate = AsyncMock(return_value=(b"ls output", b""))
        mock_process.returncode = 0
        mock_exec.return_value = mock_process
        
        result = await run_shell_command("ls")
        assert "ls output" in result
        args = mock_exec.call_args[0]
        assert args[0] == "ls"

@pytest.mark.asyncio
async def test_run_shell_command_error():
    """Test that non-zero exit codes are handled and reported."""
    with patch("asyncio.create_subprocess_shell") as mock_exec:
        mock_process = MagicMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b"permission denied"))
        mock_process.returncode = 1
        mock_exec.return_value = mock_process
        
        result = await run_shell_command("invalid_cmd")
        assert "Return Code: 1" in result
        assert "permission denied" in result

# --- Agent Loop Logic (Mocked Ollama) ---

def setup_mock_client(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.show = AsyncMock(return_value={"modelinfo": {"context_length": 32768}})
    mock_client.chat = AsyncMock()
    return mock_client

@pytest.mark.asyncio
@patch("lightweight_local_assistant.agent.ollama.AsyncClient")
async def test_ask_lightweight_local_assistant_planning_mode(mock_client_class):
    """Test that use_plan=True follows the 2-phase workflow."""
    mock_client = setup_mock_client(mock_client_class)
    mock_client.chat.side_effect = [
        # Phase 1: Planning Loop -> Agent writes the plan file
        {
            'message': {
                'role': 'assistant',
                'tool_calls': [{
                    'function': {
                        'name': 'write_file',
                        'arguments': json.dumps({'filepath': '.gemini/local_plan.md', 'content': '- [ ] Step 1'})
                    }
                }]
            }
        },
        # Phase 1: Finish planning
        {'message': {'role': 'assistant', 'content': 'Plan created.'}},
        # Phase 2: Execution Loop -> Agent does work (mocking immediate finish)
        {'message': {'role': 'assistant', 'content': 'I executed the plan.'}},
        # Phase 2: Reflection
        {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}},
        # Phase 2: Nudge 1
        {'message': {'role': 'assistant', 'content': 'Nudge 1'}},
        # Phase 2: Nudge 2
        {'message': {'role': 'assistant', 'content': 'Nudge 2'}}
    ]
    
    # We need to mock os.path.exists and open to simulate the plan file handling
    with patch("os.path.exists") as mock_exists, \
         patch("builtins.open", mock_open(read_data="- [ ] Step 1")) as mock_file, \
         patch("os.remove") as mock_remove, \
         patch("os.getpid", return_value=123):
         
        # 1. Simulate plan file not existing initially (Phase 1)
        # 2. Simulate plan file existing (Phase 2)
        mock_exists.side_effect = lambda path: path != ".gemini/local_plan.md" or mock_exists.call_count > 1
        
        assistant = LocalAssistant()
        await assistant.ask("Refactor code", use_plan=True)
        
        # Assertions
        # call_args_list[0] is for get_model_info (show)
        # call_args_list[1] is the first chat call
        plan_system_msg = mock_client.chat.call_args_list[0][1]['messages'][0]['content']
        assert "Senior Technical Planner" in plan_system_msg
        assert "Do NOT execute implementation steps yet" in plan_system_msg

@pytest.mark.asyncio
@patch("lightweight_local_assistant.agent.ollama.AsyncClient")
async def test_ask_lightweight_local_assistant_planning_mode_warning(mock_client_class):
    """Test that a warning is appended if the plan is not updated."""
    mock_client = setup_mock_client(mock_client_class)
    mock_client.chat.side_effect = [
        # Phase 1: Write Plan
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
        # Phase 1: Finish planning
        {'message': {'role': 'assistant', 'content': 'Plan created.'}},
        # Phase 2: Execute (No update to plan)
        {'message': {'role': 'assistant', 'content': 'Executed without updating plan.'}},
        # Phase 2: Reflection
        {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}},
        # Phase 2: Nudge Turn 1
        {'message': {'role': 'assistant', 'content': 'Still not updating.'}},
        # Phase 2: Nudge Turn 2
        {'message': {'role': 'assistant', 'content': 'Final attempt at not updating.'}}
    ]
    with patch("os.path.exists") as mock_exists, \
         patch("builtins.open", mock_open(read_data="- [ ] Step A")) as mock_file, \
         patch("os.remove") as mock_remove:

        # Return False for the first call (plan not created), then True for all subsequent calls
        mock_exists.side_effect = lambda path: path != ".gemini/local_plan.md" or mock_exists.call_count > 1

        assistant = LocalAssistant()
        result = await assistant.ask("Deep work", use_plan=True)
    
    assert "[Warning: Agent executed actions but failed to update the plan checklist.]" in result

@pytest.mark.asyncio
@patch("lightweight_local_assistant.agent.ollama.AsyncClient")
async def test_ask_lightweight_local_assistant_no_tool(mock_client_class):
    """Test assistant when no tool call is needed."""
    mock_client = setup_mock_client(mock_client_class)
    mock_client.chat.side_effect = [
        # 1. Main response
        {'message': {'role': 'assistant', 'content': 'direct answer'}},
        # 2. Reflection response (status: complete)
        {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}}
    ]

    assistant = LocalAssistant()
    result = await assistant.ask("What is 2+2?")
    assert result == "direct answer"

@pytest.mark.asyncio
@patch("lightweight_local_assistant.agent.ollama.AsyncClient")
async def test_ask_lightweight_local_assistant_multiple_turns(mock_client_class, tmp_path):
    """Test assistant when it calls tools iteratively over multiple turns."""
    test_file = tmp_path / "a.txt"
    test_file.write_text("source")

    mock_client = setup_mock_client(mock_client_class)
    mock_client.chat.side_effect = [
        {
            'message': {
                'role': 'assistant',
                'content': 'Thinking about reading...',
                'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'filepaths': [str(test_file)]}}}]
            }
        },
        {'message': {'role': 'assistant', 'content': 'Done!'}},
        # Reflection step: complete
        {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}}
    ]

    assistant = LocalAssistant()
    result = await assistant.ask("Analyze a.txt")

    assert result == "Done!"
    assert mock_client.chat.call_count == 3  # Tool call + Final Answer + Reflection

    # Verify the last call included the reminder (on the tool call turn)
    first_call_messages = mock_client.chat.call_args_list[0][1]['messages']
    assert first_call_messages[-1]['role'] == 'system'
    assert "REMINDER" in first_call_messages[-1]['content']

@pytest.mark.asyncio
@patch("lightweight_local_assistant.agent.ollama.AsyncClient")
async def test_ask_lightweight_local_assistant_reflection_incomplete(mock_client_class):
    """Test that the agent self-corrects when reflection returns incomplete."""
    mock_client = setup_mock_client(mock_client_class)
    mock_client.chat.side_effect = [
        # 1. First attempt: Claims done, but missed something
        {'message': {'role': 'assistant', 'content': 'Partial answer.'}},

        # 2. Reflection: Realizes incomplete
        {'message': {'role': 'assistant', 'content': '{"status": "incomplete", "reason": "Missed file B"}'}},

        # 3. Agent reacts to system message injecting the reason -> Calls tool
        {'message': {
            'role': 'assistant', 
            'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'filepaths': ['fileB.txt']}}}]
        }},

        # 4. Final Answer
        {'message': {'role': 'assistant', 'content': 'Complete answer.'}},
    ]

    assistant = LocalAssistant()
    result = await assistant.ask("Do full task")

    assert result == "Complete answer."
    assert mock_client.chat.call_count == 4

    # Verify the self-correction message was injected
    third_call_msgs = mock_client.chat.call_args_list[2][1]['messages']
    assert "SELF-CORRECTION" in third_call_msgs[-2]['content']
    assert "Missed file B" in third_call_msgs[-2]['content']

@pytest.mark.asyncio
@patch("lightweight_local_assistant.agent.ollama.AsyncClient")
async def test_ask_lightweight_local_assistant_turn_limit(mock_client_class):
    """Verify that the agent stops and returns a message when the turn limit is reached."""
    mock_client = setup_mock_client(mock_client_class)
    mock_client.chat.return_value = {
        'message': {
            'role': 'assistant',
            'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'filepaths': ['dummy']}}}]
        }
    }

    assistant = LocalAssistant()
    result = await assistant.ask("Do something")
    assert "maximum turn limit" in result
    assert mock_client.chat.call_count == 20

@pytest.mark.asyncio
@patch("lightweight_local_assistant.agent.ollama.AsyncClient")
async def test_ask_lightweight_local_assistant_max_turns_override(mock_client_class):
    """Verify that max_turns parameter correctly overrides the default."""
    mock_client = setup_mock_client(mock_client_class)
    mock_client.chat.return_value = {
        'message': {
            'role': 'assistant',
            'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'filepaths': ['dummy']}}}]
        }
    }

    assistant = LocalAssistant()
    result = await assistant.ask("Do something", max_turns=5)
    assert "maximum turn limit (5)" in result
    assert mock_client.chat.call_count == 5

@pytest.mark.asyncio
@patch("lightweight_local_assistant.agent.ollama.AsyncClient")
async def test_ask_lightweight_local_assistant_planning_min_turns(mock_client_class):
    """Verify that planning mode enforces a minimum of 30 turns."""
    mock_client = setup_mock_client(mock_client_class)

    assistant = LocalAssistant()
    result = await assistant.ask("Plan something", use_plan=True, max_turns=10)
    assert "Warning: Planning mode requires at least 30 turns" in result
    assert mock_client.chat.call_count == 0  # Should exit before any chat calls

@pytest.mark.asyncio
@patch("lightweight_local_assistant.agent.ollama.AsyncClient")
async def test_ask_lightweight_local_assistant_agent_read_missing(mock_client_class):
    """Verify that if the agent tries to read a missing file, it receives an error result."""
    mock_client = setup_mock_client(mock_client_class)
    mock_client.chat.side_effect = [
        {
            'message': {
                'role': 'assistant',
                'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'filepaths': ['gone.txt']}}}]
            }
        },
        {'message': {'content': 'The file is gone.'}},
        {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}}
    ]

    assistant = LocalAssistant()
    await assistant.ask("Read gone.txt")

    final_history = mock_client.chat.call_args_list[1][1]['messages']
    tool_msg = next(m for m in final_history if m['role'] == 'tool')
    assert "not found" in tool_msg['content']

@pytest.mark.asyncio
@patch("lightweight_local_assistant.agent.ollama.AsyncClient")
async def test_ask_lightweight_local_assistant_read_file_partial(mock_client_class, tmp_path):
    """Verify that the agent can call read_file with offset/limit/pages."""
    test_file = tmp_path / "lines.txt"
    test_file.write_text("line1\nline2\nline3\n", encoding="utf-8")

    mock_client = setup_mock_client(mock_client_class)
    mock_client.chat.side_effect = [
        {
            'message': {
                'role': 'assistant',
                'tool_calls': [{
                    'function': {
                        'name': 'read_file', 
                        'arguments': {'filepaths': [str(test_file)], 'offset': 1, 'limit': 1}
                    }
                }]
            }
        },
        {'message': {'content': 'Read line 2'}},
        {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}}
    ]

    assistant = LocalAssistant()
    await assistant.ask("Read line 2 of lines.txt")

    # Check tool result in history
    final_history = mock_client.chat.call_args_list[1][1]['messages']
    tool_msg = next(m for m in final_history if m['role'] == 'tool')
    expected_content = f"--- FILE: {test_file} (3 lines total) ---\nline2\n\n"
    assert tool_msg['content'] == expected_content

# --- Model & Ollama Tool Tests ---

@pytest.mark.asyncio
@patch("lightweight_local_assistant.models.ollama.AsyncClient")
async def test_list_local_models(mock_client_class):
    """Test listing available Ollama models."""
    from lightweight_local_assistant import list_local_models
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.list = AsyncMock(return_value={
        'models': [
            {'name': 'mistral-nemo:latest'},
            {'name': 'llama3:8b'}
        ]
    })

    result = await list_local_models()
    assert "Available local models:" in result
    assert "mistral-nemo:latest" in result
    assert "llama3:8b" in result

@pytest.mark.asyncio
@patch("lightweight_local_assistant.models.ollama.AsyncClient")
async def test_list_local_models_empty(mock_client_class):
    """Test listing models when none are installed."""
    from lightweight_local_assistant import list_local_models
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.list = AsyncMock(return_value={'models': []})
    result = await list_local_models()
    assert "No local models found" in result

@pytest.mark.asyncio
@patch("lightweight_local_assistant.models.ollama.AsyncClient")
async def test_list_local_models_error(mock_client_class):
    """Test list_local_models error handling."""
    from lightweight_local_assistant import list_local_models
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.list = AsyncMock(side_effect=Exception("Ollama connection failed"))
    result = await list_local_models()
    assert "Error listing models" in result

@pytest.mark.asyncio
@patch("lightweight_local_assistant.agent.ollama.AsyncClient")
async def test_ask_lightweight_local_assistant_custom_model(mock_client_class):
    """Test assistant using a specific model."""
    mock_client = setup_mock_client(mock_client_class)
    mock_client.chat.side_effect = [
        {'message': {'content': 'using custom model'}},
        {'message': {'role': 'assistant', 'content': '{"status": "complete"}'}}
    ]
    assistant = LocalAssistant(model="llama3")
    await assistant.ask("Hello")
    kwargs = mock_client.chat.call_args_list[0][1]
    assert kwargs['model'] == "llama3"
    assert kwargs['options'] == {'num_ctx': 32768}
