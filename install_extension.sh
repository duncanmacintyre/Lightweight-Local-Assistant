#!/bin/bash
# install_extension.sh
# Performs a global installation of the Lightweight Local Assistant for Gemini CLI.

cd "$(dirname "$0")"
SOURCE_DIR=$(pwd)

echo "🚀 Starting Installation of Lightweight Local Assistant for Gemini CLI..."

# 1. Prerequisites Check
if ! ollama list > /dev/null 2>&1; then
    echo "❌ Error: Ollama is not running. Please start Ollama and try again."
    exit 1
fi

# 2. Model Selection
MODELS=$(ollama list | tail -n +2 | awk '{print $1}')
if [ -z "$MODELS" ]; then
    echo "⚠️ No Ollama models found. Please 'ollama pull qwen3-coder:30b' first."
    exit 1
fi

echo ""
echo "--- Local Model Selection ---"
echo "Select a model to use as the default Local Worker:"
select MODEL in $MODELS; do
    if [ -n "$MODEL" ]; then
        echo "✅ Selected model: $MODEL"
        break
    else
        echo "❌ Invalid selection."
    fi
done

# 3. Installation Directory Selection
DEFAULT_INSTALL_DIR="$HOME/.gemini-lightweight-local-assistant"
echo ""
echo "--- Installation Directory ---"
read -p "Enter installation directory [$DEFAULT_INSTALL_DIR]: " INSTALL_DIR
INSTALL_DIR=${INSTALL_DIR:-$DEFAULT_INSTALL_DIR}

# Expand tilde if present
INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"

# 4. Create Installation Directory
echo "📂 Creating installation directory at $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"

# 5. Copy Essential Files
echo "📄 Copying project files..."
cp "$SOURCE_DIR/mcp_server.py" "$INSTALL_DIR/"
cp "$SOURCE_DIR/requirements.txt" "$INSTALL_DIR/"
cp "$SOURCE_DIR/pyproject.toml" "$INSTALL_DIR/"
cp -R "$SOURCE_DIR/lightweight_local_assistant" "$INSTALL_DIR/"

# 6. Create .env
echo "LOCAL_WORKER_MODEL=$MODEL" > "$INSTALL_DIR/.env"
echo "OLLAMA_BASE_URL=http://localhost:11434" >> "$INSTALL_DIR/.env"

# 7. Set up Virtual Environment
echo "🐍 Setting up Python environment (this may take a minute)..."
python3 -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/pip" install --quiet -e "$INSTALL_DIR/"

# 8. Generate run_safe wrapper
WRAPPER_SCRIPT="$INSTALL_DIR/run_safe"
cat <<EOF > "$WRAPPER_SCRIPT"
#!/bin/bash
# Wrapper for the Lightweight Local Assistant
export OLLAMA_BASE_URL="http://localhost:11434"
# Change to the current working directory from which Gemini was launched
# Gemini CLI sets the PWD when it launches MCP servers
exec "$INSTALL_DIR/.venv/bin/python" -u "$INSTALL_DIR/mcp_server.py"
EOF
chmod +x "$WRAPPER_SCRIPT"

# 9. Register with Gemini CLI
echo "🔗 Connecting Gemini CLI (Stdio Mode)..."
# We recommend users use the -s flag.
gemini mcp remove lightweight-local-assistant >/dev/null 2>&1
gemini mcp remove --scope user lightweight-local-assistant >/dev/null 2>&1
gemini mcp add --scope user lightweight-local-assistant /bin/bash "$WRAPPER_SCRIPT"

echo ""
echo "🎉 SUCCESS: Lightweight Local Assistant is ready."
echo "------------------------------------------------------------"
echo "IMPORTANT: To use the Lightweight Local Assistant, you MUST run Gemini CLI with the sandbox flag:"
echo "  gemini -s \"What files are in this directory?\""
echo "------------------------------------------------------------"
