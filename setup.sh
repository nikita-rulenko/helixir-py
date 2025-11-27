#!/bin/bash
# ============================================================
# Helixir Setup Script
# ============================================================
# This script sets up Helixir for first-time use.
# 
# Usage: ./setup.sh
# ============================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "============================================================"
echo "  🧠 Helixir Setup"
echo "  Persistent Memory Framework for AI"
echo "============================================================"
echo -e "${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# Check Dependencies
# ============================================================

echo -e "${YELLOW}Checking dependencies...${NC}"

# Check Rust
if ! command -v rustc &> /dev/null; then
    echo -e "${RED}❌ Rust not found.${NC}"
    echo ""
    echo "Rust 1.88.0+ is required for HelixDB CLI."
    echo "Install with:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo ""
    exit 1
fi
RUST_VERSION=$(rustc --version | cut -d' ' -f2)
echo -e "${GREEN}✅ Rust ${RUST_VERSION}${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found.${NC}"
    echo ""
    echo "Docker is required for HelixDB."
    echo "Install from: https://docs.docker.com/get-docker/"
    echo ""
    exit 1
fi
echo -e "${GREEN}✅ Docker $(docker --version | cut -d',' -f1 | cut -d' ' -f3-)${NC}"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running.${NC}"
    echo "Please start Docker Desktop and try again."
    exit 1
fi
echo -e "${GREEN}✅ Docker is running${NC}"

# Check Git
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git not found.${NC}"
    echo "Install Git from: https://git-scm.com/downloads"
    exit 1
fi
echo -e "${GREEN}✅ Git $(git --version | cut -d' ' -f3)${NC}"

# Check uv (required)
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ uv not found.${NC}"
    echo ""
    echo "uv is required for Python environment management."
    echo "Install with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    exit 1
fi
echo -e "${GREEN}✅ uv $(uv --version | cut -d' ' -f2)${NC}"

# Check Python version via uv
echo ""
echo -e "${YELLOW}Checking Python version...${NC}"
REQUIRED_PYTHON="3.14"

# Try to find Python 3.14+
if uv python list 2>/dev/null | grep -q "3.14"; then
    echo -e "${GREEN}✅ Python 3.14+ available${NC}"
else
    echo -e "${YELLOW}⚠️  Python 3.14 not found, installing...${NC}"
    uv python install 3.14
    echo -e "${GREEN}✅ Python 3.14 installed${NC}"
fi

# Check Ollama (optional)
echo ""
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✅ Ollama installed${NC}"
    OLLAMA_INSTALLED=true
else
    echo -e "${YELLOW}⚠️  Ollama not found (optional, for local embeddings)${NC}"
    OLLAMA_INSTALLED=false
fi

# Check Helix CLI
echo ""
if command -v helix &> /dev/null; then
    echo -e "${GREEN}✅ Helix CLI installed${NC}"
    HELIX_INSTALLED=true
else
    echo -e "${YELLOW}⚠️  Helix CLI not found${NC}"
    HELIX_INSTALLED=false
fi

# ============================================================
# Install Missing Components
# ============================================================

echo ""
echo -e "${YELLOW}Installing components...${NC}"

# Install Helix CLI if needed
if [ "$HELIX_INSTALLED" = false ]; then
    echo ""
    read -p "Install Helix CLI? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        echo "Installing Helix CLI..."
        curl -sSL https://install.helix-db.com | bash
        export PATH="$HOME/.helix/bin:$PATH"
        echo -e "${GREEN}✅ Helix CLI installed${NC}"
    else
        echo -e "${RED}Helix CLI is required. Exiting.${NC}"
        exit 1
    fi
fi

# Install Ollama if needed
if [ "$OLLAMA_INSTALLED" = false ]; then
    echo ""
    read -p "Install Ollama for local embeddings? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "Installing Ollama via Homebrew..."
            brew install ollama
        else
            echo "Installing Ollama..."
            curl -fsSL https://ollama.com/install.sh | sh
        fi
        echo -e "${GREEN}✅ Ollama installed${NC}"
        OLLAMA_INSTALLED=true
    fi
fi

# Pull embedding model if Ollama is installed
if [ "$OLLAMA_INSTALLED" = true ]; then
    echo ""
    echo "Checking embedding model..."
    if ! ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
        read -p "Download nomic-embed-text model for embeddings? [Y/n] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            ollama pull nomic-embed-text
            echo -e "${GREEN}✅ Embedding model ready${NC}"
        fi
    else
        echo -e "${GREEN}✅ nomic-embed-text model available${NC}"
    fi
fi

# ============================================================
# Create Virtual Environment & Install Dependencies
# ============================================================

echo ""
echo -e "${YELLOW}Setting up Python environment...${NC}"

# Clean up old caches if they exist
echo "Cleaning up old caches..."
rm -rf .venv helixir/.venv 2>/dev/null || true
rm -rf helixir/.ruff_cache helixir/.pytest_cache helixir/.coverage 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Create virtual environment with Python 3.14
echo "Creating virtual environment with Python 3.14..."
uv venv --python 3.14

# Install dependencies
echo "Installing dependencies..."
uv sync

echo -e "${GREEN}✅ Python environment ready${NC}"

# ============================================================
# Run Configuration Wizard
# ============================================================

echo ""
echo -e "${YELLOW}Running configuration wizard...${NC}"

uv run python -m helixir.setup.wizard

# ============================================================
# Deploy Schema to HelixDB
# ============================================================

echo ""
echo -e "${YELLOW}Deploying schema to HelixDB...${NC}"

uv run python -m helixir.setup.deploy_schema

# ============================================================
# Seed Base Memories
# ============================================================

echo ""
echo -e "${YELLOW}Seeding base memories...${NC}"

uv run python -m helixir.setup.seed_memories

# ============================================================
# Done!
# ============================================================

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  ✅ Helixir Setup Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Restart Cursor to reload MCP servers"
echo "  2. Ask your AI assistant: 'What is Helixir?'"
echo ""
echo "To activate the environment manually:"
echo "  source .venv/bin/activate"
echo ""
echo "Documentation: https://github.com/helixdb/helixir"
echo ""
