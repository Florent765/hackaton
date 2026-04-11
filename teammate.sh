#!/bin/bash
# =============================================================
# Hackathon Setup each for teammates
# Make changes in CONFIGURATION section below before running this script:
# 1) for property OWNER: add your team lead user name from Juelich
# 2) for property TEAM_FOLDER: add your team name (exactly same what the team_leader added in hackathon_setup.sh) 
# Run with: source teammate.sh
# =============================================================

# =============================================================
# CONFIGURATION — edit before running
# =============================================================

OWNER=""    # <-- replace with your team_lead username from Juelich
TEAM_FOLDER="" # <-- replace with your team_name as done by team_lead

# =============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

_log()   { echo -e "${BLUE}[INFO]${NC}  $1"; }
_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
_error() { echo -e "${RED}[ERROR]${NC} $1"; }

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    _error "This script must be sourced: source teammate.sh"
    exit 1
fi

PROJECT_NAME="training2614"
WRITE_TO_BASHRC="yes"

# =============================================================

SCRATCH_BASE="/p/scratch/$PROJECT_NAME"
TEAM_DIR="$SCRATCH_BASE/$OWNER/$TEAM_FOLDER"

UV_BASE="$TEAM_DIR/.uv"
HF_CACHE="$TEAM_DIR/.cache"

echo ""
echo "============================================="
echo "   Shared Team Environment Setup"
echo "============================================="
echo ""
echo "  Project     : $PROJECT_NAME"
echo "  Team folder : $TEAM_DIR"
echo ""

_log "Step 1/4 - Checking cluster tools..."

if ! command -v jutil &> /dev/null; then
    _error "jutil not found. Are you on JURECA?"
    return 1
fi
_ok "jutil found."

_log "Step 2/4 - Activating project $PROJECT_NAME..."

jutil env activate -p "$PROJECT_NAME"

if [[ -z "$PROJECT" ]]; then
    _error "\$PROJECT is empty after activation."
    return 1
fi
_ok "Project activated. \$PROJECT = $PROJECT"

_log "Step 3/4 - Exporting shared cache paths..."

export UV_CACHE_DIR="$UV_BASE/cache"
export UV_TOOL_DIR="$UV_BASE/tools"
export UV_PYTHON_INSTALL_DIR="$UV_BASE/python"
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub"

_ok "Shared environment variables exported in current shell."

if [[ "$WRITE_TO_BASHRC" == "yes" ]]; then
    if grep -q "TEAM SHARED UV/HF CONFIG: $TEAM_DIR" ~/.bashrc 2>/dev/null; then
        _warn "Shared config already present in ~/.bashrc"
    else
        cat << EOF >> ~/.bashrc

# TEAM SHARED UV/HF CONFIG: $TEAM_DIR
export PATH="\$HOME/.local/bin:\$PATH"
export UV_CACHE_DIR=$UV_BASE/cache
export UV_TOOL_DIR=$UV_BASE/tools
export UV_PYTHON_INSTALL_DIR=$UV_BASE/python
export HF_HOME=$HF_CACHE
export HUGGINGFACE_HUB_CACHE=$HF_CACHE/hub
EOF
        _ok "Shared config written to ~/.bashrc"
    fi
fi

_log "Step 4/4 - Checking access to shared folder and uv..."

if [[ ! -d "$TEAM_DIR" ]]; then
    _error "Shared team folder does not exist: $TEAM_DIR"
    return 1
fi

if [[ ! -r "$TEAM_DIR" || ! -x "$TEAM_DIR" ]]; then
    _error "You do not have proper access to: $TEAM_DIR"
    return 1
fi

_ok "You can access the shared team folder."

if command -v uv &> /dev/null; then
    _ok "uv found: $(uv --version 2>&1)"
else
    _warn "uv is not available in your shell. Installing for this user..."

    curl -LsSf https://astral.sh/uv/install.sh | sh

    if [[ -f "$HOME/.local/bin/env" ]]; then
        source "$HOME/.local/bin/env"
    fi

    export PATH="$HOME/.local/bin:$PATH"

    if command -v uv &> /dev/null; then
        _ok "uv installed: $(uv --version 2>&1)"
    else
        _error "uv installation seems to have failed."
        echo "Run these manually:"
        echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "source \$HOME/.local/bin/env"
        return 1
    fi
fi

echo ""
echo "============================================="
echo -e "${GREEN}   Teammate shell setup complete!${NC}"
echo "============================================="
echo ""
echo "Use the shared folder with:"
echo "  cd $TEAM_DIR/<dataset-name>"
echo "  source .venv/bin/activate"
echo ""
echo "Or run directly:"
echo "  uv run $TEAM_DIR/<dataset-name>/main.py"
echo ""