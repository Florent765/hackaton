#!/bin/bash
# =============================================================
# Hackathon Full Environment Setup Script
# Make changes in CONFIGURATION section before running
# Run with: source hackathon_setup.sh
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
    _error "This script must be sourced: source hackathon_setup.sh"
    exit 1
fi

# =============================================================
# CONFIGURATION — edit before running
# =============================================================

# Your JUDOOR username (folder owner)
OWNER=""   # Your Juelich username

# Teammates (1 to 3, leave unused ones blank "")
TEAMMATE_1=""  # Your teammates' Juelich usernames. Leave unused as it is
TEAMMATE_2=""
TEAMMATE_3=""

# Your personal folder in $SCRATCH (created first, locked to you + team)
YOUR_FOLDER=""   # <-- replace with your username or custom name

# Team folder inside your folder (shared with teammates) 
TEAM_FOLDER="" # <-- replace with your team_name

# HuggingFace datasets to download (org/repo format)
HF_DATASETS=(
    "SprintML/llm_watermark_removal"
    "SprintML/Property_Inference"
    ""
)

# =============================================================

SCRATCH_BASE="/p/scratch/training2614"
YOUR_DIR="$SCRATCH_BASE/$YOUR_FOLDER"
TEAM_DIR="$YOUR_DIR/$TEAM_FOLDER"

# Build teammates array — skip any blank entries
TEAMMATES=()
for t in "$TEAMMATE_1" "$TEAMMATE_2" "$TEAMMATE_3"; do
    [[ -n "$t" ]] && TEAMMATES+=("$t")
done

if [[ "${#TEAMMATES[@]}" -eq 0 ]]; then
    _error "No teammates specified. Add at least one."
    return 1
fi

echo ""
echo "============================================="
echo "   Hackathon Environment Setup"
echo "============================================="
echo ""
echo "  Owner      : $OWNER"
echo "  Teammates  : ${TEAMMATES[*]}"
echo "  Your folder: $YOUR_DIR"
echo "  Team folder: $TEAM_DIR"
echo ""

# -------------------------------------------------------------
# STEP 1: Check cluster
# -------------------------------------------------------------
_log "Step 1/6 - Checking cluster environment..."

if ! command -v jutil &> /dev/null; then
    _error "jutil not found. Are you on JURECA?"
    echo   "       SSH in: ssh <username>@jureca.fz-juelich.de"
    return 1
fi
_ok "jutil found — on JURECA."

# -------------------------------------------------------------
# STEP 2: Activate project
# -------------------------------------------------------------
_log "Step 2/6 - Activating project training2614..."

jutil env activate -p training2614

if [[ -z "$PROJECT" ]]; then
    _error "\$PROJECT empty after activation."
    echo   "       Try manually: jutil env activate -p training2614"
    return 1
fi
_ok "Project activated. \$PROJECT = $PROJECT"

# -------------------------------------------------------------
# STEP 3: Create folders
# -------------------------------------------------------------
_log "Step 3/6 - Creating folders in scratch..."

cd "$SCRATCH_BASE" || { _error "Cannot cd to $SCRATCH_BASE"; return 1; }

if [[ -d "$YOUR_FOLDER" ]]; then
    _warn "Folder $YOUR_FOLDER already exists. Skipping mkdir."
else
    mkdir "$YOUR_FOLDER"
    _ok "Created $YOUR_FOLDER"
fi


if [[ -d "$TEAM_DIR" ]]; then
    _warn "Team folder $TEAM_FOLDER already exists. Skipping mkdir."
else
    mkdir "$TEAM_DIR"
    _ok "Created $TEAM_FOLDER inside $YOUR_FOLDER."
fi

chmod g+s "$YOUR_DIR"

# -------------------------------------------------------------
# STEP 4: Set ACLs — owner + teammates get full access
# -------------------------------------------------------------

_log "Step 4/6 - Setting ACLs on $YOUR_DIR ..."

_lock_down_tree() {
    local dir="$1"

    # Restrict the top-level directory itself
    setfacl -m g::--- "$dir"
    setfacl -m o::--- "$dir"
    setfacl -m m:rwx "$dir"

    # Restrict everything that already exists inside
    setfacl -R -m g::--- "$dir"
    setfacl -R -m o::--- "$dir"
    setfacl -R -m m:rwx "$dir"

    # Restrictive defaults for all new files/subdirs
    setfacl -d -m g::--- "$dir"
    setfacl -d -m o::--- "$dir"
    setfacl -d -m m:rwx "$dir"

    # Apply default ACLs to all existing subdirectories too
    find "$dir" -type d -exec setfacl -d -m g::--- {} \;
    find "$dir" -type d -exec setfacl -d -m o::--- {} \;
    find "$dir" -type d -exec setfacl -d -m m:rwx {} \;
}

_grant_user_access() {
    local uname="$1"
    local dir="$2"

    # Current directory + existing tree
    setfacl -m u:"$uname":rwx "$dir"
    setfacl -R -m u:"$uname":rwx "$dir"

    # Default ACLs for future files/subdirs
    setfacl -d -m u:"$uname":rwx "$dir"
    find "$dir" -type d -exec setfacl -d -m u:"$uname":rwx {} \;
}

_lock_down_tree "$YOUR_DIR"

for uname in "$OWNER" "${TEAMMATES[@]}"; do
    _log "  Granting access to: $uname"
    _grant_user_access "$uname" "$YOUR_DIR"
    _ok "  ACL set for $uname"
done

setfacl -R -m m:rwx "$YOUR_DIR"
setfacl -R -d -m m:rwx "$YOUR_DIR"

_log "Final ACL for owner folder:"
getfacl "$YOUR_DIR"

_log "Final ACL for team folder:"
getfacl "$TEAM_DIR"

# -------------------------------------------------------------
# STEP 5: Install uv + configure shared cache
# -------------------------------------------------------------
_log "Step 5/6 - Installing uv and configuring shared cache..."

if command -v uv &> /dev/null; then
    _warn "uv already installed. Skipping."
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [[ -f "$HOME/.local/bin/env" ]]; then
        source "$HOME/.local/bin/env"
        _ok "uv installed."
    else
        _error "uv install finished but env file not found."
        return 1
    fi
fi

UV_BASE="$TEAM_DIR/.uv"
HF_CACHE="$TEAM_DIR/.cache"
mkdir -p "$UV_BASE" "$HF_CACHE"

if grep -q "UV_CACHE_DIR.*$TEAM_FOLDER" ~/.bashrc; then
    _warn "uv/HF cache config already in ~/.bashrc. Skipping."
else
    cat << EOF >> ~/.bashrc

# uv + HF cache config — shared team env (added by hackathon_setup.sh)
export UV_CACHE_DIR=$UV_BASE/cache
export UV_TOOL_DIR=$UV_BASE/tools
export UV_PYTHON_INSTALL_DIR=$UV_BASE/python
export HF_HOME=$HF_CACHE
export HUGGINGFACE_HUB_CACHE=$HF_CACHE/hub
EOF
    _ok "uv + HF cache paths written to ~/.bashrc"
fi

export UV_CACHE_DIR="$UV_BASE/cache"
export UV_TOOL_DIR="$UV_BASE/tools"
export UV_PYTHON_INSTALL_DIR="$UV_BASE/python"
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub"

# -------------------------------------------------------------
# STEP 6: Download each dataset then create its per-task uv env
# -------------------------------------------------------------
_log "Step 6/6 - Downloading datasets and creating per-task environments..."

if [[ -n "$HF_TOKEN" ]]; then
    _ok "HF_TOKEN found — will be used for authentication."
else
    _log "No HF_TOKEN set — proceeding without (fine for public datasets)."
fi

# Bootstrap venv just for downloading — deleted after use
BOOTSTRAP_VENV="$TEAM_DIR/.bootstrap"
if [[ ! -d "$BOOTSTRAP_VENV" ]]; then
    _log "Creating bootstrap venv for downloads..."
    uv venv -p 3.12 "$BOOTSTRAP_VENV"
    VIRTUAL_ENV="$BOOTSTRAP_VENV" uv pip install huggingface_hub
    _ok "Bootstrap venv ready."
else
    _warn "Bootstrap venv already exists. Skipping."
fi

BOOTSTRAP_PYTHON="$BOOTSTRAP_VENV/bin/python"

for dataset in "${HF_DATASETS[@]}"; do
    dataset_name="${dataset##*/}"
    dest="$TEAM_DIR/$dataset_name"
    VENV_DIR="$dest/.venv"
    REQ="$dest/requirements.txt"

    echo ""
    _log "--- Task: $dataset_name ---"

    # 1. Download dataset
    if [[ -d "$dest" ]]; then
        _warn "  Already downloaded. Skipping."
    else
        _log "  Downloading $dataset..."
        "$BOOTSTRAP_PYTHON" - <<PYEOF
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$dataset",
    repo_type="dataset",
    local_dir="$dest",
    token=os.environ.get("HF_TOKEN"),
)
PYEOF
        if [[ $? -ne 0 ]]; then
            _error "  Download failed — skipping venv setup."
            continue
        fi
        _ok "  Downloaded $dataset_name."
    fi

    # create output folder safely (always run)
    mkdir -p "$dest/output"
    
    # fix mask so both users can write
    setfacl -m m:rwx "$dest/output"
    setfacl -d -m m:rwx "$dest/output"

    # 2. Create proper per-task uv project + venv from requirements.txt
    if [[ -d "$VENV_DIR" ]]; then
        _warn "  Venv already exists. Skipping."
        continue
    fi

    if [[ ! -f "$REQ" ]]; then
        _warn "  No requirements.txt found in $dataset_name — skipping venv."
        continue
    fi

    _log "  Setting up uv project for $dataset_name..."
    cd "$dest" || { _error "  Cannot cd into $dest"; continue; }

    # Init uv project — creates pyproject.toml
    uv init -p 3.12 --no-workspace

    # Create venv inside task folder
    uv venv -p 3.12 "$VENV_DIR"

    # Parse requirements.txt: strip comments and blank lines
    DEPS=$(grep -v '^\s*#' "$REQ" | grep -v '^\s*$' | tr '\n' ' ')

    if [[ -z "$DEPS" ]]; then
        _warn "  requirements.txt is empty — skipping uv add."
    else
        # uv add writes deps into pyproject.toml and generates uv.lock
        VIRTUAL_ENV="$VENV_DIR" uv add $DEPS && \
            _ok "  Done: pyproject.toml + uv.lock + .venv ready." || \
            _error "  Failed to install requirements for $dataset_name."
    fi

    cd "$TEAM_DIR"
done

# Clean up bootstrap venv — no trace left
if [[ -d "$BOOTSTRAP_VENV" ]]; then
    rm -rf "$BOOTSTRAP_VENV"
    _ok "Bootstrap venv removed."
fi

# =============================================================
echo ""
echo "============================================="
echo -e "${GREEN}   Setup complete!${NC}"
echo "============================================="
echo "  Team access : $OWNER + ${TEAMMATES[*]}"
echo "  uv          : $(uv --version 2>&1)"
echo ""
echo "  Activate a task env:"
echo "  source $TEAM_DIR/<dataset-name>/.venv/bin/activate"
echo ""
echo "  Submit a job: sbatch job.sh"
echo "============================================="
echo ""