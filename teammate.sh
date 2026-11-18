#!/bin/bash
# =============================================================
# HACKATHON SETUP: STEP 2 OF 2 - TEAMMATES RUN AFTER TEAM LEAD
# =============================================================
# EXECUTION ORDER (IMPORTANT):
# 1. The TEAM LEAD must first run hackathon_setup.sh and
#    finish creating the shared folders, ACLs, datasets and envs.
# 2. ONLY AFTER THE TEAM LEAD FINISHES, each remaining team
#    member runs THIS script from their own JURECA account.
#    Run: source teammate.sh
#
# The team lead runs hackathon_setup.sh, not this script.
# Every remaining member runs teammate.sh individually.
#
# Before running, set TEAM_LEAD to the team lead's Juelich
# username and TEAM_FOLDER to the EXACT team name used by
# the team lead in hackathon_setup.sh.
# =============================================================

# =============================================================
# CONFIGURATION - EACH TEAMMATE EDITS THESE VALUES
# =============================================================
TEAM_LEAD=""    # Juelich username of your team lead
TEAM_FOLDER=""  # Exact team name from the team lead's setup
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

if [[ -z "$TEAM_LEAD" || -z "$TEAM_FOLDER" || "$TEAM_LEAD" == */* || "$TEAM_FOLDER" == */* || "$TEAM_FOLDER" == "." || "$TEAM_FOLDER" == ".." ]]; then
    _error "Set valid TEAM_LEAD and TEAM_FOLDER values before running."
    return 1
fi

PROJECT_NAME="training2625"
WRITE_TO_BASHRC="yes"

SCRATCH_BASE="/p/scratch/$PROJECT_NAME"
TEAM_DIR="$SCRATCH_BASE/$TEAM_LEAD/$TEAM_FOLDER"
UV_BASE="$TEAM_DIR/.uv"
HF_CACHE="$TEAM_DIR/.cache"

echo ""
echo "============================================="
echo "   Shared Team Setup - Teammates (Step 2/2)"
echo "============================================="
echo ""
echo "  Project     : $PROJECT_NAME"
echo "  Team lead   : $TEAM_LEAD"
echo "  Team folder : $TEAM_DIR"
echo ""

_log "Step 1/4 - Checking cluster tools..."

if ! command -v jutil &> /dev/null; then
    _error "jutil not found. Are you on JURECA?"
    return 1
fi
_ok "jutil found."

_log "Step 2/4 - Activating project $PROJECT_NAME..."

if ! jutil env activate -p "$PROJECT_NAME"; then
    _error "Could not activate project $PROJECT_NAME."
    return 1
fi

if [[ -z "$PROJECT" ]]; then
    _error '\$PROJECT is empty after activation.'
    return 1
fi
_ok "Project activated. \$PROJECT = $PROJECT"

# Verify the team lead has finished the shared setup before writing config.
if [[ ! -d "$TEAM_DIR" ]]; then
    _error "Shared team folder does not exist: $TEAM_DIR"
    echo "       Ask the team lead to finish hackathon_setup.sh first."
    return 1
fi

if [[ ! -r "$TEAM_DIR" || ! -w "$TEAM_DIR" || ! -x "$TEAM_DIR" ]]; then
    _error "You do not have read/write/traverse access to: $TEAM_DIR"
    echo "       Ask the team lead to check the shared-folder ACLs."