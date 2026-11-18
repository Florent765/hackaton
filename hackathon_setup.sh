#!/bin/bash
# =============================================================
# HACKATHON SETUP: STEP 1 OF 2 - TEAM LEAD RUNS FIRST
# =============================================================
# EXECUTION ORDER (IMPORTANT):
# 1. The TEAM LEAD configures and runs THIS script first.
#    Run: source hackathon_setup.sh
#    Wait until it finishes successfully.
# 2. ONLY AFTER THAT, each remaining team member configures
#    and runs teammate.sh from their own JURECA account.
#    Run: source teammate.sh
#
# The team lead runs only this script. Teammates run only
# teammate.sh. Do not run both scripts from the same account.
#
# Before running, set TEAM_LEAD to your Juelich username,
# TEAMMATE_1 through TEAMMATE_3 to your teammates' Juelich
# usernames, and TEAM_FOLDER to the agreed team name.
# =============================================================

# =============================================================
# CONFIGURATION - TEAM LEAD EDITS THESE VALUES
# =============================================================
TEAM_LEAD=""    # Your Juelich username
TEAMMATE_1=""   # Teammate's Juelich username; leave blank if unused
TEAMMATE_2=""
TEAMMATE_3=""
TEAM_FOLDER=""  # Your team name; teammates must use the exact same value
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

if [[ -z "$TEAM_LEAD" || -z "$TEAM_FOLDER" || "$TEAM_LEAD" == */* || "$TEAM_FOLDER" == */* || "$TEAM_FOLDER" == "." || "$TEAM_FOLDER" == ".." ]]; then
    _error "Set valid TEAM_LEAD and TEAM_FOLDER values before running."
    return 1
fi

if [[ "$TEAM_LEAD" != "$(id -un)" ]]; then
    _error "TEAM_LEAD must match the username running this script: $(id -un)"
    return 1
fi

# Hugging Face datasets to download (org/repo format)
HF_DATASETS=(
    "SprintML/unlearning"
    "SprintML/model_equality"
    "SprintML/Hackathon_Watermark_Localisation_Task"
)

PROJECT_NAME="training2666"
SCRATCH_BASE="/p/scratch/$PROJECT_NAME"
TEAM_LEAD_DIR="$SCRATCH_BASE/$TEAM_LEAD"
TEAM_DIR="$TEAM_LEAD_DIR/$TEAM_FOLDER"

# Build teammates array, skipping blank entries
TEAMMATES=()
for t in "$TEAMMATE_1" "$TEAMMATE_2" "$TEAMMATE_3"; do
    [[ -n "$t" ]] && TEAMMATES+=("$t")
done

if [[ "${#TEAMMATES[@]}" -eq 0 ]]; then
    _error "No teammates specified. Add at least one."
    return 1
fi

for t in "${TEAMMATES[@]}"; do
    if [[ "$t" == */* || "$t" == "$TEAM_LEAD" ]]; then
        _error "Invalid teammate username: $t"
        return 1
    fi
done

echo ""
echo "============================================="
echo "   Hackathon Setup - Team Lead (Step 1/2)"
echo "============================================="
echo ""
echo "  Team lead   : $TEAM_LEAD"
echo "  Teammates   : ${TEAMMATES[*]}"
echo "  Lead folder : $TEAM_LEAD_DIR"
echo "  Team folder : $TEAM_DIR"
echo ""