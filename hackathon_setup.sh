_log "Step 6/6 - Downloading datasets and creating per-task environments..."

if [[ -n "$HF_TOKEN" ]]; then
    _ok "HF_TOKEN found - will be used for authentication."
else
    _log "No HF_TOKEN set - proceeding without (fine for public datasets)."
fi

BOOTSTRAP_VENV="$TEAM_DIR/.bootstrap"
if [[ ! -d "$BOOTSTRAP_VENV" ]]; then
    _log "Creating bootstrap venv for downloads..."
    uv venv -p 3.12 "$BOOTSTRAP_VENV" || return 1
    VIRTUAL_ENV="$BOOTSTRAP_VENV" uv pip install huggingface_hub || return 1
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

    if [[ -d "$dest" ]]; then
        _warn "  Dataset directory already exists. Skipping download."
    else
        _log "  Downloading $dataset..."
        "$BOOTSTRAP_PYTHON" - "$dataset" "$dest" <<'PYEOF'
import os
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    repo_type="dataset",
    local_dir=sys.argv[2],
    token=os.environ.get("HF_TOKEN"),
)
PYEOF
        if [[ $? -ne 0 ]]; then
            _error "  Download failed - skipping venv setup."
            continue
        fi
        _ok "  Downloaded $dataset_name."
    fi

    mkdir -p "$dest/output" || { _error "Cannot create output directory"; continue; }
    setfacl -m m:rwx "$dest/output"
    setfacl -d -m m:rwx "$dest/output"

    if [[ -d "$VENV_DIR" ]]; then
        _warn "  Venv already exists. Skipping."
        continue
    fi

    if [[ ! -f "$REQ" ]]; then
        _warn "  No requirements.txt found in $dataset_name - skipping venv."
        continue
    fi

    _log "  Setting up uv project for $dataset_name..."
    cd "$dest" || { _error "  Cannot cd into $dest"; continue; }

    if [[ ! -f pyproject.toml ]]; then
        uv init -p 3.12 --no-workspace || { _error "  uv init failed"; cd "$TEAM_DIR"; continue; }
    fi

    uv venv -p 3.12 "$VENV_DIR" || { _error "  uv venv failed"; cd "$TEAM_DIR"; continue; }

    # Remove standalone comments and blank lines from requirements.
    # Original behavior uses uv add to create pyproject.toml and uv.lock.
    mapfile -t DEPS < <(grep -vE '^[[:space:]]*(#|$)' "$REQ")

    if [[ "${#DEPS[@]}" -eq 0 ]]; then
        _warn "  requirements.txt is empty - skipping uv add."
    else
        if VIRTUAL_ENV="$VENV_DIR" uv add "${DEPS[@]}"; then
            _ok "  Done: pyproject.toml + uv.lock + .venv ready."
        else
            _error "  Failed to install requirements for $dataset_name."
        fi
    fi

    cd "$TEAM_DIR" || return 1
done

if [[ -d "$BOOTSTRAP_VENV" ]]; then
    rm -rf "$BOOTSTRAP_VENV"
    _ok "Bootstrap venv removed."
fi

echo ""
echo "============================================="
echo -e "${GREEN}   Team lead setup finished!${NC}"
echo "============================================="
echo "  Team access : $TEAM_LEAD + ${TEAMMATES[*]}"
echo "  uv          : $(uv --version 2>&1)"
echo ""
echo "  NEXT: Each remaining team member must now run:"
echo "        source teammate.sh"
echo "  with TEAM_LEAD=$TEAM_LEAD and TEAM_FOLDER=$TEAM_FOLDER"
echo ""
echo "  Activate a task environment:"
echo "  source $TEAM_DIR/<dataset-name>/.venv/bin/activate"
echo ""
echo "  Submit a job: sbatch job.sh"
echo "============================================="
echo ""