#!/bin/bash
set -e

# Helper: detect if script is sourced
script_sourced() {
    [ "${BASH_SOURCE[0]}" != "${0}" ]
}

# Helper: on error, return (never exit) so we don't kill the shell
die() {
    return 1
}

usage() {
    echo "Usage (you can chain multiple actions):"
    echo "  To clean caches/artifacts:               $0 clean"
    echo "  To remove downloaded data (after clean): $0 all"
    echo "  To install regular:                      source $0 install [cpu|cuda121|...]"
    echo "  To install editable (no extras):         source $0 install -e"
    echo "  To install editable with extras:         source $0 install [cpu|cuda121|...] -e"
    echo "  To chain:                                source $0 clean all install cpu"
    die
}

# Ensure we have at least one argument
if [[ $# -lt 1 ]]; then
    usage
fi

# Determine if “install” appears among the arguments
install_requested=false
for arg in "$@"; do
    if [[ "$arg" == "install" ]]; then
        install_requested=true
        break
    fi
done

# If install is requested, require sourcing so that activation persists
if $install_requested && ! script_sourced; then
    echo "Error: To install and keep the venv activated, you must source this script."
    echo "  Example: source setup.sh install cpu"
    die
fi

# Shared cleanup steps
common_clean() {
    echo "Removing virtual environments, caches, build artifacts, and temporary files..."

    # Virtual environments
    rm -rf .venv venv env .env

    # Python bytecode caches
    find . -type d -name "__pycache__"    -exec rm -rf {} +
    find . -type f -name "*.py[co]"        -delete

    # Build/dist/packaging artifacts
    rm -rf build dist **/*.egg-info pip-wheel-metadata

    # Test and coverage caches
    rm -rf .pytest_cache .mypy_cache .coverage htmlcov

    # Tox environments
    rm -rf .tox

    # Editor/IDE caches (optional)
    rm -rf .vscode .idea

    # Jupyter checkpoints
    find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

    # macOS metadata
    find . -name ".DS_Store" -delete

    # Project-specific outputs
    rm -rf logs
}

# Process each subcommand in order
while [[ $# -gt 0 ]]; do
    case "$1" in
        clean)
            common_clean
            echo "Clean complete."
            shift
            ;;
        all)
            echo "Removing downloaded data and results..."
            rm -rf data results
            echo "Data and results removal complete."
            shift
            ;;
        install)
            shift

            # Parse -e and extras manually
            EDITABLE=false
            EXTRAS=""
            new_args=()
            for arg in "$@"; do
                if [[ "$arg" == "-e" ]]; then
                    EDITABLE=true
                else
                    new_args+=("$arg")
                fi
            done
            # Reset positional parameters to what's left after removing -e
            set -- "${new_args[@]}"

            # Now $1 (if present) is the extra; otherwise empty
            EXTRAS=${1:-}

            # Create venv if missing
            if [[ ! -d ".venv" ]]; then
                echo "Creating virtual environment..."
                uv venv .venv --python 3.10
            else
                echo ".venv already exists; skipping creation."
            fi

            # Activate (persists because script is sourced)
            echo "Activating virtual environment..."
            source .venv/bin/activate

            # Decide install command
            if $EDITABLE; then
                if [[ -n "$EXTRAS" ]]; then
                    echo "Installing in editable mode with extras: [$EXTRAS]"
                    uv pip install -e ".[${EXTRAS}]"
                else
                    echo "Installing in editable mode (no extras)..."
                    uv pip install -e .
                fi
            else
                if [[ -n "$EXTRAS" ]]; then
                    echo "Installing dependencies for: $EXTRAS"
                    uv pip install ".[${EXTRAS}]"
                else
                    echo "No extra specified; installing base package"
                    uv pip install .
                fi
            fi

            echo "Installation complete."
            shift
            ;;
        *)
            usage
            ;;
    esac
done

# If script was executed (not sourced) and no install was requested, remind user
if ! script_sourced && ! $install_requested; then
    echo "Hint: For install, run 'source setup.sh install ...' so activation persists."
fi
