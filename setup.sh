#!/usr/bin/env bash
# ============================================================================ #
#                  DonaDataset — Environment Setup Script
# ============================================================================ #
# Sets up a Python virtual environment with uv and installs the dependencies
# needed to run the download and validation scripts.
#
# Usage:
#   ./setup.sh
# ============================================================================ #

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}ℹ${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error()   { echo -e "${RED}✗${NC} $1"; }

print_header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                DonaDataset — Environment Setup                   ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
}

check_uv() {
    if command -v uv &> /dev/null; then
        print_success "uv is already installed: $(uv --version)"
        return 0
    else
        return 1
    fi
}

install_uv() {
    print_info "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    for p in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
        [[ -f "$p/uv" ]] && export PATH="$p:$PATH" && break
    done
    if command -v uv &> /dev/null; then
        print_success "uv installed: $(uv --version)"
    else
        print_error "Failed to install uv. Install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
}

create_venv() {
    print_info "Creating virtual environment in ${VENV_DIR}..."
    if [[ -d "$VENV_DIR" ]]; then
        print_warning "Virtual environment already exists — recreating..."
        rm -rf "$VENV_DIR"
    fi
    uv venv "$VENV_DIR"
    print_success "Virtual environment created"
}

install_dependencies() {
    print_info "Installing dependencies (scripts group)..."
    source "$VENV_DIR/bin/activate"
    uv sync --group scripts
    print_success "Dependencies installed"
}

print_next_steps() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                       Setup complete!                            ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    print_info "Activate the virtual environment:"
    echo ""
    echo "  source .venv/bin/activate"
    echo ""
    print_info "Available commands:"
    echo "  python scripts/download.py                  Download all splits"
    echo "  python scripts/download.py --split train    Download a single split"
    echo "  python scripts/validate.py                  Validate dataset integrity"
    echo "  python scripts/validate.py --split train    Validate a single split"
    echo ""
}

print_header
cd "$SCRIPT_DIR"

check_uv || install_uv
create_venv
install_dependencies
print_next_steps
