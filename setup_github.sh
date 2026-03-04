#!/usr/bin/env bash
# Run this from the dashboard/ directory after cloning or creating the repo on GitHub
# Usage: bash setup_github.sh

set -e

REPO_URL="https://github.com/wernerhl/ledgers-bolivia.git"

echo "=== Ledgers of the Self-Employed — GitHub Setup ==="
echo ""

# Check git
if ! command -v git &>/dev/null; then
  echo "Error: git not found. Install git first."
  exit 1
fi

# Init if needed
if [ ! -d ".git" ]; then
  git init
  git remote add origin "$REPO_URL"
else
  echo "Git repo already initialized."
fi

# Create folder structure
mkdir -p code paper

# Move code files
[ -f simulate_diary.py ]      && mv simulate_diary.py code/
[ -f ledgers_analytics.py ]   && mv ledgers_analytics.py code/
[ -f paper_illustrations.py ] && mv paper_illustrations.py code/

# Copy paper if present
[ -f ledgers_paper.pdf ] && cp ledgers_paper.pdf paper/

echo ""
echo "Files ready. Now:"
echo "  1. Create the repo at: https://github.com/new"
echo "     Name: ledgers-bolivia  |  Public  |  No README"
echo ""
echo "  2. Run:"
echo "     git add ."
echo "     git commit -m 'Initial commit — Ledgers of the Self-Employed dashboard'"
echo "     git branch -M main"
echo "     git push -u origin main"
echo ""
echo "  3. Enable GitHub Pages:"
echo "     → Settings → Pages → Source: GitHub Actions"
echo ""
echo "  4. Dashboard will be live at:"
echo "     https://wernerhl.github.io/ledgers-bolivia"
