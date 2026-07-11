#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  AccessLearn — Setup & Run
#  Usage:  bash run.sh          (installs deps + starts server)
#          bash run.sh --noinstall  (skip install, just run)
# ─────────────────────────────────────────────────────────────────

set -e
CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║   AccessLearn — Inclusive AI Learning    ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${NC}"

# ── Check Python ──────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo -e "${RED}❌ Python 3 not found. Install from https://python.org${NC}"
  exit 1
fi
echo -e "${GREEN}✅ Python: $(python3 --version)${NC}"

# ── Virtual environment ───────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "📦 Creating virtual environment…"
  python3 -m venv .venv
fi
source .venv/bin/activate

# ── Install dependencies ──────────────────────────────────────────
if [ "$1" != "--noinstall" ]; then
  echo "📦 Installing dependencies (this may take a minute)…"
  pip install --upgrade pip -q
  pip install -r requirements.txt -q
  # ffmpeg check (needed by pydub)
  if ! command -v ffmpeg &>/dev/null; then
    echo -e "\n⚠️  ffmpeg not found. Install it for MP3 support:"
    echo "    macOS  : brew install ffmpeg"
    echo "    Ubuntu : sudo apt install ffmpeg"
    echo "    Windows: https://ffmpeg.org/download.html"
    echo ""
  fi
  echo -e "${GREEN}✅ Dependencies installed${NC}"
fi

# ── Set Gemini API key ────────────────────────────────────────────
export GEMINI_API_KEY="${GEMINI_API_KEY:-AIzaSyCAmlqHeDK_95FYvLIgcdV-z6W8Xd8_yak}"

# ── Launch ────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}🚀 Starting AccessLearn on http://localhost:5000${NC}"
echo "   Press Ctrl+C to stop"
echo ""
python3 app.py
