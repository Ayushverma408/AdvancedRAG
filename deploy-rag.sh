#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  deploy-rag.sh  —  Deploy advanced-rag-poc on RAG box
#
#  Usage:
#    ./deploy-rag.sh           → pull + install + restart
#    ./deploy-rag.sh status    → show service status + health
# ──────────────────────────────────────────────────────────────

RAG_DIR="/home/ec2-user/advanced-rag-poc"
RAG_PORT=8000
SERVICE="scrubref-rag"

BOLD="\033[1m"; RESET="\033[0m"
GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"

info()    { echo -e "${CYAN}${BOLD}[deploy]${RESET} $*"; }
success() { echo -e "${GREEN}${BOLD}[deploy]${RESET} $*"; }
warn()    { echo -e "${YELLOW}${BOLD}[deploy]${RESET} $*"; }
err()     { echo -e "${RED}${BOLD}[deploy]${RESET} $*"; }

wait_for_port() {
    local port=$1 max=${2:-30} i=0
    while (( i < max )); do
        curl -sf "http://localhost:${port}/health" >/dev/null 2>&1 && return 0
        sleep 2; (( i += 2 ))
        (( i % 6 == 0 )) && echo -ne "    RAG API: ${i}s elapsed…\r"
    done
    return 1
}

# ── status ────────────────────────────────────────────────────
if [[ "$1" == "status" ]]; then
    echo ""; echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}         🩺  ScrubRef RAG Box  —  Status${RESET}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"; echo ""

    sudo systemctl status "$SERVICE" --no-pager -l | head -20
    echo ""

    HEALTH=$(curl -sf "http://localhost:${RAG_PORT}/health" 2>/dev/null)
    if [[ -n "$HEALTH" ]]; then
        success "RAG API  ✓  port $RAG_PORT responding"
        echo "  $HEALTH"
    else
        err "RAG API  ✗  port $RAG_PORT not responding"
    fi

    echo ""; echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"; echo ""
    exit 0
fi

echo ""; echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}         🩺  ScrubRef  —  Deploying RAG Box${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"; echo ""

cd "$RAG_DIR" || { err "Cannot find $RAG_DIR"; exit 1; }

# ── git pull ──────────────────────────────────────────────────
info "Pulling latest code…"
git pull || { err "git pull failed"; exit 1; }
success "git pull ✓"
echo ""

# ── pip install ───────────────────────────────────────────────
info "Installing Python dependencies…"
source venv/bin/activate || { err "Cannot activate venv — is it set up?"; exit 1; }
pip install -r requirements.txt -q || { err "pip install failed"; exit 1; }
success "pip install ✓"
echo ""

# ── restart ───────────────────────────────────────────────────
info "Restarting $SERVICE…"
sudo systemctl restart "$SERVICE" || { err "systemctl restart failed"; exit 1; }

info "Waiting for RAG API to warm up (~5s)…"
if wait_for_port $RAG_PORT 30; then
    success "RAG API  ✓  running on port $RAG_PORT"
    HEALTH=$(curl -sf "http://localhost:${RAG_PORT}/health" 2>/dev/null)
    echo "  $HEALTH"
else
    err "RAG API did not come up — check: sudo journalctl -u $SERVICE -n 50"
    exit 1
fi

# ── Summary ───────────────────────────────────────────────────
echo ""; echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
success "Deploy complete!"
echo -e "  Status : ${BOLD}./deploy-rag.sh status${RESET}"
echo -e "  Logs   : ${BOLD}sudo journalctl -u $SERVICE -f${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"; echo ""
