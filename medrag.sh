#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  medrag.sh  —  Start / stop the Medical RAG stack
#  Run from inside advanced-rag-poc/:
#    ./medrag.sh          → kill any running instances, start fresh
#    ./medrag.sh stop     → kill everything, exit
# ──────────────────────────────────────────────────────────────

API_PORT=8000
UI_PORT=7860
API_DIR="$(cd "$(dirname "$0")" && pwd)"
UI_DIR="$(cd "$(dirname "$0")/../surgery-rag-ui" && pwd)"
WEB_API_DIR="$(cd "$(dirname "$0")/../scrubref-api" && pwd)"
API_LOG="/tmp/rag_api.log"
UI_LOG="/tmp/rag_ui.log"
CF_LOG="/tmp/rag_cf.log"

# ── colours ───────────────────────────────────────────────────
BOLD="\033[1m"; RESET="\033[0m"
GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"

info()    { echo -e "${CYAN}${BOLD}[medrag]${RESET} $*"; }
success() { echo -e "${GREEN}${BOLD}[medrag]${RESET} $*"; }
warn()    { echo -e "${YELLOW}${BOLD}[medrag]${RESET} $*"; }
error()   { echo -e "${RED}${BOLD}[medrag]${RESET} $*"; }

# ── helpers ───────────────────────────────────────────────────
port_pids() { lsof -ti:"$1" -sTCP:LISTEN 2>/dev/null; }
is_running() { [[ -n "$(port_pids "$1")" ]]; }

stop_services() {
    local killed=0
    for port in $API_PORT $UI_PORT; do
        local pids
        pids=$(port_pids "$port")
        if [[ -n "$pids" ]]; then
            echo "$pids" | xargs kill -9 2>/dev/null
            warn "Killed existing process on port $port"
            killed=1
        fi
    done
    # Kill any running cloudflared tunnel
    if pgrep -f "cloudflared tunnel run medrag" >/dev/null 2>&1; then
        pkill -f "cloudflared tunnel run medrag" 2>/dev/null
        warn "Killed existing cloudflared tunnel"
        killed=1
    fi
    [[ $killed -eq 1 ]] && sleep 1
}

wait_for_api() {
    info "Waiting for API to warm up (loading 4 books into memory)…"
    local max=90 i=0
    while (( i < max )); do
        if curl -sf "http://localhost:${API_PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
        (( i += 2 ))
        # Print progress dot every 10s
        (( i % 10 == 0 )) && echo -ne "    ${i}s elapsed…\r"
    done
    return 1
}

wait_for_ui() {
    local max=20 i=0
    while (( i < max )); do
        if curl -sf "http://localhost:${UI_PORT}/" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        (( i++ ))
    done
    return 1
}

check_db() {
    if [[ ! -f "$WEB_API_DIR/.env" ]]; then
        warn "scrubref-api .env not found — skipping DB check"
        return 0
    fi

    info "Checking Supabase database connection…"

    # Load DATABASE_URL from scrubref-api .env and test with Prisma
    local result
    result=$(cd "$WEB_API_DIR" && node -e "
require('dotenv').config();
const { PrismaClient } = require('@prisma/client');
const p = new PrismaClient();
p.\$connect()
  .then(() => p.\$queryRaw\`SELECT 1\`)
  .then(() => { p.\$disconnect(); console.log('ok'); process.exit(0); })
  .catch(e => { console.error(e.message); process.exit(1); });
" 2>&1)

    if [[ "$result" == "ok" ]]; then
        success "Supabase DB  ✓  connection healthy"
        return 0
    else
        # Extract just the first line of the error (avoid Prisma noise)
        local short_err
        short_err=$(echo "$result" | grep -v "^$" | grep -v "^\s" | head -1)
        error "Supabase DB  ✗  connection failed"
        error "  Error : $short_err"
        error "  Fix   : check DATABASE_URL in $WEB_API_DIR/.env"
        error "          → app.supabase.com → Project Settings → Database → reset password if needed"
        return 1
    fi
}

# ── stop-only mode ────────────────────────────────────────────
if [[ "$1" == "stop" ]]; then
    if is_running $API_PORT || is_running $UI_PORT || pgrep -f "cloudflared tunnel run medrag" >/dev/null 2>&1; then
        stop_services
        success "All services stopped."
    else
        info "Nothing is running."
    fi
    exit 0
fi

# ── start (always restart) ────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}         🏥  Medical RAG  —  Starting up${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

# Kill any existing instances
stop_services

# ── Pre-flight: Supabase DB check ────────────────────────────
echo ""
if ! check_db; then
    warn "Continuing despite DB error — RAG pipeline will still work, but scrubref-api queries will fail."
fi

echo ""

# ── Start API ─────────────────────────────────────────────────
info "Starting FastAPI backend  (port ${API_PORT})…"
info "  Log : $API_LOG"
cd "$API_DIR" || { error "Cannot find $API_DIR"; exit 1; }
nohup venv/bin/uvicorn src.api:app --port "$API_PORT" > "$API_LOG" 2>&1 &
API_PID=$!
info "  PID : $API_PID  (warming up — loading 4 books + cross-encoder…)"

# ── Wait for API ──────────────────────────────────────────────
if wait_for_api; then
    books=$(grep "pipeline_ready" "$API_LOG" | grep -o '"books":\[[^]]*\]' | head -1)
    success "FastAPI backend  ✓  ready on :${API_PORT}  (PID $API_PID)"
    [[ -n "$books" ]] && info "  Books loaded: $(echo "$books" | tr -d '"books:[]' | tr ',' ' | ')"
else
    error "FastAPI backend did not start in time."
    error "  Check logs: $API_LOG"
    exit 1
fi

echo ""

# ── Start Chainlit UI ─────────────────────────────────────────
info "Starting Chainlit UI  (port ${UI_PORT})…"
info "  Log : $UI_LOG"
cd "$UI_DIR" || { error "Cannot find $UI_DIR"; exit 1; }
nohup venv/bin/chainlit run app.py --port "$UI_PORT" > "$UI_LOG" 2>&1 &
UI_PID=$!
info "  PID : $UI_PID"

if wait_for_ui; then
    success "Chainlit UI  ✓  ready on :${UI_PORT}  (PID $UI_PID)"
else
    error "Chainlit UI did not start in time."
    error "  Check logs: $UI_LOG"
    exit 1
fi

# ── Open browser ──────────────────────────────────────────────
echo ""
info "Opening browser…"
sleep 3
open "https://scrubref.shuf.site"

# ── Start Cloudflare tunnel ───────────────────────────────────
echo ""
if command -v cloudflared >/dev/null 2>&1; then
    info "Starting Cloudflare tunnel…"
    info "  Log : $CF_LOG"
    nohup cloudflared tunnel run medrag > "$CF_LOG" 2>&1 &
    CF_PID=$!
    sleep 3
    if pgrep -f "cloudflared tunnel run medrag" >/dev/null 2>&1; then
        success "Cloudflare tunnel  ✓  live  (PID $CF_PID)  →  https://scrubref.shuf.site"
    else
        warn "Cloudflare tunnel did not start."
        warn "  Check logs: $CF_LOG"
    fi
else
    warn "cloudflared not found — skipping tunnel  (brew install cloudflared)"
fi

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
success "Stack is live at ${BOLD}http://localhost:${UI_PORT}${RESET}"
echo -e "  Public URL : ${BOLD}https://scrubref.shuf.site${RESET}"
echo -e "  API logs   : ${API_LOG}"
echo -e "  UI logs    : ${UI_LOG}"
echo -e "  CF logs    : ${CF_LOG}"
echo -e "  To stop    : ${BOLD}./medrag.sh stop${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
