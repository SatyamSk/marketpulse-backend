from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import pandas as pd
import numpy as np
import os, json, subprocess, sys, threading, traceback, hashlib, hmac, time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_DIR        = os.path.dirname(os.path.abspath(__file__))
LOG_FILE        = os.path.join(DATA_DIR, "pipeline_live.log")
client          = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PIPELINE_SECRET = os.getenv("PIPELINE_SECRET", "marketpulse2024")
ADMIN_SECRET    = os.getenv("ADMIN_SECRET", PIPELINE_SECRET)
IST             = timezone(timedelta(hours=5, minutes=30))

import database as db
import requests as http_requests

app = FastAPI(title="MarketPulse AI API")

# ── Durable storage fallback (GitHub Data Repo) ─────────────────────
GITHUB_DATA_REPO = os.getenv("GITHUB_DATA_REPO", "SatyamSk/MarketPulseAIData")
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_DATA_REPO}/main"

def _fetch_raw_text(path: str) -> str | None:
    try:
        r = http_requests.get(f"{RAW_BASE}/{path.lstrip('/')}", timeout=15)
        if not r.ok:
            return None
        return r.text
    except Exception:
        return None

def _fetch_github_snapshot() -> tuple[list[dict], list[dict], dict | None, str | None]:
    """
    Returns: (headlines, sectors, msi, last_run_str)
    """
    headlines_csv = _fetch_raw_text("latest_headlines.csv")
    sectors_csv = _fetch_raw_text("latest_sectors.csv")
    msi_json = _fetch_raw_text("latest_msi.json")
    status_json = _fetch_raw_text("pipeline_status.json")

    if not headlines_csv or not sectors_csv:
        return ([], [], None, None)

    try:
        hdf = pd.read_csv(pd.io.common.StringIO(headlines_csv))
        sdf = pd.read_csv(pd.io.common.StringIO(sectors_csv))
        headlines = hdf.replace({np.nan: None}).to_dict(orient="records")
        sectors = sdf.replace({np.nan: None}).to_dict(orient="records")
    except Exception:
        return ([], [], None, None)

    msi = None
    if msi_json:
        try:
            msi = json.loads(msi_json)
        except Exception:
            msi = None

    last_run = None
    if status_json:
        try:
            last_run = json.loads(status_json).get("last_run")
        except Exception:
            last_run = None

    return (headlines, sectors, msi, last_run)

# ── KEEP-ALIVE: Prevent Render free tier from sleeping ─────────────
def _keep_alive_worker():
    """Self-ping every 13 minutes to prevent Render spin-down."""
    import time as _time
    render_url = os.getenv("RENDER_EXTERNAL_URL", "")
    if not render_url:
        print("  [i] RENDER_EXTERNAL_URL not set — keep-alive disabled.")
        return
    health_url = f"{render_url}/api/health"
    print(f"  ✓ Keep-alive active: pinging {health_url} every 13 min")
    while True:
        _time.sleep(780)  # 13 minutes
        try:
            http_requests.get(health_url, timeout=10)
        except Exception:
            pass

@app.on_event("startup")
def startup_event():
    """Start keep-alive thread on server boot."""
    threading.Thread(target=_keep_alive_worker, daemon=True).start()

# ── SECURITY: Restricted CORS ─────────────────────────────────────
ALLOWED_ORIGINS = [
    "https://marketpulsewithai.vercel.app",
    "http://localhost:5173", "http://localhost:3000", "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ── RATE LIMITING ──────────────────────────────────────────────────
chat_limiter: dict = defaultdict(list)
CHAT_LIMIT = 15  # per minute
brief_usage: dict = defaultdict(list)
BRIEF_MAX = 2

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")

def check_chat_rate(request: Request):
    ip = get_client_ip(request)
    now = time.time()
    chat_limiter[ip] = [t for t in chat_limiter[ip] if now - t < 60]
    if len(chat_limiter[ip]) >= CHAT_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 15 requests/minute.")
    chat_limiter[ip].append(now)

def verify_admin(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization token.")
    token = auth.replace("Bearer ", "")
    expected = hashlib.sha256(ADMIN_SECRET.encode()).hexdigest()
    if not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid admin token.")

# ── HELPERS ────────────────────────────────────────────────────────
def safe(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    if isinstance(v, (bool, np.bool_)): return bool(v)
    if isinstance(v, (int, np.integer)): return int(v)
    if isinstance(v, (float, np.floating)): return float(v)
    return str(v)

def to_records(data: list[dict]) -> list[dict]:
    return [{k: safe(v) for k, v in row.items()} for row in data] if data else []

def classify_regime(avg_nss: float, avg_risk: float) -> dict:
    if avg_nss > 20 and avg_risk < 20:
        return {"regime": "Risk On", "description": "Broad bullish sentiment, low systemic risk. Momentum trades favored.", "nifty_implication": "Gap-up open likely. Momentum trades have higher probability.", "watch": "High-momentum sectors showing positive velocity.", "avoid": "Defensive over-positioning not needed in Risk On conditions."}
    elif avg_nss < -20 and avg_risk > 35:
        return {"regime": "Panic", "description": "Widespread negative sentiment with high systemic risk. Defensive only.", "nifty_implication": "Heavy selling pressure expected. Watch key support levels.", "watch": "Defensive sectors — Banking if NSS is stable.", "avoid": "All high-beta positions. Reduce exposure immediately."}
    elif avg_nss > 0 and avg_risk > 25:
        return {"regime": "Complacent", "description": "Positive headlines masking elevated underlying risk. Watch for reversal.", "nifty_implication": "Deceptively calm open possible. Reversal risk elevated.", "watch": "Divergence signals — sectors where NSS and impact-weighted disagree.", "avoid": "Overleveraged positions. Risk is higher than headlines suggest."}
    else:
        return {"regime": "Risk Off", "description": "Cautious market conditions. Capital preservation favored.", "nifty_implication": "Flat to gap-down open likely. Avoid chasing early moves.", "watch": "Sectors with positive velocity — early recovery signs.", "avoid": "High-leverage positions and sectors with negative velocity."}

# ── HEALTH CHECK ───────────────────────────────────────────────────
@app.get("/api/health")
def health_check():
    pipeline_info = db.get_latest_pipeline_info()
    return {
        "status": "ok",
        "timestamp": datetime.now(IST).isoformat(),
        "database": "connected",
        "last_pipeline": pipeline_info.get("completed_at") if pipeline_info else None,
    }

@app.get("/api/version")
def version():
    # Render exposes these for builds; if absent, still return something useful.
    return {
        "service": "marketpulse-backend",
        "render_git_commit": os.getenv("RENDER_GIT_COMMIT"),
        "render_service_id": os.getenv("RENDER_SERVICE_ID"),
        "timestamp": datetime.now(IST).isoformat(),
    }

# ── DASHBOARD ──────────────────────────────────────────────────────
@app.get("/api/dashboard")
def get_dashboard():
    headlines_raw = db.get_latest_headlines()
    sectors_raw = db.get_latest_sectors()
    pipeline_info = db.get_latest_pipeline_info()
    accuracy = db.get_accuracy_stats(30)

    # IMPORTANT: Never 404 here. The frontend treats non-200 as "pipeline disconnected".
    # Instead return an empty-but-valid dashboard payload so the UI can render a proper "no data yet" state.
    if not headlines_raw:
        # Fallback: try to load the latest snapshot from GitHub durable storage.
        gh_headlines, gh_sectors, gh_msi, gh_last_run = _fetch_github_snapshot()
        if gh_headlines:
            headlines_raw = gh_headlines
            sectors_raw = gh_sectors
            pipeline_info = pipeline_info or {"completed_at": gh_last_run, "msi": (gh_msi or {}).get("msi", 0), "msi_level": (gh_msi or {}).get("level", "Low")}
        else:
        empty_regime = {
            "regime": "Risk Off",
            "description": "No pipeline run detected yet. Run the pipeline (Admin → Run Agent Intelligence) to generate today's analysis.",
            "nifty_implication": "No call available until the first run completes.",
            "watch": "—",
            "avoid": "—",
        }
        return {
            "last_updated": None,
            "market_regime": empty_regime,
            "benchmark": [],
            "headlines": [],
            "pareto": [],
            "contagion_flows": [],
            "velocity_trend": [],
            "shock_headlines": [],
            "shock_counts": {"major": 0, "shock": 0, "watch": 0},
            "market_stress_index": {"msi": 0, "level": "Low"},
            "model_accuracy": {"accuracy": 0, "total_predictions": 0, "correct": 0},
            "summary_stats": {
                "total_headlines": 0,
                "geopolitical_flags": 0,
                "avg_nss": 0,
                "avg_risk": 0,
            },
        }

    headlines_df = pd.DataFrame(headlines_raw)
    benchmark_df = pd.DataFrame(sectors_raw)

    # Geopolitical filter
    geo_hl = [h for h in headlines_raw if h.get("geopolitical_risk")]

    avg_nss  = float(benchmark_df["composite_sentiment_index"].mean()) if not benchmark_df.empty else 0.0
    avg_risk = float(benchmark_df["avg_weighted_risk"].mean()) if not benchmark_df.empty else 0.0
    regime   = classify_regime(avg_nss, avg_risk)

    # Pareto
    pareto = []
    if not benchmark_df.empty:
        pareto_df = benchmark_df.sort_values("avg_weighted_risk", ascending=False).copy()
        total_risk = max(pareto_df["avg_weighted_risk"].sum(), 1)
        pareto_df["cumulative_pct"] = (pareto_df["avg_weighted_risk"].cumsum() / total_risk * 100).round(1)
        pareto = pareto_df[["sector", "avg_weighted_risk", "cumulative_pct"]].to_dict(orient="records")

    # Contagion flows
    contagion = []
    if geo_hl:
        sector_impacts = {}
        for h in geo_hl:
            s = h.get("sector", "Other")
            sector_impacts.setdefault(s, []).append(float(h.get("impact_score", 5)))
        contagion = [{"source": "Geopolitical Event", "target": s, "value": round(np.mean(v), 1)} for s, v in sector_impacts.items()]

    # Shock headlines
    shock_headlines = [h for h in headlines_raw if h.get("shock_status") in ("Major Shock", "Shock", "Watch")]
    shock_counts = {
        "major": len([h for h in shock_headlines if h.get("shock_status") == "Major Shock"]),
        "shock": len([h for h in shock_headlines if h.get("shock_status") == "Shock"]),
        "watch": len([h for h in shock_headlines if h.get("shock_status") == "Watch"]),
    }

    # MSI
    msi = {"msi": pipeline_info.get("msi", 0), "level": pipeline_info.get("msi_level", "Low")} if pipeline_info else {"msi": 0, "level": "Low"}

    return {
        "last_updated": pipeline_info.get("completed_at") if pipeline_info else None,
        "market_regime": regime,
        "benchmark": to_records(sectors_raw),
        "headlines": to_records(sorted(headlines_raw, key=lambda h: float(h.get("impact_score", 0)), reverse=True)),
        "pareto": pareto,
        "contagion_flows": contagion,
        "velocity_trend": [],
        "shock_headlines": to_records(shock_headlines),
        "shock_counts": shock_counts,
        "market_stress_index": msi,
        "model_accuracy": {"accuracy": accuracy["accuracy"], "total_predictions": accuracy["total"], "correct": accuracy["correct"]},
        "summary_stats": {
            "total_headlines": len(headlines_raw),
            "geopolitical_flags": len(geo_hl),
            "avg_nss": round(avg_nss, 1),
            "avg_risk": round(avg_risk, 1),
        },
    }

# ── CHAT ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., max_length=2000)
    history: list = Field(default_factory=list, max_length=20)
    context_headlines: list = Field(default_factory=list, max_length=25)
    context_sectors: list = Field(default_factory=list, max_length=15)

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest, request: Request):
    check_chat_rate(request)

    sector_ctx = "\n".join([
        f"• {s.get('sector',''):12} | Risk {s.get('avg_weighted_risk', 0)} ({s.get('risk_level','')}) "
        f"| CSI {s.get('composite_sentiment_index', 0)} | Signal: {s.get('investment_signal', '')}"
        for s in req.context_sectors[:10]
    ])
    hl_ctx = "\n".join([
        f"• [{h.get('sector','')}] {h.get('title','')} | {str(h.get('sentiment','')).upper()} "
        f"| Impact: {h.get('impact_score','')}/10 | {h.get('one_line_insight','')}"
        for h in req.context_headlines[:20]
    ])
    has_data = bool(req.context_sectors or req.context_headlines)

    system_prompt = f"""You are MarketPulse AI — a sharp, friendly market intelligence assistant for Indian intraday traders.

BEHAVIOR RULES:
1. For greetings or casual messages — respond naturally and warmly in 1-2 sentences.
2. For ANY market question — answer strictly using the data below. Be punchy and direct.
3. For stock-specific questions — use sector data and headlines to infer. Don't say you don't have data if the sector is covered.
4. Never say "I don't have specific stock recommendations" — give what you DO know.
5. Always end market answers with one concrete forward-looking implication.

TODAY'S DATA ({datetime.now(IST).strftime('%d %B %Y')}):
{'Data available.' if has_data else 'No pipeline data loaded yet.'}

SECTOR SCORES:
{sector_ctx if sector_ctx else 'No sector data.'}

TOP HEADLINES:
{hl_ctx if hl_ctx else 'No headlines.'}"""

    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m["role"], "content": m["content"]} for m in req.history[-6:]]
    messages.append({"role": "user", "content": req.message[:2000]})

    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.3, max_tokens=450)
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI service temporarily unavailable: {str(e)[:100]}")

# ── PIPELINE ───────────────────────────────────────────────────────
@app.get("/api/pipeline/status")
def pipeline_status():
    info = db.get_latest_pipeline_info()
    lock_path = os.path.join(DATA_DIR, "pipeline.lock")
    is_running = False
    if os.path.exists(lock_path):
        age = datetime.now().timestamp() - os.path.getmtime(lock_path)
        is_running = age < 900

    # If Render wiped the DB, fall back to GitHub snapshot status.
    if not info:
        _, _, _, gh_last_run = _fetch_github_snapshot()
        if gh_last_run:
            return {
                "last_headlines_update": gh_last_run,
                "headlines_count": 0,
                "is_running": is_running,
                "data_available": True,
                "storage": "github",
            }
    return {
        "last_headlines_update": info.get("completed_at") if info else None,
        "headlines_count": info.get("headline_count", 0) if info else 0,
        "is_running": is_running,
        "data_available": info is not None,
        "storage": "sqlite",
    }

class PipelineRequest(BaseModel):
    secret: str
    max_per_feed: int = 12
    # Legacy/static mode intentionally removed. Everything runs through agent.

@app.post("/api/pipeline/run")
def trigger_pipeline(req: PipelineRequest, request: Request):
    verify_admin(request)
    if req.secret != PIPELINE_SECRET:
        raise HTTPException(status_code=401, detail="Invalid secret key.")

    max_per_feed = max(3, min(50, req.max_per_feed))
    total_approx = max_per_feed * 16
    use_agent = True

    def run():
        lock_path = os.path.join(DATA_DIR, "pipeline.lock")
        with open(lock_path, "w") as f:
            f.write(datetime.now().isoformat())
        try:
            with open(LOG_FILE, "w") as log_f:
                mode_label = "AGENT" if use_agent else "LEGACY"
                log_f.write(f"🚀 {mode_label} pipeline at {datetime.now(IST).strftime('%H:%M:%S IST')}\n{'='*50}\n")
                log_f.flush()

            if use_agent:
                # Run autonomous agent
                from agent import run_agent_pipeline
                result = run_agent_pipeline()
                with open(LOG_FILE, "a") as f:
                    f.write(f"\nAgent result: {json.dumps(result, default=str)[:2000]}\n")
                    f.write(f"\n{'='*50}\n✅ Agent pipeline complete.\n")
            else:
                # Legacy pipeline
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                with open(LOG_FILE, "a") as log_f:
                    subprocess.run(
                        [sys.executable, os.path.join(DATA_DIR, "pipeline.py"), f"--max-per-feed={max_per_feed}"],
                        cwd=DATA_DIR, env=env, stdout=log_f, stderr=subprocess.STDOUT,
                    )
                with open(LOG_FILE, "a") as f:
                    f.write(f"\n{'='*50}\n✅ Legacy pipeline complete.\n")
        except Exception as e:
            with open(LOG_FILE, "a") as f:
                f.write(f"\n🔥 ERROR: {e}\n{traceback.format_exc()}")
        finally:
            if os.path.exists(lock_path):
                os.remove(lock_path)

    threading.Thread(target=run, daemon=True).start()
    msg = "Agent intelligence gathering started"
    return {
        "status": "started",
        "mode": "agent",
        "message": msg,
        "started_at": datetime.now(IST).isoformat(),
    }

# ── AGENT RESULT ───────────────────────────────────────────────────
@app.get("/api/agent/result")
def get_agent_result():
    result_path = os.path.join(DATA_DIR, "agent_result.json")
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="No agent analysis available yet. Run the agent pipeline first.")
    with open(result_path, "r") as f:
        return json.load(f)

# ── BRIEF ──────────────────────────────────────────────────────────
@app.get("/api/brief/status")
def brief_status(request: Request):
    ip = get_client_ip(request)
    now = datetime.now()
    cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    brief_usage[ip] = [t for t in brief_usage[ip] if t >= cutoff]
    used = len(brief_usage[ip])
    remaining = max(0, BRIEF_MAX - used)
    return {"allowed": remaining > 0, "used": used, "remaining": remaining, "limit": BRIEF_MAX}

class BriefRequest(BaseModel):
    top_headlines: list = Field(default_factory=list, max_length=10)
    sector_summary: list = Field(default_factory=list, max_length=15)
    regime: dict

@app.post("/api/brief")
def generate_brief(request: Request, req: BriefRequest):
    ip = get_client_ip(request)
    now = datetime.now()
    cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    brief_usage[ip] = [t for t in brief_usage[ip] if t >= cutoff]
    if len(brief_usage[ip]) >= BRIEF_MAX:
        raise HTTPException(status_code=429, detail={"error": "daily_limit_reached", "message": "Both daily brief generations used. Resets at midnight.", "used": len(brief_usage[ip]), "limit": BRIEF_MAX, "remaining": 0})

    brief_usage[ip].append(now)
    used_now = len(brief_usage[ip])
    remaining_now = max(0, BRIEF_MAX - used_now)

    hl_lines = []
    for h in req.top_headlines[:8]:
        shock = h.get("shock_status", "Normal")
        tag = " MAJOR SHOCK" if shock == "Major Shock" else " SHOCK" if shock == "Shock" else ""
        hl_lines.append(f"• [{h.get('sector','')}] {h.get('title','')} (Impact: {h.get('impact_score','')}/10, {str(h.get('sentiment','')).upper()}{tag})")

    sec_lines = []
    for s in sorted(req.sector_summary, key=lambda x: x.get("avg_weighted_risk", 0), reverse=True):
        sec_lines.append(f"• {s.get('sector',''):12} | Risk {s.get('avg_weighted_risk', 0):5.1f} | CSI {s.get('composite_sentiment_index', 0):+6.1f} | {s.get('risk_level','')}")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior market strategist writing a forward-looking market outlook for Indian intraday traders. Frame everything as EXPECTED conditions. Use markdown bold for key numbers. No disclaimers. No fluff. Every sentence must contain a specific number, sector name, or directional call."},
                {"role": "user", "content": f"Write a forward-looking market outlook:\n\n## Expected Regime: {req.regime.get('regime', '')}\n## Nifty Expected Move\n## Highest Probability Risk Today\n## Sector Outlook\n## Key Events to Watch\n## Trading Implication\n\n---\nRegime: {req.regime.get('regime', '')} — {req.regime.get('description', '')}\nNifty: {req.regime.get('nifty_implication', '')}\n\nSECTOR DATA:\n{''.join(sec_lines)}\n\nTOP HEADLINES:\n{''.join(hl_lines)}"}
            ],
            temperature=0.25, max_tokens=650,
        )
        return {"brief": response.choices[0].message.content, "used": used_now, "remaining": remaining_now, "limit": BRIEF_MAX}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI service temporarily unavailable: {str(e)[:100]}")

# ── NEW: ACCURACY & HISTORY ENDPOINTS ──────────────────────────────
@app.get("/api/accuracy")
def get_accuracy():
    stats_7 = db.get_accuracy_stats(7)
    stats_30 = db.get_accuracy_stats(30)
    stats_90 = db.get_accuracy_stats(90)
    sources = db.get_source_reliability()
    weights = db.get_dynamic_weights("sector")
    return {
        "accuracy_7d": stats_7,
        "accuracy_30d": stats_30,
        "accuracy_90d": stats_90,
        "source_reliability": sources,
        "dynamic_weights": weights,
    }

@app.get("/api/history")
def get_history(sector: str = None, days: int = 30):
    days = min(days, 90)
    history = db.get_sector_history(sector, days)
    return {"history": history, "days": days, "sector": sector}

@app.get("/api/stocks/search")
def search_stocks(q: str = ""):
    if len(q) < 2:
        raise HTTPException(status_code=400, detail="Search query must be at least 2 characters.")
    results = db.search_stock(q)
    # Aggregate stats
    if results:
        sentiments = [r.get("sentiment", "neutral") for r in results]
        avg_impact = round(np.mean([float(r.get("impact_score", 5)) for r in results]), 1)
        pos = sentiments.count("positive")
        neg = sentiments.count("negative")
        return {
            "query": q, "total": len(results), "headlines": results,
            "aggregate": {"avg_impact": avg_impact, "positive": pos, "negative": neg, "neutral": len(results) - pos - neg,
                          "net_sentiment": "bullish" if pos > neg else "bearish" if neg > pos else "neutral"}
        }
    return {"query": q, "total": 0, "headlines": [], "aggregate": None}

# ── ADMIN AUTH ─────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    password: str = Field(..., max_length=200)

@app.post("/api/admin/login")
def admin_login(req: LoginRequest):
    if req.password != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = hashlib.sha256(ADMIN_SECRET.encode()).hexdigest()
    return {"token": token, "message": "Authenticated."}

@app.get("/api/admin/logs")
def view_logs(request: Request):
    verify_admin(request)
    if not os.path.exists(LOG_FILE):
        return {"logs": "No logs yet."}
    with open(LOG_FILE, "r") as f:
        return {"logs": f.read()[-5000:]}

# ── PUBLIC STREAM: "agent thinking" / live pipeline logs (SSE) ─────
@app.get("/api/pipeline/stream")
def stream_pipeline_logs():
    """
    Server-Sent Events stream of pipeline logs.
    This is intentionally read-only and contains no secrets; it powers the optional "Show agent thinking" UI.
    """
    def event_generator():
        # Ensure file exists so EventSource doesn't immediately fail.
        if not os.path.exists(LOG_FILE):
            yield "event: status\ndata: No pipeline logs yet. Run the pipeline.\n\n"
            return

        try:
            with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                # Start at end so new viewers don't download huge logs.
                f.seek(0, os.SEEK_END)
                while True:
                    line = f.readline()
                    if line:
                        # SSE requires each event end with a blank line.
                        msg = line.rstrip("\n").replace("\r", "")
                        yield f"data: {msg}\n\n"
                    else:
                        time.sleep(0.5)
        except Exception as e:
            yield f"event: error\ndata: Stream error: {str(e)[:120]}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Render/proxies sometimes buffer; this helps streaming.
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/api/admin/backtest")
def trigger_backtest(request: Request):
    verify_admin(request)
    def run():
        try:
            from backtester import run_backtest
            run_backtest()
        except Exception as e:
            print(f"Backtest error: {e}")
    threading.Thread(target=run, daemon=True).start()
    return {"status": "started", "message": "Backtest initiated."}

# ── REMOVED: /api/pipeline/force-run (security vulnerability) ─────
# ── REMOVED: /api/logs/marketpulse-secret-view (security vulnerability) ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
