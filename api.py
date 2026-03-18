from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="MarketPulse AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── RATE LIMITING ─────────────────────────────────────────
brief_usage:   dict[str, list[datetime]] = defaultdict(list)
chat_cooldown: dict[str, datetime]       = {}

BRIEF_MAX_PER_DAY = 2
CHAT_WORD_LIMIT   = 100
CHAT_COOLDOWN_HRS = 14

def get_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def check_brief_limit(ip: str) -> tuple[bool, int, int]:
    now    = datetime.now()
    cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    brief_usage[ip] = [t for t in brief_usage[ip] if t >= cutoff]
    used      = len(brief_usage[ip])
    remaining = max(0, BRIEF_MAX_PER_DAY - used)
    return remaining > 0, used, remaining

def check_chat_cooldown(ip: str) -> tuple[bool, int]:
    if ip not in chat_cooldown:
        return True, 0
    cooldown_until = chat_cooldown[ip]
    if datetime.now() >= cooldown_until:
        del chat_cooldown[ip]
        return True, 0
    mins = int((cooldown_until - datetime.now()).total_seconds() / 60)
    return False, mins

# ── HELPERS ───────────────────────────────────────────────

def safe(v):
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    if isinstance(v, (np.ndarray, list)):
        return [safe(i) for i in v]
    if isinstance(v, dict):
        return {k: safe(val) for k, val in v.items()}
    return str(v)

def to_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return [{k: safe(v) for k, v in row.items()} for row in df.to_dict(orient="records")]

def load_data():
    hl_path  = os.path.join(DATA_DIR, "latest_headlines.csv")
    sec_path = os.path.join(DATA_DIR, "latest_sectors.csv")
    if not os.path.exists(hl_path) or not os.path.exists(sec_path):
        raise HTTPException(
            status_code=404,
            detail="Pipeline data not found. Run pipeline.py first."
        )
    return pd.read_csv(hl_path), pd.read_csv(sec_path)

def classify_regime(avg_nss: float, avg_risk: float) -> dict:
    if avg_nss > 20 and avg_risk < 20:
        return {
            "regime":            "Risk On",
            "description":       "Broad bullish sentiment, low systemic risk. Momentum trades favored.",
            "nifty_implication": "Gap-up open likely. Momentum trades have higher probability.",
            "watch":             "High-momentum sectors showing positive velocity.",
            "avoid":             "Defensive over-positioning not needed in Risk On conditions.",
        }
    elif avg_nss < -20 and avg_risk > 35:
        return {
            "regime":            "Panic",
            "description":       "Widespread negative sentiment with high systemic risk. Defensive only.",
            "nifty_implication": "Heavy selling pressure expected. Watch key support levels.",
            "watch":             "Defensive sectors — Banking if NSS is stable.",
            "avoid":             "All high-beta positions. Reduce exposure immediately.",
        }
    elif avg_nss > 0 and avg_risk > 25:
        return {
            "regime":            "Complacent",
            "description":       "Positive headlines masking elevated underlying risk. Watch for reversal.",
            "nifty_implication": "Deceptively calm open possible. Reversal risk elevated.",
            "watch":             "Divergence signals — sectors where NSS and impact-weighted disagree.",
            "avoid":             "Overleveraged positions. Risk is higher than headlines suggest.",
        }
    else:
        return {
            "regime":            "Risk Off",
            "description":       "Cautious market conditions. Capital preservation favored.",
            "nifty_implication": "Flat to gap-down open likely. Avoid chasing early moves.",
            "watch":             "Sectors with positive velocity — early recovery signs.",
            "avoid":             "High-leverage positions and sectors with negative velocity.",
        }
# ── STATUS ────────────────────────────────────────────────

@app.get("/api/status")
def status():
    path     = os.path.join(DATA_DIR, "latest_sectors.csv")
    last_run = None
    if os.path.exists(path):
        last_run = datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
    return {
        "status":      "ok",
        "last_run":    last_run,
        "server_time": datetime.now().isoformat(),
    }

# ── DASHBOARD ─────────────────────────────────────────────

@app.get("/api/dashboard")
def get_dashboard():
    headlines, benchmark = load_data()

    # Clean geopolitical_risk column
    if "geopolitical_risk" in headlines.columns:
        headlines["geopolitical_risk"] = headlines["geopolitical_risk"].apply(
            lambda x: str(x).lower() in ["true", "1", "yes"]
        )

    geo_hl   = headlines[headlines["geopolitical_risk"] == True] if "geopolitical_risk" in headlines.columns else pd.DataFrame()
    avg_nss  = float(benchmark["composite_sentiment_index"].mean()) if "composite_sentiment_index" in benchmark.columns and not benchmark.empty else 0.0
    avg_risk = float(benchmark["avg_weighted_risk"].mean()) if "avg_weighted_risk" in benchmark.columns and not benchmark.empty else 0.0
    regime   = classify_regime(avg_nss, avg_risk)

    # Pareto
    pareto_df  = benchmark.sort_values("avg_weighted_risk", ascending=False).copy()
    total_risk = max(pareto_df["avg_weighted_risk"].sum(), 1)
    pareto_df["cumulative_pct"] = (
        pareto_df["avg_weighted_risk"].cumsum() / total_risk * 100
    ).round(1)

    # Contagion flows
    contagion = []
    if not geo_hl.empty and "sector" in geo_hl.columns and "impact_score" in geo_hl.columns:
        contagion = [
            {
                "source": "Geopolitical Event",
                "target": r["sector"],
                "value":  round(float(r["impact_score"]), 1),
            }
            for _, r in geo_hl.groupby("sector")["impact_score"].mean().reset_index().iterrows()
        ]

    # Correlation Matrix
    correlation = {"sectors": [], "values": []}
    master_path = os.path.join(DATA_DIR, "master_sector_scores.csv")
    if os.path.exists(master_path):
        master = pd.read_csv(master_path)
        csi_col = "composite_sentiment_index"
        if csi_col in master.columns and "run_date" in master.columns:
            pivot = master.pivot_table(
                index="run_date", columns="sector", values=csi_col
            ).fillna(0)
            if len(pivot) >= 2:
                corr = pivot.corr().round(2)
                correlation = {
                    "sectors": list(corr.columns),
                    "values":  [[safe(v) for v in row] for row in corr.values.tolist()],
                }

    # Velocity Trend
    velocity_trend = []
    trend_path = os.path.join(DATA_DIR, "sector_trend_analysis.csv")
    if os.path.exists(trend_path):
        trend = pd.read_csv(trend_path)
        if "csi_3day_ma" in trend.columns and "run_date" in trend.columns:
            pivot = trend.pivot_table(
                index="run_date", columns="sector", values="csi_3day_ma"
            ).fillna(0).reset_index().rename(columns={"run_date": "date"})
            pivot["run"] = range(1, len(pivot) + 1)
            velocity_trend = to_records(pivot)

    # Shock headlines
    shock_path       = os.path.join(DATA_DIR, "shock_headlines.csv")
    shock_headlines  = to_records(pd.read_csv(shock_path)) if os.path.exists(shock_path) else []
    shock_counts     = {
        "major": len([h for h in shock_headlines if h.get("shock_status") == "Major Shock"]),
        "shock": len([h for h in shock_headlines if h.get("shock_status") == "Shock"]),
        "watch": len([h for h in shock_headlines if h.get("shock_status") == "Watch"]),
    }

    high_risk = benchmark[benchmark["risk_level"] == "HIGH"]["sector"].tolist() if "risk_level" in benchmark.columns else []

    return {
        "last_updated":       datetime.now().isoformat(),
        "market_regime":      regime,
        "benchmark":          to_records(benchmark),
        "headlines":          to_records(
            headlines.sort_values("impact_score", ascending=False)
            if "impact_score" in headlines.columns else headlines
        ),
        "pareto":             to_records(pareto_df[["sector", "avg_weighted_risk", "cumulative_pct"]]),
        "contagion_flows":    contagion,
        "correlation_matrix": correlation,
        "velocity_trend":     velocity_trend,
        "shock_headlines":    shock_headlines,
        "shock_counts":       shock_counts,
        "summary_stats": {
            "total_headlines":    len(headlines),
            "geopolitical_flags": len(geo_hl),
            "high_risk_sectors":  high_risk,
            "avg_nss":            round(avg_nss, 1),
            "avg_risk":           round(avg_risk, 1),
        },
    }

# ── SECTOR DETAIL ─────────────────────────────────────────

@app.get("/api/sectors/{sector_name}")
def get_sector(sector_name: str):
    headlines, benchmark = load_data()
    sector_bm = benchmark[benchmark["sector"].str.lower() == sector_name.lower()]
    if sector_bm.empty:
        raise HTTPException(status_code=404, detail="Sector not found")
    sector_hl = headlines[
        headlines["sector"].str.lower() == sector_name.lower()
    ] if "sector" in headlines.columns else pd.DataFrame()
    return {
        "sector":    sector_name,
        "metrics":   to_records(sector_bm)[0],
        "headlines": to_records(
            sector_hl.sort_values("impact_score", ascending=False)
            if "impact_score" in sector_hl.columns else sector_hl
        ),
    }

# ── BRIEF STATUS ──────────────────────────────────────────

@app.get("/api/brief/status")
def brief_status(request: Request):
    ip = get_ip(request)
    allowed, used, remaining = check_brief_limit(ip)
    return {
        "allowed":   allowed,
        "used":      used,
        "remaining": remaining,
        "limit":     BRIEF_MAX_PER_DAY,
    }

# ── BRIEF (2 per IP per day) ──────────────────────────────

class BriefRequest(BaseModel):
    top_headlines: list
    sector_summary: list
    regime: dict

@app.post("/api/brief")
def generate_brief(request: Request, req: BriefRequest):
    ip = get_ip(request)
    allowed, used, remaining = check_brief_limit(ip)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error":     "daily_limit_reached",
                "message":   "You have used both daily brief generations. Resets at midnight.",
                "used":      used,
                "limit":     BRIEF_MAX_PER_DAY,
                "remaining": 0,
            }
        )

    brief_usage[ip].append(datetime.now())
    _, used_now, remaining_now = check_brief_limit(ip)

    # Build context from Python-calculated data
    hl_lines = []
    for h in req.top_headlines[:8]:
        shock = h.get("shock_status", "Normal")
        shock_tag = " 🚨 MAJOR SHOCK" if shock == "Major Shock" else " ⚠ SHOCK" if shock == "Shock" else ""
        hl_lines.append(
            f"• [{h.get('sector','')}] {h.get('title','')} "
            f"(Impact: {h.get('impact_score','')}/10, "
            f"{str(h.get('sentiment','')).upper()}{shock_tag})"
        )

    sec_lines = []
    for s in sorted(req.sector_summary, key=lambda x: x.get("avg_weighted_risk", 0), reverse=True):
        sec_lines.append(
            f"• {s.get('sector',''):12} | Risk {s.get('avg_weighted_risk', 0):5.1f} "
            f"| NSS {s.get('sentiment_nss', 0):+6.1f} "
            f"| CSI {s.get('composite_sentiment_index', 0):+6.1f} "
            f"| Velocity {s.get('sentiment_velocity', 0):+5.1f} "
            f"| {s.get('risk_level',''):6} | {s.get('sector_classification','')}"
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a senior market strategist writing a forward-looking market outlook for Indian intraday traders.
Frame everything as EXPECTED conditions — what is likely to happen, not what already happened.
Use language like 'expected to', 'likely to', 'anticipated', 'watch for', 'probability of'.
Use markdown bold for key numbers and sector names. No disclaimers. No fluff.
Every sentence must contain a specific number, sector name, or price level."""
            },
            {
                "role": "user",
                "content": f"""Write a forward-looking market outlook using EXACTLY this structure:

## Expected Regime: {req.regime.get('regime', '')}
(What this regime means for today's expected price action — one sentence)

## Nifty Expected Move
(Specific expected direction with levels — e.g. "Expected gap-down open near 22,100. Watch 21,950 support.")

## Highest Probability Risk Today
(The single event most likely to move markets — quantify the expected impact and which sectors get hit)

## Sector Outlook
**Expected Underperformer:** (sector name — why, with specific scores)
**Expected Outperformer:** (sector name — why, with specific scores)

## Key Events to Watch This Session
(3 bullet points — what to monitor, not what already happened)

## Trading Implication
(One specific forward-looking call: what to expect and how to position for it)

---
Regime: {req.regime.get('regime', '')} — {req.regime.get('description', '')}
Nifty Expected: {req.regime.get('nifty_implication', '')}

SECTOR DATA (Python-calculated — use these exact numbers):
{''.join(sec_lines)}

TOP HEADLINES (AI-classified, Python-scored):
{''.join(hl_lines)}"""
            }
        ],
        temperature=0.25,
        max_tokens=650,
    )

    return {
        "brief":     response.choices[0].message.content,
        "used":      used_now,
        "remaining": remaining_now,
        "limit":     BRIEF_MAX_PER_DAY,
    }

# ── CHAT (100 word limit + 14h cooldown) ─────────────────

class ChatRequest(BaseModel):
    message:           str
    history:           list
    context_headlines: list
    context_sectors:   list

@app.post("/api/chat")
def chat(request: Request, req: ChatRequest):
    ip = get_ip(request)

    # Check cooldown first
    allowed, mins_remaining = check_chat_cooldown(ip)
    if not allowed:
        hours = mins_remaining // 60
        mins  = mins_remaining % 60
        raise HTTPException(
            status_code=429,
            detail={
                "error":             "cooldown_active",
                "message":           f"Word limit reached. Chat available again in {hours}h {mins}m.",
                "minutes_remaining": mins_remaining,
            }
        )

    # Count words in this message
    word_count = len(req.message.split())
    if word_count > CHAT_WORD_LIMIT:
        raise HTTPException(
            status_code=400,
            detail={
                "error":      "message_too_long",
                "message":    f"Message is {word_count} words. Maximum is {CHAT_WORD_LIMIT} words.",
                "word_count": word_count,
                "limit":      CHAT_WORD_LIMIT,
            }
        )

    # Count total session words
    total_words = word_count + sum(
        len(m.get("content", "").split())
        for m in req.history
        if m.get("role") == "user"
    )

    if total_words > CHAT_WORD_LIMIT:
        chat_cooldown[ip] = datetime.now() + timedelta(hours=CHAT_COOLDOWN_HRS)
        raise HTTPException(
            status_code=429,
            detail={
                "error":          "session_limit_reached",
                "message":        f"You've used {total_words} words this session. 14-hour cooldown activated.",
                "total_words":    total_words,
                "limit":          CHAT_WORD_LIMIT,
                "cooldown_until": chat_cooldown[ip].isoformat(),
            }
        )

    # Build RAG context from Python-calculated data
    sector_context = "\n".join([
        f"• {s.get('sector',''):12} | NSS {s.get('sentiment_nss', 0):+6.1f} "
        f"| CSI {s.get('composite_sentiment_index', 0):+6.1f} "
        f"| Risk {s.get('avg_weighted_risk', 0):5.1f} ({s.get('risk_level','')})"
        f"| Velocity {s.get('sentiment_velocity', 0):+5.1f}"
        f"| {s.get('divergence_flag', 'Normal')}"
        for s in req.context_sectors
    ])

    headline_context = "\n".join([
        f"• [{h.get('sector','')}] {h.get('title','')} "
        f"| {str(h.get('sentiment','')).upper()} "
        f"(conf: {h.get('sentiment_confidence', 0.7):.2f}) "
        f"| Impact: {h.get('impact_score', '')}/10 "
        f"| Shock: {h.get('shock_status', 'Normal')} "
        f"| Geo: {h.get('geopolitical_risk', False)} "
        f"| {h.get('one_line_insight', '')}"
        for h in req.context_headlines[:20]
    ])

    system_prompt = f"""You are a sharp market intelligence assistant for Indian intraday traders.
CRITICAL: Answer ONLY using the data provided below. Every number you cite must come from this context.
If something is not in the data, say "I don't have that data today."
Use markdown bold for key numbers and sector names.
Always end with a concrete forward-looking trading implication.

TODAY'S SECTOR DATA (Python-calculated scores — use these exact numbers):
{sector_context}

TODAY'S HEADLINES (AI-classified, Python-scored):
{headline_context}

Date: {datetime.now().strftime('%d %B %Y')}"""

    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m["role"], "content": m["content"]} for m in req.history[-6:]]
    messages.append({"role": "user", "content": req.message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=500,
    )

    answer = response.choices[0].message.content

    sources = list(dict.fromkeys([
        h.get("title", "")
        for h in req.context_headlines[:20]
        if h.get("sector", "").lower() in req.message.lower()
        or any(w in answer.lower() for w in h.get("title", "").lower().split()[:3])
    ]))[:3]

    words_used_now  = total_words
    words_remaining = max(0, CHAT_WORD_LIMIT - words_used_now)

    return {
        "answer":          answer,
        "sources":         sources,
        "words_used":      words_used_now,
        "words_remaining": words_remaining,
        "word_limit":      CHAT_WORD_LIMIT,
    }
# ── PIPELINE TRIGGER ─────────────────────────────────────
# Simple secret key to prevent random people from triggering it

PIPELINE_SECRET = os.getenv("PIPELINE_SECRET", "marketpulse2024")

class PipelineRequest(BaseModel):
    secret:      str
    max_per_feed: int = 8   # default 8 per feed = ~40 total

@app.post("/api/pipeline/run")
def trigger_pipeline(req: PipelineRequest):
    if req.secret != PIPELINE_SECRET:
        raise HTTPException(status_code=401, detail="Invalid secret key.")

    # Clamp between 3 and 20 per feed
    max_per_feed = max(3, min(20, req.max_per_feed))
    total_approx = max_per_feed * 5   # 5 feeds

    import subprocess, sys, threading

    def run():
        subprocess.run(
            [
                sys.executable,
                os.path.join(DATA_DIR, "pipeline.py"),
                "--once",
                f"--max-per-feed={max_per_feed}",
            ],
            cwd=DATA_DIR,
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return {
        "status":      "started",
        "message":     f"Pipeline started — fetching up to {total_approx} headlines ({max_per_feed} per source). Refresh data in ~2 minutes.",
        "started_at":  datetime.now().isoformat(),
        "max_per_feed": max_per_feed,
        "approx_total": total_approx,
    }

@app.get("/api/pipeline/status")
def pipeline_status():
    """Check when pipeline last ran and what data exists."""
    hl_path  = os.path.join(DATA_DIR, "latest_headlines.csv")
    sec_path = os.path.join(DATA_DIR, "latest_sectors.csv")

    hl_time  = None
    sec_time = None
    hl_count = 0

    if os.path.exists(hl_path):
        hl_time  = datetime.fromtimestamp(os.path.getmtime(hl_path)).isoformat()
        hl_count = len(pd.read_csv(hl_path))

    if os.path.exists(sec_path):
        sec_time = datetime.fromtimestamp(os.path.getmtime(sec_path)).isoformat()

    # Check if pipeline is currently running
    is_running = False
    lock_path  = os.path.join(DATA_DIR, "pipeline.lock")
    if os.path.exists(lock_path):
        # Lock older than 10 minutes = stale, ignore it
        age = datetime.now().timestamp() - os.path.getmtime(lock_path)
        is_running = age < 600

    return {
        "last_headlines_update": hl_time,
        "last_sectors_update":   sec_time,
        "headlines_count":       hl_count,
        "is_running":            is_running,
        "data_available":        os.path.exists(hl_path) and os.path.exists(sec_path),
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
