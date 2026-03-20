from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine
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

# Database Connection
DB_URL = os.getenv("DATABASE_URL")
if DB_URL and DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)
engine = create_engine(DB_URL) if DB_URL else None

app = FastAPI(title="MarketPulse AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── RATE LIMITING ─────────────────────────────────────────
brief_usage:   dict[str, list[datetime]] = defaultdict(list)
chat_cooldown: dict[str, datetime]       = {}
BRIEF_MAX_PER_DAY = 2
CHAT_WORD_LIMIT   = 100
CHAT_COOLDOWN_HRS = 14

def get_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")

def check_brief_limit(ip: str) -> tuple[bool, int, int]:
    now = datetime.now()
    cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    brief_usage[ip] = [t for t in brief_usage[ip] if t >= cutoff]
    used = len(brief_usage[ip])
    remaining = max(0, BRIEF_MAX_PER_DAY - used)
    return remaining > 0, used, remaining

def check_chat_cooldown(ip: str) -> tuple[bool, int]:
    if ip not in chat_cooldown: return True, 0
    cooldown_until = chat_cooldown[ip]
    if datetime.now() >= cooldown_until:
        del chat_cooldown[ip]
        return True, 0
    mins = int((cooldown_until - datetime.now()).total_seconds() / 60)
    return False, mins

# ── HELPERS ───────────────────────────────────────────────
def safe(v):
    if v is None: return None
    if isinstance(v, (bool, np.bool_)): return bool(v)
    if isinstance(v, (int, np.integer)): return int(v)
    if isinstance(v, (float, np.floating)): return None if np.isnan(v) or np.isinf(v) else float(v)
    if isinstance(v, (np.ndarray, list)): return [safe(i) for i in v]
    if isinstance(v, dict): return {k: safe(val) for k, val in v.items()}
    return str(v)

def to_records(df: pd.DataFrame) -> list[dict]:
    if df.empty: return []
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return [{k: safe(v) for k, v in row.items()} for row in df.to_dict(orient="records")]

def load_data():
    if engine:
        try:
            headlines = pd.read_sql_table("latest_headlines", engine)
            sectors = pd.read_sql_table("latest_sectors", engine)
            return headlines, sectors
        except Exception:
            raise HTTPException(status_code=404, detail="Pipeline data not found in database. Run pipeline.py first.")
    raise HTTPException(status_code=500, detail="DATABASE_URL not configured.")

def classify_regime(avg_nss: float, avg_risk: float) -> dict:
    if avg_nss > 20 and avg_risk < 20:
        return {"regime": "Risk On", "description": "Broad bullish sentiment, low systemic risk. Momentum trades favored.", "nifty_implication": "Gap-up open likely. Momentum trades have higher probability.", "watch": "High-momentum sectors showing positive velocity.", "avoid": "Defensive over-positioning not needed in Risk On conditions."}
    elif avg_nss < -20 and avg_risk > 35:
        return {"regime": "Panic", "description": "Widespread negative sentiment with high systemic risk. Defensive only.", "nifty_implication": "Heavy selling pressure expected. Watch key support levels.", "watch": "Defensive sectors — Banking if NSS is stable.", "avoid": "All high-beta positions. Reduce exposure immediately."}
    elif avg_nss > 0 and avg_risk > 25:
        return {"regime": "Complacent", "description": "Positive headlines masking elevated underlying risk. Watch for reversal.", "nifty_implication": "Deceptively calm open possible. Reversal risk elevated.", "watch": "Divergence signals — sectors where NSS and impact-weighted disagree.", "avoid": "Overleveraged positions. Risk is higher than headlines suggest."}
    else:
        return {"regime": "Risk Off", "description": "Cautious market conditions. Capital preservation favored.", "nifty_implication": "Flat to gap-down open likely. Avoid chasing early moves.", "watch": "Sectors with positive velocity — early recovery signs.", "avoid": "High-leverage positions and sectors with negative velocity."}

# ── DASHBOARD ─────────────────────────────────────────────
@app.get("/api/dashboard")
def get_dashboard():
    headlines, benchmark = load_data()

    if "geopolitical_risk" in headlines.columns:
        headlines["geopolitical_risk"] = headlines["geopolitical_risk"].apply(lambda x: str(x).lower() in ["true", "1", "yes"])

    geo_hl = headlines[headlines["geopolitical_risk"] == True] if "geopolitical_risk" in headlines.columns else pd.DataFrame()
    avg_nss = float(benchmark["composite_sentiment_index"].mean()) if "composite_sentiment_index" in benchmark.columns and not benchmark.empty else 0.0
    avg_risk = float(benchmark["avg_weighted_risk"].mean()) if "avg_weighted_risk" in benchmark.columns and not benchmark.empty else 0.0
    regime = classify_regime(avg_nss, avg_risk)

    pareto_df = benchmark.sort_values("avg_weighted_risk", ascending=False).copy()
    total_risk = max(pareto_df["avg_weighted_risk"].sum(), 1)
    pareto_df["cumulative_pct"] = (pareto_df["avg_weighted_risk"].cumsum() / total_risk * 100).round(1)

    correlation = {"sectors": [], "values": []}
    velocity_trend = []
    shock_headlines = []
    shock_counts = {"major": 0, "shock": 0, "watch": 0}

    if engine:
        try:
            master = pd.read_sql_table("master_sector_scores", engine)
            if "composite_sentiment_index" in master.columns and "run_date" in master.columns:
                pivot = master.pivot_table(index="run_date", columns="sector", values="composite_sentiment_index").fillna(0)
                if len(pivot) >= 2:
                    corr = pivot.corr().round(2)
                    correlation = {"sectors": list(corr.columns), "values": [[safe(v) for v in row] for row in corr.values.tolist()]}
        except: pass

        try:
            trend = pd.read_sql_table("sector_trend_analysis", engine)
            if "csi_3day_ma" in trend.columns and "run_date" in trend.columns:
                pivot = trend.pivot_table(index="run_date", columns="sector", values="csi_3day_ma").fillna(0).reset_index().rename(columns={"run_date": "date"})
                pivot["run"] = range(1, len(pivot) + 1)
                velocity_trend = to_records(pivot)
        except: pass

        try:
            shock_df = pd.read_sql_table("shock_headlines", engine)
            shock_headlines = to_records(shock_df)
            shock_counts = {
                "major": len([h for h in shock_headlines if h.get("shock_status") == "Major Shock"]),
                "shock": len([h for h in shock_headlines if h.get("shock_status") == "Shock"]),
                "watch": len([h for h in shock_headlines if h.get("shock_status") == "Watch"]),
            }
        except: pass

    high_risk = benchmark[benchmark["risk_level"] == "HIGH"]["sector"].tolist() if "risk_level" in benchmark.columns else []

    return {
        "last_updated":       datetime.now().isoformat(),
        "market_regime":      regime,
        "benchmark":          to_records(benchmark),
        "headlines":          to_records(headlines.sort_values("impact_score", ascending=False) if "impact_score" in headlines.columns else headlines),
        "pareto":             to_records(pareto_df[["sector", "avg_weighted_risk", "cumulative_pct"]]),
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

@app.get("/api/sectors/{sector_name}")
def get_sector(sector_name: str):
    headlines, benchmark = load_data()
    sector_bm = benchmark[benchmark["sector"].str.lower() == sector_name.lower()]
    if sector_bm.empty:
        raise HTTPException(status_code=404, detail="Sector not found")
    sector_hl = headlines[headlines["sector"].str.lower() == sector_name.lower()] if "sector" in headlines.columns else pd.DataFrame()
    return {
        "sector":    sector_name,
        "metrics":   to_records(sector_bm)[0],
        "headlines": to_records(sector_hl.sort_values("impact_score", ascending=False) if "impact_score" in sector_hl.columns else sector_hl),
    }

# ── BRIEF ──────────────────────────────────────────
@app.get("/api/brief/status")
def brief_status(request: Request):
    ip = get_ip(request)
    allowed, used, remaining = check_brief_limit(ip)
    return {"allowed": allowed, "used": used, "remaining": remaining, "limit": BRIEF_MAX_PER_DAY}

class BriefRequest(BaseModel):
    top_headlines: list
    sector_summary: list
    regime: dict

@app.post("/api/brief")
def generate_brief(request: Request, req: BriefRequest):
    ip = get_ip(request)
    allowed, used, remaining = check_brief_limit(ip)
    if not allowed: raise HTTPException(status_code=429, detail={"error": "daily_limit_reached", "message": "Limit reached.", "used": used, "limit": BRIEF_MAX_PER_DAY, "remaining": 0})
    
    brief_usage[ip].append(datetime.now())
    _, used_now, remaining_now = check_brief_limit(ip)

    hl_lines = [f"• [{h.get('sector','')}] {h.get('title','')} (Impact: {h.get('impact_score','')}/10, {str(h.get('sentiment','')).upper()})" for h in req.top_headlines[:8]]
    sec_lines = [f"• {s.get('sector',''):12} | Risk {s.get('avg_weighted_risk', 0):5.1f} | CSI {s.get('composite_sentiment_index', 0):+6.1f} | {s.get('risk_level','')}" for s in sorted(req.sector_summary, key=lambda x: x.get("avg_weighted_risk", 0), reverse=True)]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a senior market strategist. Write a forward-looking outlook for Indian intraday traders. Use exact data provided."},
            {"role": "user", "content": f"Write outlook.\nRegime: {req.regime.get('regime', '')}\nSECTOR DATA:\n{''.join(sec_lines)}\nTOP HEADLINES:\n{''.join(hl_lines)}"}
        ],
        temperature=0.25, max_tokens=650,
    )
    return {"brief": response.choices[0].message.content, "used": used_now, "remaining": remaining_now, "limit": BRIEF_MAX_PER_DAY}

# ── CHAT ─────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list
    context_headlines: list
    context_sectors: list

@app.post("/api/chat")
def chat(request: Request, req: ChatRequest):
    ip = get_ip(request)
    allowed, mins_remaining = check_chat_cooldown(ip)
    if not allowed: raise HTTPException(status_code=429, detail={"error": "cooldown_active", "message": f"Wait {mins_remaining//60}h {mins_remaining%60}m.", "minutes_remaining": mins_remaining})

    word_count = len(req.message.split())
    if word_count > CHAT_WORD_LIMIT: raise HTTPException(status_code=400, detail={"error": "message_too_long"})

    total_words = word_count + sum(len(m.get("content", "").split()) for m in req.history if m.get("role") == "user")
    if total_words > CHAT_WORD_LIMIT:
        chat_cooldown[ip] = datetime.now() + timedelta(hours=CHAT_COOLDOWN_HRS)
        raise HTTPException(status_code=429, detail={"error": "session_limit_reached"})

    sector_context = "\n".join([f"• {s.get('sector','')} | CSI {s.get('composite_sentiment_index', 0):+6.1f} | Risk {s.get('avg_weighted_risk', 0):5.1f}" for s in req.context_sectors])
    headline_context = "\n".join([f"• [{h.get('sector','')}] {h.get('title','')} | Impact: {h.get('impact_score', '')}/10" for h in req.context_headlines[:20]])

    messages = [{"role": "system", "content": f"You are an Indian market assistant. Use ONLY this data:\nSECTORS:\n{sector_context}\nHEADLINES:\n{headline_context}"}]
    messages += [{"role": m["role"], "content": m["content"]} for m in req.history[-6:]]
    messages.append({"role": "user", "content": req.message})

    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2, max_tokens=500)
    answer = response.choices[0].message.content

    sources = list(dict.fromkeys([h.get("title", "") for h in req.context_headlines[:20] if h.get("sector", "").lower() in req.message.lower() or any(w in answer.lower() for w in h.get("title", "").lower().split()[:3])]))[:3]

    return {"answer": answer, "sources": sources, "words_used": total_words, "words_remaining": max(0, CHAT_WORD_LIMIT - total_words), "word_limit": CHAT_WORD_LIMIT}

# ── PIPELINE TRIGGER ─────────────────────────────────────
PIPELINE_SECRET = os.getenv("PIPELINE_SECRET", "marketpulse2024")

class PipelineRequest(BaseModel):
    secret: str
    max_per_feed: int = 14

@app.post("/api/pipeline/run")
def trigger_pipeline(req: PipelineRequest):
    if req.secret != PIPELINE_SECRET: raise HTTPException(status_code=401, detail="Invalid secret key.")
    max_per_feed = max(3, min(100, req.max_per_feed))
    total_approx = max_per_feed * 37

    import subprocess, sys, threading
    def run():
        subprocess.run([sys.executable, os.path.join(DATA_DIR, "pipeline.py"), "--once", f"--max-per-feed={max_per_feed}"], cwd=DATA_DIR)

    threading.Thread(target=run, daemon=True).start()
    return {"status": "started", "message": f"Pipeline started — fetching up to {total_approx} headlines. Check back in ~15 mins.", "started_at": datetime.now().isoformat(), "max_per_feed": max_per_feed, "approx_total": total_approx}

@app.get("/api/pipeline/status")
def pipeline_status():
    hl_time, hl_count, is_running = None, 0, False
    
    if engine:
        try:
            status_df = pd.read_sql_table("pipeline_status", engine)
            if not status_df.empty: hl_time = status_df.iloc[0]["last_run"]
            hl_count = len(pd.read_sql_table("latest_headlines", engine))
        except: pass

    lock_path = "/tmp/pipeline.lock"
    if os.path.exists(lock_path):
        age = datetime.now().timestamp() - os.path.getmtime(lock_path)
        is_running = age < 900 # Assume running if lock is < 15 mins old

    return {
        "last_headlines_update": hl_time,
        "headlines_count": hl_count,
        "is_running": is_running,
        "data_available": hl_count > 0,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
