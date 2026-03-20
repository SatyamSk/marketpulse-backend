from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import json
import requests
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_DIR         = os.path.dirname(os.path.abspath(__file__))
LOG_FILE         = os.path.join(DATA_DIR, "pipeline_live.log")
client           = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PIPELINE_SECRET  = os.getenv("PIPELINE_SECRET", "marketpulse2024")

RAW_BASE_URL = "https://raw.githubusercontent.com/SatyamSk/MarketPulseAIData/main/"

app = FastAPI(title="MarketPulse AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def safe(v):
    if v is None or pd.isna(v): return None
    if isinstance(v, (bool, np.bool_)): return bool(v)
    if isinstance(v, (int, np.integer)): return int(v)
    if isinstance(v, (float, np.floating)): return float(v)
    return str(v)

def to_records(df: pd.DataFrame) -> list[dict]:
    if df.empty: return []
    return [{k: safe(v) for k, v in row.items()} for row in df.to_dict(orient="records")]

def load_data():
    try:
        headlines = pd.read_csv(f"{RAW_BASE_URL}latest_headlines.csv")
        sectors = pd.read_csv(f"{RAW_BASE_URL}latest_sectors.csv")
        
        for col in ["impact_score", "sentiment_confidence", "valence", "arousal"]:
            if col in headlines.columns: headlines[col] = pd.to_numeric(headlines[col], errors="coerce")
        for col in ["avg_weighted_risk", "composite_sentiment_index", "sentiment_nss", "sentiment_velocity", "avg_impact", "momentum_score"]:
            if col in sectors.columns: sectors[col] = pd.to_numeric(sectors[col], errors="coerce")
            
        return headlines, sectors
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"GitHub Data not found. Error: {e}")

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


@app.get("/api/dashboard")
def get_dashboard():
    headlines, benchmark = load_data()

    if "geopolitical_risk" in headlines.columns:
        headlines["geopolitical_risk"] = headlines["geopolitical_risk"].apply(
            lambda x: str(x).lower() in ["true", "1", "yes"]
        )

    geo_hl   = headlines[headlines["geopolitical_risk"] == True] if "geopolitical_risk" in headlines.columns else pd.DataFrame()
    avg_nss  = float(benchmark["composite_sentiment_index"].mean()) if not benchmark.empty else 0.0
    avg_risk = float(benchmark["avg_weighted_risk"].mean()) if not benchmark.empty else 0.0
    regime   = classify_regime(avg_nss, avg_risk)

    pareto_df  = benchmark.sort_values("avg_weighted_risk", ascending=False).copy()
    total_risk = max(pareto_df["avg_weighted_risk"].sum(), 1)
    pareto_df["cumulative_pct"] = (
        pareto_df["avg_weighted_risk"].cumsum() / total_risk * 100
    ).round(1)

    # Build contagion_flows from geo-flagged headlines
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

    shock_headlines, msi = [], {}
    try:
        shock_df = pd.read_csv(f"{RAW_BASE_URL}shock_headlines.csv")
        shock_headlines = to_records(shock_df)
    except Exception:
        pass
    try:
        msi_req = requests.get(f"{RAW_BASE_URL}latest_msi.json")
        if msi_req.status_code == 200:
            msi = msi_req.json()
    except Exception:
        pass

    shock_counts = {
        "major": len([h for h in shock_headlines if h.get("shock_status") == "Major Shock"]),
        "shock": len([h for h in shock_headlines if h.get("shock_status") == "Shock"]),
        "watch": len([h for h in shock_headlines if h.get("shock_status") == "Watch"]),
    }

    return {
        "last_updated":        datetime.now().isoformat(),
        "market_regime":       regime,
        "benchmark":           to_records(benchmark),
        "headlines":           to_records(
            headlines.sort_values("impact_score", ascending=False)
            if "impact_score" in headlines.columns else headlines
        ),
        "pareto":              to_records(pareto_df[["sector", "avg_weighted_risk", "cumulative_pct"]]),
        "contagion_flows":     contagion,
        "velocity_trend":      [],
        "shock_headlines":     shock_headlines,
        "shock_counts":        shock_counts,
        "market_stress_index": msi,
        "summary_stats": {
            "total_headlines":    len(headlines),
            "geopolitical_flags": len(geo_hl),
            "avg_nss":            round(avg_nss, 1),
            "avg_risk":           round(avg_risk, 1),
        },
    }

# ==========================================
# 🤖 THE COPILOT CHAT ENDPOINT
# ==========================================
class ChatRequest(BaseModel):
    message: str
    history: list
    context_headlines: list
    context_sectors: list

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    sector_ctx = "\n".join([
        f"• {s.get('sector','')} | Risk {s.get('avg_weighted_risk',0)} ({s.get('risk_level','')}) | CSI {s.get('composite_sentiment_index',0)}"
        for s in req.context_sectors
    ])
    hl_ctx = "\n".join([
        f"• [{h.get('sector','')}] {h.get('title','')} | Impact: {h.get('impact_score','')}/10 | {h.get('one_line_insight','')}"
        for h in req.context_headlines[:15]
    ])

    system_prompt = f"""You are MarketPulse AI — a sharp, friendly market intelligence assistant for Indian intraday traders.

BEHAVIOR RULES:
1. For greetings, small talk, or general questions (hi, hello, how are you, what can you do, thanks, etc.) — respond naturally and conversationally. Keep it brief and warm.
2. For ANY market-related question — answer strictly using the data provided below. Every number must come from this context. If a specific piece of market data is not available, say "I don't have that in today's data."
3. Always end market answers with a concrete, forward-looking trading implication.
4. Never make up stock prices, company names, or figures not in the data.

TODAY'S MARKET DATA ({datetime.now().strftime('%d %B %Y')}):

SECTOR SCORES:
{sector_ctx}

TOP HEADLINES:
{hl_ctx}"""

    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m["role"], "content": m["content"]} for m in req.history[-4:]]
    messages.append({"role": "user", "content": req.message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=400,
    )
    return {"answer": response.choices[0].message.content}

# ==========================================
# 🚀 DIAGNOSTICS & TRIGGER
# ==========================================
@app.get("/api/pipeline/force-run")
def force_run_pipeline():
    max_per_feed = 12
    def run():
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with open(LOG_FILE, "w") as log_f:
            log_f.write(f"🚀 OVERRIDING GITHUB REPOSITORY AT {datetime.now().strftime('%H:%M:%S')}\n{'='*50}\n")
            log_f.flush()
            try:
                process = subprocess.Popen(["python", os.path.join(DATA_DIR, "pipeline.py"), f"--max-per-feed={max_per_feed}"], cwd=DATA_DIR, stdout=log_f, stderr=subprocess.STDOUT, env=env, text=True)
                process.wait()
                with open(LOG_FILE, "a") as append_f: append_f.write(f"\n{'='*50}\n✅ UPLOAD COMPLETE.\n")
            except Exception as e:
                with open(LOG_FILE, "a") as append_f: append_f.write(f"\n🔥 FATAL ERROR: {e}\n{traceback.format_exc()}")
    threading.Thread(target=run, daemon=True).start()
    return HTMLResponse("<h1 style='text-align: center; margin-top: 50px;'>🚀 Writing to GitHub!</h1><p style='text-align: center;'><a href='/api/logs/marketpulse-secret-view'>Watch Live Logs</a></p>")

@app.get("/api/logs/marketpulse-secret-view")
def view_live_logs():
    if not os.path.exists(LOG_FILE): return HTMLResponse("<body style='background:#000; color:#0f0; padding:20px;'>No logs yet.</body>")
    with open(LOG_FILE, "r") as f: logs = f.read()
    return HTMLResponse(f"<html><head><style>body {{ background:#0d1117; color:#58a6ff; font-family:monospace; padding:20px; }} pre {{ white-space: pre-wrap; }}</style><script>setTimeout(() => location.reload(), 3000); window.onload = () => window.scrollTo(0, document.body.scrollHeight);</script></head><body><h2>🟢 Live GitHub Sync Logs</h2><pre>{logs}</pre></body></html>")
@app.get("/api/pipeline/status")
def pipeline_status():
    hl_time  = None
    hl_count = 0
    is_running = False

    # Check lock file
    lock_path = "/tmp/pipeline.lock"
    if os.path.exists(lock_path):
        age = datetime.now().timestamp() - os.path.getmtime(lock_path)
        is_running = age < 900

    # Read status from GitHub
    try:
        r = requests.get(f"{RAW_BASE_URL}pipeline_status.json", timeout=5)
        if r.status_code == 200:
            s = r.json()
            hl_time = s.get("last_run")
    except Exception:
        pass

    # Count headlines from GitHub
    try:
        df = pd.read_csv(f"{RAW_BASE_URL}latest_headlines.csv")
        hl_count = len(df)
    except Exception:
        pass

    return {
        "last_headlines_update": hl_time,
        "headlines_count":       hl_count,
        "is_running":            is_running,
        "data_available":        hl_count > 0,
    }


class PipelineRequest(BaseModel):
    secret:       str
    max_per_feed: int = 12


@app.post("/api/pipeline/run")
def trigger_pipeline(req: PipelineRequest):
    if req.secret != PIPELINE_SECRET:
        raise HTTPException(status_code=401, detail="Invalid secret key.")

    max_per_feed = max(3, min(50, req.max_per_feed))
    total_approx = max_per_feed * len([
        "ET Markets", "ET Economy", "ET Tech", "ET Startups", "ET Industry",
        "Livemint Markets", "Livemint Companies", "Livemint Economy",
        "BS Markets", "BS Economy", "MC Latest News",
        "Financial Express Markets", "PIB Economy", "RBI", "SEBI", "Reuters India"
    ])

    def run():
        lock_path = "/tmp/pipeline.lock"
        with open(lock_path, "w") as f:
            f.write(datetime.now().isoformat())
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            subprocess.run(
                [sys.executable, os.path.join(DATA_DIR, "pipeline.py"),
                 f"--max-per-feed={max_per_feed}"],
                cwd=DATA_DIR,
                env=env,
            )
        finally:
            if os.path.exists(lock_path):
                os.remove(lock_path)

    threading.Thread(target=run, daemon=True).start()

    return {
        "status":       "started",
        "message":      f"Pipeline started — fetching up to {total_approx} headlines from 16 sources. Headlines from last 48 hours only.",
        "started_at":   datetime.now().isoformat(),
        "max_per_feed": max_per_feed,
        "approx_total": total_approx,
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
    
