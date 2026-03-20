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
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_DIR         = os.path.dirname(os.path.abspath(__file__))
LOG_FILE         = os.path.join(DATA_DIR, "pipeline_live.log")
client           = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PIPELINE_SECRET  = os.getenv("PIPELINE_SECRET", "marketpulse2024")

# GITHUB RAW DATA URLs
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
        # Read directly from GitHub's raw CDN
        headlines = pd.read_csv(f"{RAW_BASE_URL}latest_headlines.csv")
        sectors = pd.read_csv(f"{RAW_BASE_URL}latest_sectors.csv")
        
        for col in ["impact_score", "sentiment_confidence", "valence", "arousal"]:
            if col in headlines.columns: headlines[col] = pd.to_numeric(headlines[col], errors="coerce")
        for col in ["avg_weighted_risk", "composite_sentiment_index", "sentiment_nss", "sentiment_velocity", "avg_impact", "momentum_score"]:
            if col in sectors.columns: sectors[col] = pd.to_numeric(sectors[col], errors="coerce")
            
        return headlines, sectors
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"GitHub Data not found. Run the pipeline first. Error: {e}")

@app.get("/api/dashboard")
def get_dashboard():
    headlines, benchmark = load_data()

    if "geopolitical_risk" in headlines.columns:
        headlines["geopolitical_risk"] = headlines["geopolitical_risk"].apply(lambda x: str(x).lower() in ["true", "1", "yes"])

    avg_nss  = float(benchmark["composite_sentiment_index"].mean()) if not benchmark.empty else 0.0
    avg_risk = float(benchmark["avg_weighted_risk"].mean()) if not benchmark.empty else 0.0
    regime = {"regime": "Risk On", "description": "Broad bullish sentiment.", "nifty_implication": "Gap-up"} if avg_nss > 20 and avg_risk < 20 else {"regime": "Panic", "description": "High risk.", "nifty_implication": "Selling pressure"}

    pareto_df = benchmark.sort_values("avg_weighted_risk", ascending=False).copy()
    total_risk = max(pareto_df["avg_weighted_risk"].sum(), 1)
    pareto_df["cumulative_pct"] = (pareto_df["avg_weighted_risk"].cumsum() / total_risk * 100).round(1)

    shock_headlines, msi = [], {}
    
    try:
        shock_df = pd.read_csv(f"{RAW_BASE_URL}shock_headlines.csv")
        shock_headlines = to_records(shock_df)
    except: pass
    
    try:
        msi_req = requests.get(f"{RAW_BASE_URL}latest_msi.json")
        if msi_req.status_code == 200:
            msi = msi_req.json()
    except: pass

    shock_counts = {"major": len([h for h in shock_headlines if h.get("shock_status") == "Major Shock"]), "shock": len([h for h in shock_headlines if h.get("shock_status") == "Shock"]), "watch": len([h for h in shock_headlines if h.get("shock_status") == "Watch"])}

    return {
        "last_updated": datetime.now().isoformat(),
        "market_regime": regime,
        "benchmark": to_records(benchmark),
        "headlines": to_records(headlines.sort_values("impact_score", ascending=False) if "impact_score" in headlines.columns else headlines),
        "pareto": to_records(pareto_df[["sector", "avg_weighted_risk", "cumulative_pct"]]),
        "velocity_trend": [], # Disabled for raw github approach unless you append locally
        "shock_headlines": shock_headlines,
        "shock_counts": shock_counts,
        "market_stress_index": msi,
        "summary_stats": {
            "total_headlines": len(headlines),
            "avg_nss": round(avg_nss, 1),
            "avg_risk": round(avg_risk, 1),
        },
    }

@app.get("/api/pipeline/status")
def pipeline_status():
    hl_time, hl_count = None, 0
    
    try:
        # Check GitHub for last run time
        status_req = requests.get(f"{RAW_BASE_URL}pipeline_status.json")
        if status_req.status_code == 200:
            hl_time = status_req.json().get("last_run")
            
        hl_df = pd.read_csv(f"{RAW_BASE_URL}latest_headlines.csv", usecols=[0])
        hl_count = len(hl_df)
    except: pass

    lock_path = "/tmp/pipeline.lock"
    is_running = False
    if os.path.exists(lock_path):
        is_running = (datetime.now().timestamp() - os.path.getmtime(lock_path)) < 600

    return {
        "last_headlines_update": hl_time,
        "headlines_count": hl_count,
        "is_running": is_running,
        "data_available": hl_count > 0,
    }


# ==========================================
# 🚀 OVERRIDE START & LIVE LOGS
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
                with open(LOG_FILE, "a") as append_f:
                    append_f.write(f"\n{'='*50}\n✅ UPLOAD COMPLETE.\n")
            except Exception as e:
                with open(LOG_FILE, "a") as append_f: append_f.write(f"\n🔥 FATAL ERROR: {e}\n{traceback.format_exc()}")

    threading.Thread(target=run, daemon=True).start()
    return HTMLResponse("<h1 style='text-align: center; margin-top: 50px;'>🚀 Writing to GitHub!</h1><p style='text-align: center;'><a href='/api/logs/marketpulse-secret-view'>Watch Live Logs</a></p>")

@app.get("/api/logs/marketpulse-secret-view")
def view_live_logs():
    if not os.path.exists(LOG_FILE): return HTMLResponse("<body style='background:#000; color:#0f0; padding:20px;'>No logs yet.</body>")
    with open(LOG_FILE, "r") as f: logs = f.read()
    return HTMLResponse(f"<html><head><style>body {{ background:#0d1117; color:#58a6ff; font-family:monospace; padding:20px; }} pre {{ white-space: pre-wrap; }}</style><script>setTimeout(() => location.reload(), 3000); window.onload = () => window.scrollTo(0, document.body.scrollHeight);</script></head><body><h2>🟢 Live GitHub Sync Logs</h2><pre>{logs}</pre></body></html>")

class PipelineRequest(BaseModel):
    secret: str
    max_per_feed: int = 12

@app.post("/api/pipeline/run")
def trigger_pipeline(req: PipelineRequest):
    if req.secret != PIPELINE_SECRET: raise HTTPException(status_code=401, detail="Invalid secret key.")
    return {"status": "Use the /api/pipeline/force-run link directly in your browser."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
