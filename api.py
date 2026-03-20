from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os
import json
import subprocess
import sys
import threading
import traceback
from datetime import datetime, timedelta
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_DIR         = os.path.dirname(os.path.abspath(__file__))
LOG_FILE         = os.path.join(DATA_DIR, "pipeline_live.log")
client           = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PIPELINE_SECRET  = os.getenv("PIPELINE_SECRET", "marketpulse2024")

# SUPABASE DATABASE CONNECTION
DB_URL = os.getenv("DATABASE_URL")
if DB_URL and DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)
engine = create_engine(DB_URL) if DB_URL else None

app = FastAPI(title="MarketPulse AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

brief_usage:   dict[str, list[datetime]] = defaultdict(list)
chat_sessions: dict[str, dict]           = {}
BRIEF_MAX    = 2
CHAT_LIMIT   = 100

def get_ip(request: Request) -> str:
    fwd = request.headers.get("X-Forwarded-For")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")

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
    if not engine: raise HTTPException(status_code=500, detail="DATABASE_URL not configured.")
    try:
        headlines = pd.read_sql_table("latest_headlines", engine)
        sectors = pd.read_sql_table("latest_sectors", engine)
        
        for col in ["impact_score", "sentiment_confidence", "valence", "arousal"]:
            if col in headlines.columns: headlines[col] = pd.to_numeric(headlines[col], errors="coerce")
        for col in ["avg_weighted_risk", "composite_sentiment_index", "sentiment_nss", "sentiment_velocity", "avg_impact", "momentum_score"]:
            if col in sectors.columns: sectors[col] = pd.to_numeric(sectors[col], errors="coerce")
            
        return headlines, sectors
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Database error: {e}")

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

    velocity_trend, shock_headlines, msi = [], [], {}
    if engine:
        try:
            trend = pd.read_sql_table("sector_trend_analysis", engine)
            pivot = trend.pivot_table(index="run_date", columns="sector", values="csi_3day_ma").fillna(0).reset_index().rename(columns={"run_date": "date"})
            pivot["run"] = range(1, len(pivot) + 1)
            velocity_trend = to_records(pivot)
        except: pass
        try: shock_headlines = to_records(pd.read_sql_table("shock_headlines", engine))
        except: pass
        try:
            msi_df = pd.read_sql_table("latest_msi", engine)
            if not msi_df.empty: msi = json.loads(msi_df.iloc[0]["msi_data"])
        except: pass

    shock_counts = {"major": len([h for h in shock_headlines if h.get("shock_status") == "Major Shock"]), "shock": len([h for h in shock_headlines if h.get("shock_status") == "Shock"]), "watch": len([h for h in shock_headlines if h.get("shock_status") == "Watch"])}

    return {
        "last_updated": datetime.now().isoformat(),
        "market_regime": regime,
        "benchmark": to_records(benchmark),
        "headlines": to_records(headlines.sort_values("impact_score", ascending=False) if "impact_score" in headlines.columns else headlines),
        "pareto": to_records(pareto_df[["sector", "avg_weighted_risk", "cumulative_pct"]]),
        "velocity_trend": velocity_trend,
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
    if engine:
        try:
            status_df = pd.read_sql_table("pipeline_status", engine)
            if not status_df.empty: hl_time = status_df.iloc[0]["last_run"]
            hl_count = len(pd.read_sql_table("latest_headlines", engine))
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

class PipelineRequest(BaseModel):
    secret: str
    max_per_feed: int = 12

@app.post("/api/pipeline/run")
def trigger_pipeline(req: PipelineRequest):
    if req.secret != PIPELINE_SECRET: 
        raise HTTPException(status_code=401, detail="Invalid secret key.")
    
    max_per_feed = max(3, min(50, req.max_per_feed))
    total_approx = max_per_feed * 37

    def run():
        print(f"\n🚀 BACKGROUND THREAD: Launching pipeline.py (Max/Feed: {max_per_feed})...")
        try:
            result = subprocess.run(
                ["python", os.path.join(DATA_DIR, "pipeline.py"), f"--max-per-feed={max_per_feed}"],
                cwd=DATA_DIR,
                capture_output=True,
                text=True
            )
            print("✅ PIPELINE SCRIPT FINISHED.")
            if result.stdout:
                print("\n📝 SCRIPT OUTPUT:\n", result.stdout)
            if result.stderr:
                print("\n❌ SCRIPT ERRORS:\n", result.stderr)
        except Exception as e:
            print(f"\n🔥 FATAL THREAD ERROR: {e}")
            print(traceback.format_exc())

    threading.Thread(target=run, daemon=True).start()

    return {
        "status": "started", 
        "message": f"Diagnostic Pipeline started. Fetching ~{total_approx} headlines. Check Render logs.", 
        "started_at": datetime.now().isoformat(), 
        "max_per_feed": max_per_feed, 
        "approx_total": total_approx
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
