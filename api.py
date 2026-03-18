from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="MarketPulse AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPERS ---

def safe(v):
    """Convert any value to JSON-safe Python native type."""
    if pd.isna(v) or v is None:
        return None
    if isinstance(v, (bool, np.bool_)): return bool(v)
    if isinstance(v, (int, np.integer)): return int(v)
    if isinstance(v, (float, np.floating)): return float(v)
    if isinstance(v, (np.ndarray, list)): return [safe(i) for i in v]
    if isinstance(v, dict): return {k: safe(val) for k, val in v.items()}
    return str(v)

def to_records(df: pd.DataFrame) -> list[dict]:
    """Scrub DataFrame of NaNs and convert to clean dictionaries."""
    if df.empty: return []
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return [{k: safe(v) for k, v in row.items()} for row in df.to_dict(orient="records")]

def load_data():
    hl_path = os.path.join(DATA_DIR, "latest_headlines.csv")
    sec_path = os.path.join(DATA_DIR, "latest_sectors.csv")
    if not os.path.exists(hl_path) or not os.path.exists(sec_path):
        raise HTTPException(status_code=404, detail="Pipeline data not found. Run pipeline.py first.")
    return pd.read_csv(hl_path), pd.read_csv(sec_path)

# --- ENDPOINTS ---

@app.get("/api/status")
def status():
    path = os.path.join(DATA_DIR, "latest_sectors.csv")
    last_run = datetime.fromtimestamp(os.path.getmtime(path)).isoformat() if os.path.exists(path) else None
    return {"status": "ok", "last_run": last_run, "server_time": datetime.now().isoformat()}

@app.get("/api/dashboard")
def get_dashboard():
    headlines, benchmark = load_data()
    
    # Pareto
    pareto_df = benchmark.sort_values("avg_weighted_risk", ascending=False).copy()
    pareto_df["cumulative_pct"] = (pareto_df["avg_weighted_risk"].cumsum() / max(pareto_df["avg_weighted_risk"].sum(), 1) * 100).round(1)

    # Regime Logic
    avg_nss = float(benchmark["composite_sentiment_index"].mean()) if not benchmark.empty else 0.0
    avg_risk = float(benchmark["avg_weighted_risk"].mean()) if not benchmark.empty else 0.0
    
    if avg_nss > 20 and avg_risk < 25:
        regime = {"regime": "Risk On", "description": "Broad bullish sentiment, low systemic risk.", "nifty_implication": "Gap-up likely. Momentum trades favored.", "watch": "High-momentum sectors", "avoid": "Defensive positioning"}
    elif avg_nss < -20 and avg_risk > 45:
        regime = {"regime": "Panic", "description": "Widespread negative sentiment. Defensive only.", "nifty_implication": "Heavy selling expected. Watch support levels.", "watch": "Defensive sectors", "avoid": "High-beta positions"}
    elif avg_nss > 0 and avg_risk > 35:
        regime = {"regime": "Complacent", "description": "Positive headlines masking elevated underlying risk.", "nifty_implication": "Deceptively calm open possible. Reversal risk.", "watch": "Divergence signals", "avoid": "Overleveraged positions"}
    else:
        regime = {"regime": "Risk Off", "description": "Cautious market conditions. Capital preservation favored.", "nifty_implication": "Flat to gap-down open. Avoid chasing moves.", "watch": "Early recovery signs", "avoid": "High-leverage positions"}

    # Contagion Flows
    geo_hl = headlines[headlines["geopolitical_risk"].astype(str).str.lower().isin(["true", "1"])]
    contagion = [{"source": "Geopolitical Event", "target": r["sector"], "value": r["impact_score"]} 
                 for _, r in geo_hl.groupby("sector")["impact_score"].mean().reset_index().iterrows()]

    # Correlation Matrix
    correlation = {"sectors": [], "values": []}
    master_path = os.path.join(DATA_DIR, "master_sector_scores.csv")
    if os.path.exists(master_path):
        master = pd.read_csv(master_path)
        pivot = master.pivot_table(index="run_date", columns="sector", values="composite_sentiment_index").fillna(0)
        if len(pivot) >= 2:
            corr = pivot.corr().round(2)
            correlation = {"sectors": list(corr.columns), "values": corr.values.tolist()}

    # Velocity Trend
    velocity_trend = []
    trend_path = os.path.join(DATA_DIR, "sector_trend_analysis.csv")
    if os.path.exists(trend_path):
        trend = pd.read_csv(trend_path)
        pivot = trend.pivot_table(index="run_date", columns="sector", values="csi_3day_ma").fillna(0).reset_index()
        pivot = pivot.rename(columns={"run_date": "date"})
        pivot["run"] = range(1, len(pivot) + 1)
        velocity_trend = to_records(pivot)

    return {
        "last_updated": datetime.now().isoformat(),
        "market_regime": regime,
        "benchmark": to_records(benchmark),
        "headlines": to_records(headlines.sort_values("impact_score", ascending=False)),
        "pareto": to_records(pareto_df[["sector", "avg_weighted_risk", "cumulative_pct"]]),
        "contagion_flows": contagion,
        "correlation_matrix": correlation,
        "velocity_trend": velocity_trend,
        "summary_stats": {
            "total_headlines": len(headlines),
            "geopolitical_flags": len(geo_hl),
            "high_risk_sectors": benchmark[benchmark["risk_level"] == "HIGH"]["sector"].tolist() if not benchmark.empty else [],
            "avg_nss": round(avg_nss, 1),
            "avg_risk": round(avg_risk, 1),
        }
    }

@app.get("/api/sectors/{sector_name}")
def get_sector(sector_name: str):
    headlines, benchmark = load_data()
    sector_bm = benchmark[benchmark["sector"].str.lower() == sector_name.lower()]
    if sector_bm.empty:
        raise HTTPException(status_code=404, detail="Sector not found")
        
    sector_hl = headlines[headlines["sector"].str.lower() == sector_name.lower()]
    return {
        "sector": sector_name,
        "metrics": to_records(sector_bm)[0],
        "headlines": to_records(sector_hl.sort_values("impact_score", ascending=False))
    }

# --- AI ENDPOINTS ---

class BriefRequest(BaseModel):
    top_headlines: list
    sector_summary: list
    regime: dict

@app.post("/api/brief")
def generate_brief(req: BriefRequest):
    hl_text = "\n".join([f"- {h['title']} ({h.get('sector','')}, Impact: {h.get('impact_score','')})" for h in req.top_headlines[:8]])
    sec_text = "\n".join([f"- {s['sector']}: Risk {s.get('avg_weighted_risk','')}, CSI {s.get('composite_sentiment_index','')}" for s in req.sector_summary])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a sharp Bloomberg market analyst. Write a punchy pre-market brief. Use markdown bold for key terms. No fluff."},
            {"role": "user", "content": f"Regime: {req.regime.get('regime')}\n\nTop Sectors:\n{sec_text}\n\nKey Headlines:\n{hl_text}"}
        ],
        temperature=0.3
    )
    return {"brief": response.choices[0].message.content}

class ChatRequest(BaseModel):
    message: str
    history: list
    context_headlines: list
    context_sectors: list

@app.post("/api/chat")
def chat(req: ChatRequest):
    hl_context = "\n".join([f"- {h['title']} | Sector: {h.get('sector','')} | Impact: {h.get('impact_score','')}/10 | Insight: {h.get('one_line_insight','')}" for h in req.context_headlines[:20]])
    sec_context = "\n".join([f"- {s['sector']}: Risk {s.get('avg_weighted_risk','')}, CSI {s.get('composite_sentiment_index','')}" for s in req.context_sectors])
    
    system_prompt = f"You are an AI market assistant. Answer ONLY using this context. Cite numbers accurately.\n\nSECTORS:\n{sec_context}\n\nHEADLINES:\n{hl_context}"
    
    messages = [{"role": "system", "content": system_prompt}] + req.history[-5:] + [{"role": "user", "content": req.message}]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2
    )
    
    answer = response.choices[0].message.content
    sources = list(set([h.get("title", "") for h in req.context_headlines[:20] if h.get("sector", "").lower() in req.message.lower() or h.get("title", "").split()[0].lower() in answer.lower()]))[:3]
    
    return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)