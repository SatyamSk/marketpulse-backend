from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import json
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_DIR         = os.path.dirname(os.path.abspath(__file__))
client           = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PIPELINE_SECRET  = os.getenv("PIPELINE_SECRET", "marketpulse2024")

app = FastAPI(title="MarketPulse AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

brief_usage:   dict[str, list[datetime]] = defaultdict(list)
chat_sessions: dict[str, dict]           = {}

BRIEF_MAX    = 2
CHAT_LIMIT   = 100
COOLDOWN_HRS = 14


def get_ip(request: Request) -> str:
    fwd = request.headers.get("X-Forwarded-For")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def safe(v):
    if v is None: return None
    if isinstance(v, (bool, np.bool_)):    return bool(v)
    if isinstance(v, (int, np.integer)):   return int(v)
    if isinstance(v, (float, np.floating)):
        if np.isnan(v) or np.isinf(v): return None
        return float(v)
    if isinstance(v, (np.ndarray, list)): return [safe(i) for i in v]
    if isinstance(v, dict):               return {k: safe(val) for k, val in v.items()}
    return str(v)


def to_records(df: pd.DataFrame) -> list[dict]:
    if df.empty: return []
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return [{k: safe(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def load_data():
    hl_path  = os.path.join(DATA_DIR, "latest_headlines.csv")
    sec_path = os.path.join(DATA_DIR, "latest_sectors.csv")
    if not os.path.exists(hl_path) or not os.path.exists(sec_path):
        raise HTTPException(status_code=404, detail="Pipeline data not found. Run pipeline first.")
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


@app.get("/api/status")
def status():
    path     = os.path.join(DATA_DIR, "latest_sectors.csv")
    last_run = None
    if os.path.exists(path):
        last_run = datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
    return {"status": "ok", "last_run": last_run, "server_time": datetime.now().isoformat()}


@app.get("/api/dashboard")
def get_dashboard():
    headlines, benchmark = load_data()

    if "geopolitical_risk" in headlines.columns:
        headlines["geopolitical_risk"] = headlines["geopolitical_risk"].apply(
            lambda x: str(x).lower() in ["true", "1", "yes"]
        )

    geo_hl   = headlines[headlines["geopolitical_risk"] == True] if "geopolitical_risk" in headlines.columns else pd.DataFrame()
    avg_nss  = float(benchmark["composite_sentiment_index"].mean()) if "composite_sentiment_index" in benchmark.columns and not benchmark.empty else 0.0
    avg_risk = float(benchmark["avg_weighted_risk"].mean()) if "avg_weighted_risk" in benchmark.columns and not benchmark.empty else 0.0
    regime   = classify_regime(avg_nss, avg_risk)

    pareto_df  = benchmark.sort_values("avg_weighted_risk", ascending=False).copy()
    total_risk = max(pareto_df["avg_weighted_risk"].sum(), 1)
    pareto_df["cumulative_pct"] = (
        pareto_df["avg_weighted_risk"].cumsum() / total_risk * 100
    ).round(1)

    contagion = []
    if not geo_hl.empty and "sector" in geo_hl.columns and "impact_score" in geo_hl.columns:
        contagion = [
            {"source": "Geopolitical Event", "target": r["sector"], "value": round(float(r["impact_score"]), 1)}
            for _, r in geo_hl.groupby("sector")["impact_score"].mean().reset_index().iterrows()
        ]

    correlation = {"sectors": [], "values": []}
    master_path = os.path.join(DATA_DIR, "master_sector_scores.csv")
    if os.path.exists(master_path):
        master = pd.read_csv(master_path)
        csi_col = "composite_sentiment_index"
        if csi_col in master.columns and "run_date" in master.columns:
            pivot = master.pivot_table(index="run_date", columns="sector", values=csi_col).fillna(0)
            if len(pivot) >= 2:
                corr = pivot.corr().round(2)
                correlation = {
                    "sectors": list(corr.columns),
                    "values":  [[safe(v) for v in row] for row in corr.values.tolist()],
                }

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

    shock_path      = os.path.join(DATA_DIR, "shock_headlines.csv")
    shock_headlines = to_records(pd.read_csv(shock_path)) if os.path.exists(shock_path) else []
    shock_counts    = {
        "major": len([h for h in shock_headlines if h.get("shock_status") == "Major Shock"]),
        "shock": len([h for h in shock_headlines if h.get("shock_status") == "Shock"]),
        "watch": len([h for h in shock_headlines if h.get("shock_status") == "Watch"]),
    }

    msi = {}
    msi_path = os.path.join(DATA_DIR, "latest_msi.json")
    if os.path.exists(msi_path):
        with open(msi_path) as f:
            msi = json.load(f)

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
        "market_stress_index": msi,
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


@app.get("/api/brief/status")
def brief_status(request: Request):
    ip = get_ip(request)
    now    = datetime.now()
    cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    brief_usage[ip] = [t for t in brief_usage[ip] if t >= cutoff]
    used      = len(brief_usage[ip])
    remaining = max(0, BRIEF_MAX - used)
    return {"allowed": remaining > 0, "used": used, "remaining": remaining, "limit": BRIEF_MAX}


class BriefRequest(BaseModel):
    top_headlines:  list
    sector_summary: list
    regime:         dict


@app.post("/api/brief")
def generate_brief(request: Request, req: BriefRequest):
    ip = get_ip(request)
    now    = datetime.now()
    cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    brief_usage[ip] = [t for t in brief_usage[ip] if t >= cutoff]
    used      = len(brief_usage[ip])
    remaining = max(0, BRIEF_MAX - used)

    if remaining <= 0:
        raise HTTPException(status_code=429, detail={
            "error": "daily_limit_reached",
            "message": "Both daily brief generations used. Resets at midnight.",
            "used": used, "limit": BRIEF_MAX, "remaining": 0,
        })

    brief_usage[ip].append(datetime.now())
    _, used_now, remaining_now = len(brief_usage[ip]), len(brief_usage[ip]), max(0, BRIEF_MAX - len(brief_usage[ip]))

    hl_lines = []
    for h in req.top_headlines[:8]:
        shock = h.get("shock_status", "Normal")
        tag   = " 🚨 MAJOR SHOCK" if shock == "Major Shock" else " ⚠ SHOCK" if shock == "Shock" else ""
        hl_lines.append(
            f"• [{h.get('sector','')}] {h.get('title','')} "
            f"(Impact: {h.get('impact_score','')}/10, {str(h.get('sentiment','')).upper()}{tag})"
        )

    sec_lines = []
    for s in sorted(req.sector_summary, key=lambda x: x.get("avg_weighted_risk", 0), reverse=True):
        sec_lines.append(
            f"• {s.get('sector',''):12} | Risk {s.get('avg_weighted_risk',0):5.1f} "
            f"| CSI {s.get('composite_sentiment_index',0):+6.1f} "
            f"| Velocity {s.get('sentiment_velocity',0):+5.1f} "
            f"| {s.get('risk_level',''):6} | {s.get('sector_classification','')}"
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior market strategist writing a forward-looking market outlook "
                    "for Indian intraday traders. Frame everything as EXPECTED conditions. "
                    "Use language like 'expected to', 'likely to', 'anticipated', 'watch for'. "
                    "Use markdown bold for key numbers and sector names. No disclaimers. No fluff. "
                    "Every sentence must contain a specific number, sector name, or price level."
                ),
            },
            {
                "role": "user",
                "content": f"""Write a forward-looking market outlook with EXACTLY this structure:

## Expected Regime: {req.regime.get('regime', '')}
(What this means for today's expected price action)

## Nifty Expected Move
(Specific direction with levels)

## Highest Probability Risk Today
(Single event most likely to move markets)

## Sector Outlook
**Expected Underperformer:** (sector — why, with scores)
**Expected Outperformer:** (sector — why, with scores)

## Key Events to Watch This Session
(3 bullet points)

## Trading Implication
(One specific forward-looking call)

---
Regime: {req.regime.get('regime', '')} — {req.regime.get('description', '')}
Nifty: {req.regime.get('nifty_implication', '')}

SECTOR DATA:
{''.join(sec_lines)}

TOP HEADLINES:
{''.join(hl_lines)}""",
            }
        ],
        temperature=0.25,
        max_tokens=650,
    )

    return {
        "brief":     response.choices[0].message.content,
        "used":      used_now,
        "remaining": remaining_now,
        "limit":     BRIEF_MAX,
    }


class ChatRequest(BaseModel):
    message:           str
    history:           list
    context_headlines: list
    context_sectors:   list


@app.post("/api/chat")
def chat(request: Request, req: ChatRequest):
    ip = get_ip(request)

    if ip in chat_sessions:
        session = chat_sessions[ip]
        if datetime.now() < session.get("cooldown_until", datetime.now()):
            mins = int((session["cooldown_until"] - datetime.now()).total_seconds() / 60)
            raise HTTPException(status_code=429, detail={
                "error": "cooldown_active",
                "message": f"Word limit reached. Available again in {mins // 60}h {mins % 60}m.",
                "minutes_remaining": mins,
            })

    word_count = len(req.message.split())
    if word_count > CHAT_LIMIT:
        raise HTTPException(status_code=400, detail={
            "error": "message_too_long",
            "message": f"Message is {word_count} words. Max is {CHAT_LIMIT}.",
            "word_count": word_count, "limit": CHAT_LIMIT,
        })

    total_words = word_count + sum(
        len(m.get("content", "").split()) for m in req.history if m.get("role") == "user"
    )

    if total_words > CHAT_LIMIT:
        cooldown_until = datetime.now() + timedelta(hours=COOLDOWN_HRS)
        chat_sessions[ip] = {"cooldown_until": cooldown_until}
        raise HTTPException(status_code=429, detail={
            "error": "session_limit_reached",
            "message": f"Used {total_words} words. 14-hour cooldown activated.",
            "total_words": total_words, "limit": CHAT_LIMIT,
            "cooldown_until": cooldown_until.isoformat(),
        })

    sector_ctx = "\n".join([
        f"• {s.get('sector',''):12} | NSS {s.get('sentiment_nss',0):+6.1f} "
        f"| CSI {s.get('composite_sentiment_index',0):+6.1f} "
        f"| Risk {s.get('avg_weighted_risk',0):5.1f} ({s.get('risk_level','')})"
        for s in req.context_sectors
    ])

    hl_ctx = "\n".join([
        f"• [{h.get('sector','')}] {h.get('title','')} "
        f"| {str(h.get('sentiment','')).upper()} "
        f"| Impact: {h.get('impact_score','')}/10 "
        f"| Shock: {h.get('shock_status','Normal')} "
        f"| {h.get('one_line_insight','')}"
        for h in req.context_headlines[:20]
    ])

    system_prompt = f"""You are a sharp market intelligence assistant for Indian intraday traders.
Answer ONLY using the data below. Every number must come from this context.
If something is not in the data, say "I don't have that data today."
Always end with a concrete forward-looking trading implication.

SECTOR DATA:
{sector_ctx}

HEADLINES:
{hl_ctx}

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

    return {
        "answer":          response.choices[0].message.content,
        "sources":         [],
        "words_used":      total_words,
        "words_remaining": max(0, CHAT_LIMIT - total_words),
        "word_limit":      CHAT_LIMIT,
    }


@app.get("/api/pipeline/status")
def pipeline_status():
    hl_path  = os.path.join(DATA_DIR, "latest_headlines.csv")
    sec_path = os.path.join(DATA_DIR, "latest_sectors.csv")
    hl_time  = None
    hl_count = 0

    if os.path.exists(hl_path):
        hl_time  = datetime.fromtimestamp(os.path.getmtime(hl_path)).isoformat()
        try:
            hl_count = len(pd.read_csv(hl_path))
        except Exception:
            hl_count = 0

    is_running = False
    lock_path  = os.path.join(DATA_DIR, "pipeline.lock")
    if os.path.exists(lock_path):
        age = datetime.now().timestamp() - os.path.getmtime(lock_path)
        is_running = age < 600

    return {
        "last_headlines_update": hl_time,
        "headlines_count":       hl_count,
        "is_running":            is_running,
        "data_available":        os.path.exists(hl_path) and os.path.exists(sec_path),
    }


class PipelineRequest(BaseModel):
    secret:       str
    max_per_feed: int = 12


@app.post("/api/pipeline/run")
def trigger_pipeline(req: PipelineRequest):
    if req.secret != PIPELINE_SECRET:
        raise HTTPException(status_code=401, detail="Invalid secret key.")

    max_per_feed = max(3, min(50, req.max_per_feed))
    total_approx = max_per_feed * 37

    def run():
        subprocess.run(
            [sys.executable, os.path.join(DATA_DIR, "pipeline.py"),
             "--once", f"--max-per-feed={max_per_feed}"],
            cwd=DATA_DIR,
        )

    threading.Thread(target=run, daemon=True).start()

    return {
        "status":       "started",
        "message":      f"Pipeline started — fetching up to {total_approx} headlines from 37 sources. Check status in ~2 minutes.",
        "started_at":   datetime.now().isoformat(),
        "max_per_feed": max_per_feed,
        "approx_total": total_approx,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


