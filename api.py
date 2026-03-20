import feedparser
import pandas as pd
import numpy as np
import json
import os
import sys
import hashlib
import concurrent.futures
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv
from dateutil import parser as dateparser
from sqlalchemy import create_engine

load_dotenv()

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RSS_FEEDS = [
    ("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",      "ET Markets"),
    ("https://economictimes.indiatimes.com/economy/rssfeeds/1373380680.cms",      "ET Economy"),
    ("https://economictimes.indiatimes.com/tech/rssfeeds/13357263.cms",           "ET Tech"),
    ("https://economictimes.indiatimes.com/small-biz/startups/rssfeeds/13357270.cms","ET Startups"),
    ("https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms",       "ET Industry"),
    ("https://www.livemint.com/rss/markets",   "Livemint Markets"),
    ("https://www.livemint.com/rss/companies", "Livemint Companies"),
    ("https://www.livemint.com/rss/economy",   "Livemint Economy"),
    ("https://www.livemint.com/rss/politics",  "Livemint Politics"),
    ("https://www.business-standard.com/rss/markets-106.rss",        "BS Markets"),
    ("https://www.business-standard.com/rss/economy-policy-101.rss", "BS Economy"),
    ("https://www.business-standard.com/rss/finance-103.rss",        "BS Finance"),
    ("https://www.business-standard.com/rss/companies-101.rss",      "BS Companies"),
    ("https://www.moneycontrol.com/rss/marketreports.xml", "MC Market Reports"),
    ("https://www.moneycontrol.com/rss/latestnews.xml",    "MC Latest News"),
    ("https://www.moneycontrol.com/rss/business.xml",      "MC Business"),
    ("https://www.financialexpress.com/market/feed/",  "Financial Express Markets"),
    ("https://www.financialexpress.com/economy/feed/", "Financial Express Economy"),
    ("https://www.thehindubusinessline.com/markets/?service=rss",   "BL Markets"),
    ("https://www.thehindubusinessline.com/economy/?service=rss",   "BL Economy"),
    ("https://www.thehindubusinessline.com/companies/?service=rss", "BL Companies"),
    ("https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",   "PIB Economy"),
    ("https://pib.gov.in/RssMain.aspx?ModId=37&Lang=1&Regid=3",  "PIB Commerce"),
    ("https://pib.gov.in/RssMain.aspx?ModId=25&Lang=1&Regid=3",  "PIB Finance"),
    ("https://pib.gov.in/RssMain.aspx?ModId=3&Lang=1&Regid=3",   "PIB Infrastructure"),
    ("https://pib.gov.in/RssMain.aspx?ModId=14&Lang=1&Regid=3",  "PIB Defence"),
    ("https://www.rbi.org.in/scripts/rss.aspx",                   "RBI"),
    ("https://www.sebi.gov.in/sebi_data/rss/sebi_news.xml",       "SEBI"),
    ("https://inc42.com/feed/",              "Inc42"),
    ("https://entrackr.com/feed/",           "Entrackr"),
    ("https://yourstory.com/feed",           "YourStory"),
    ("https://mercomindia.com/feed/",        "Mercom India"),
    ("https://www.constructionworld.in/feed","Construction World"),
    ("https://www.cio.in/rss.xml",           "CIO India"),
    ("https://feeds.reuters.com/reuters/INbusinessNews", "Reuters India"),
    ("https://feeds.bbci.co.uk/news/business/rss.xml",  "BBC Business"),
    ("https://www.thehindu.com/business/Economy/feeder/default.rss", "The Hindu Economy"),
]

SECTOR_WEIGHTS = {
    "Banking":       1.5, "Energy":        1.4, "IT":            1.2,
    "Fintech":       1.2, "Manufacturing": 1.1, "Healthcare":    1.1,
    "FMCG":          1.0, "Startup":       0.9, "Retail":        0.8,
    "Other":         0.7,
}

CATALYST_WEIGHTS = {
    "government_contract":  1.7, "policy_change":        1.6, "pib_announcement":     1.6,
    "rbi_action":           1.5, "sebi_action":          1.5, "fii_flow":             1.4,
    "capex_announcement":   1.4, "earnings":             1.3, "sector_tailwind":      1.2,
    "regulatory":           1.1, "management_change":    1.0, "global_event":         0.9,
    "other":                0.7,
}

SIGNAL_HALF_LIFE = {"intraday": 4, "swing_2_5days": 48, "positional_weeks": 168}

VALID_SECTORS    = set(SECTOR_WEIGHTS.keys())
VALID_SENTIMENTS = {"positive", "negative", "neutral"}

def get_max_per_feed() -> int:
    for arg in sys.argv:
        if arg.startswith("--max-per-feed="):
            try: return max(3, min(50, int(arg.split("=")[1])))
            except: pass
    return 12

def parse_publish_time(entry: dict) -> datetime:
    for field in ["published", "updated", "created"]:
        val = entry.get(field)
        if val:
            try:
                dt = dateparser.parse(val)
                if dt and dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except: pass
    return datetime.now(timezone.utc)

def fetch_news() -> list[dict]:
    max_per_feed = get_max_per_feed()
    print(f"  Fetching — {max_per_feed} per feed · {len(RSS_FEEDS)} sources...")
    headlines, seen = [], set()

    for feed_url, feed_label in RSS_FEEDS:
        try:
            feed  = feedparser.parse(feed_url)
            count = 0
            for entry in feed.entries:
                if count >= max_per_feed: break
                title = entry.get("title", "").strip()
                desc  = entry.get("summary", entry.get("description", "")).strip()
                if len(title) < 10 or title in seen: continue
                
                content_hash = hashlib.md5(title[:60].lower().encode()).hexdigest()[:8]
                if content_hash in seen: continue
                seen.update([title, content_hash])

                publish_dt = parse_publish_time(entry)
                hours_old  = max(0, (datetime.now(timezone.utc) - publish_dt).total_seconds() / 3600)

                headlines.append({
                    "title": title, "description": desc[:800], "source": feed_label,
                    "source_url": feed_url, "published": publish_dt.isoformat(),
                    "hours_old": round(hours_old, 1), "url": entry.get("link", entry.get("url", "")).strip(),
                    "is_govt_source": any(x in feed_label for x in ["PIB", "RBI", "SEBI"]),
                })
                count += 1
        except Exception as e: print(f"    FAIL {feed_label} → {e}")
    return headlines

def classify_headline(headline: dict) -> dict:
    sectors_list = ", ".join(sorted(VALID_SECTORS))
    govt_note = "OFFICIAL REGULATORY source. Give impact_score >= 7 unless routine." if headline.get("is_govt_source") else ""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a senior equity analyst. Find NON-OBVIOUS opportunities. {govt_note} Return valid JSON."},
                {"role": "user", "content": f"Analyze this headline. Return keys: sector [{sectors_list}], sentiment [positive, negative, neutral], sentiment_confidence (0-1), impact_score (1-10), valence (0-1), arousal (0-1), geopolitical_risk (bool), affected_companies (list max 4), second_order_beneficiaries (list max 4), catalyst_type [{','.join(CATALYST_WEIGHTS.keys())}], price_direction [bullish, bearish, neutral], time_horizon [intraday, swing_2_5days, positional_weeks], conviction [high, medium, low], macro_sensitivity [high, medium, low], one_line_insight (string max 300c), signal_reason (string max 300c), contrarian_flag (bool), contrarian_reason (string).\n\nHeadline: {headline['title']}\nDesc: {headline['description'][:400]}"}
            ],
            temperature=0.1, response_format={"type": "json_object"},
        )
        res = json.loads(response.choices[0].message.content)
        sector = res.get("sector", "Other")
        return {
            "sector": sector if sector in VALID_SECTORS else "Other",
            "sentiment": res.get("sentiment", "neutral") if res.get("sentiment") in VALID_SENTIMENTS else "neutral",
            "sentiment_confidence": float(np.clip(res.get("sentiment_confidence", 0.7), 0.0, 1.0)),
            "impact_score": int(np.clip(res.get("impact_score", 5), 1, 10)),
            "valence": float(np.clip(res.get("valence", 0.5), 0.0, 1.0)),
            "arousal": float(np.clip(res.get("arousal", 0.5), 0.0, 1.0)),
            "geopolitical_risk": bool(res.get("geopolitical_risk", False)),
            "affected_companies": res.get("affected_companies", []),
            "second_order_beneficiaries": res.get("second_order_beneficiaries", []),
            "catalyst_type": res.get("catalyst_type", "other"),
            "price_direction": res.get("price_direction", "neutral"),
            "time_horizon": res.get("time_horizon", "intraday"),
            "conviction": res.get("conviction", "low"),
            "macro_sensitivity": res.get("macro_sensitivity", "medium"),
            "one_line_insight": str(res.get("one_line_insight", ""))[:300],
            "signal_reason": str(res.get("signal_reason", ""))[:300],
            "contrarian_flag": bool(res.get("contrarian_flag", False)),
            "contrarian_reason": str(res.get("contrarian_reason", ""))[:300],
        }
    except Exception:
        return {"sector": "Other", "sentiment": "neutral", "sentiment_confidence": 0.5, "impact_score": 5, "valence": 0.5, "arousal": 0.5, "geopolitical_risk": False, "affected_companies": [], "second_order_beneficiaries": [], "catalyst_type": "other", "price_direction": "neutral", "time_horizon": "intraday", "conviction": "low", "macro_sensitivity": "medium", "one_line_insight": "", "signal_reason": "", "contrarian_flag": False, "contrarian_reason": ""}

def process_all_headlines(headlines: list) -> list:
    print(f"  Classifying {len(headlines)} headlines using High-Speed Threading...")
    analyzed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_h = {executor.submit(classify_headline, h): h for h in headlines}
        for future in concurrent.futures.as_completed(future_to_h):
            h = future_to_h[future]
            try:
                analyzed.append({**h, **future.result()})
                print(f"    ✓ Analyzed: {h['title'][:50]}...")
            except Exception as e: print(f"    [!] Failed: {e}")
    return analyzed

def calculate_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["sentiment_num"] = df["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1}).fillna(0)
    df["weighted_risk_score"] = (df["impact_score"].astype(float) * df["sentiment"].map({"positive": -0.5, "neutral": 0.0, "negative": 1.0}).fillna(0) * df["sector"].map(SECTOR_WEIGHTS).fillna(1.0) * df["geopolitical_risk"].apply(lambda x: 1.5 if bool(x) else 1.0) * df["is_govt_source"].apply(lambda x: 1.3 if bool(x) else 1.0))
    df["weighted_risk_score"] = (df["weighted_risk_score"] * 10).clip(0, 100).round(2)
    df["signal_decay"] = df.apply(lambda r: float(np.exp(-0.693 * float(r.get("hours_old", 0)) / SIGNAL_HALF_LIFE.get(str(r.get("time_horizon", "intraday")), 4))), axis=1).round(3)
    df["recency_weighted_impact"] = (df["impact_score"].astype(float) * df["signal_decay"]).round(2)
    df["catalyst_weight"] = df["catalyst_type"].map(CATALYST_WEIGHTS).fillna(0.7)

    g_mean = float(df["impact_score"].astype(float).mean())
    g_std = float(df["impact_score"].astype(float).std()) or 1.0
    df["z_score"] = ((df["impact_score"].astype(float) - g_mean) / g_std).round(2)
    df["shock_status"] = df["z_score"].apply(lambda z: "Major Shock" if z > 2.0 else "Shock" if z > 1.0 else "Watch" if z > 0.5 else "Normal")

    sector_rows = []
    for sector, group in df.groupby("sector"):
        total = len(group)
        pos, neg = int((group["sentiment"] == "positive").sum()), int((group["sentiment"] == "negative").sum())
        nss = round(((pos / total) - (neg / total)) * 100, 1) if total > 0 else 0.0
        iws = round((group["sentiment_num"] * group["impact_score"].astype(float)).sum() / float(group["impact_score"].astype(float).sum()) * 100, 1) if group["impact_score"].astype(float).sum() > 0 else 0.0
        csi = round(nss * 0.40 + iws * 0.60, 1)
        
        avg_risk = round(float(group["weighted_risk_score"].mean()), 1)
        avg_impact = round(float(group["impact_score"].astype(float).mean()), 1)
        momentum_score = round((float((group.get("price_direction", pd.Series([])) == "bullish").mean()) - float((group.get("price_direction", pd.Series([])) == "bearish").mean())) * 100, 1)

        sector_rows.append({
            "sector": sector, "avg_weighted_risk": avg_risk, "sentiment_nss": nss, "composite_sentiment_index": csi,
            "sentiment_velocity": 0.0, "risk_level": "HIGH" if avg_risk >= 50 else "MEDIUM" if avg_risk >= 25 else "LOW",
            "avg_impact": avg_impact, "momentum_score": momentum_score,
            "divergence_flag": "High Divergence" if abs(nss - iws) > 30 else "Normal",
            "sector_classification": "Watch Closely" if avg_impact >= 5 and avg_risk >= 25 else "Monitor Risk",
            "investment_signal": "BUY BIAS" if csi > 30 and risk < 25 and momentum_score > 20 else "NEUTRAL"
        })
    return df, pd.DataFrame(sector_rows)

def calculate_market_stress_index(headlines_df: pd.DataFrame, sector_df: pd.DataFrame) -> dict:
    if headlines_df.empty: return {"msi": 0, "level": "Low"}
    risk_comp = min(float(sector_df["avg_weighted_risk"].mean()), 100) if not sector_df.empty else 0
    shock_comp = min(headlines_df["shock_status"].isin(["Major Shock", "Shock"]).sum() / len(headlines_df) * 300, 100)
    msi = round(risk_comp * 0.60 + shock_comp * 0.40, 1)
    return {"msi": msi, "level": "Critical" if msi >= 75 else "High" if msi >= 50 else "Elevated" if msi >= 30 else "Low"}

def save_all(headlines_df: pd.DataFrame, sector_df: pd.DataFrame, msi: dict):
    # SUPABASE DATABASE CONNECTION
    DB_URL = os.getenv("DATABASE_URL")
    if DB_URL and DB_URL.startswith("postgres://"):
        DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)
    
    if not DB_URL:
        print("  [!] ERROR: DATABASE_URL missing. Cannot save to Supabase.")
        return

    engine = create_engine(DB_URL)
    run_date = datetime.now().strftime("%Y-%m-%d")
    print("  Uploading data directly to Supabase PostgreSQL...")

    headlines_df.astype(str).to_sql("latest_headlines", engine, if_exists="replace", index=False)
    sector_df.astype(str).to_sql("latest_sectors", engine, if_exists="replace", index=False)
    
    pd.DataFrame([{"msi_data": json.dumps(msi)}]).to_sql("latest_msi", engine, if_exists="replace", index=False)
    pd.DataFrame([{"last_run": datetime.now().isoformat()}]).to_sql("pipeline_status", engine, if_exists="replace", index=False)
    
    shock_cols = [c for c in ["title", "sector", "sentiment", "impact_score", "z_score", "shock_status", "one_line_insight", "geopolitical_risk", "url"] if c in headlines_df.columns]
    headlines_df[headlines_df["shock_status"].isin(["Major Shock", "Shock", "Watch"])][shock_cols].astype(str).to_sql("shock_headlines", engine, if_exists="replace", index=False)
    
    headlines_df.assign(run_date=run_date).astype(str).to_sql("master_headlines", engine, if_exists="append", index=False)
    sector_df.assign(run_date=run_date).astype(str).to_sql("master_sector_scores", engine, if_exists="append", index=False)
    
    try:
        master = pd.read_sql_table("master_sector_scores", engine)
        if "composite_sentiment_index" in master.columns:
            master["composite_sentiment_index"] = pd.to_numeric(master["composite_sentiment_index"])
            trend = master.groupby(["run_date", "sector"])["composite_sentiment_index"].mean().reset_index()
            trend["csi_3day_ma"] = trend.groupby("sector")["composite_sentiment_index"].transform(lambda x: x.rolling(3, min_periods=1).mean()).round(2)
            trend.to_sql("sector_trend_analysis", engine, if_exists="replace", index=False)
    except Exception as e: print(f"  Trend error: {e}")
    print("  ✓ Supabase upload complete!")

def run_pipeline():
    start_time = datetime.now()
    print(f"\n{'='*60}\nMARKETPULSE PIPELINE — {start_time.strftime('%Y-%m-%d %H:%M')}\n{'='*60}")
    
    headlines = fetch_news()
    if not headlines: return
    
    analyzed_headlines = process_all_headlines(headlines) # 10x Speed Threading!
    df = pd.DataFrame(analyzed_headlines)
    headlines_df, sector_df = calculate_metrics(df)
    msi = calculate_market_stress_index(headlines_df, sector_df)
    
    save_all(headlines_df, sector_df, msi)
    print("  PIPELINE COMPLETE.\n")
@app.post("/api/pipeline/run")
def trigger_pipeline(req: PipelineRequest):
    if req.secret != PIPELINE_SECRET: 
        raise HTTPException(status_code=401, detail="Invalid secret key.")
    
    max_per_feed = max(3, min(50, req.max_per_feed))
    total_approx = max_per_feed * 37

    def run():
        import traceback
        print("\n🚀 BACKGROUND THREAD STARTED: Launching pipeline.py...")
        try:
            # We use "python" explicitly and capture ALL outputs and errors
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
        "message": f"Pipeline diagnostic started. Check Render logs.", 
        "started_at": datetime.now().isoformat()
    }
if __name__ == "__main__":
    lock_path = "/tmp/pipeline.lock" # Linux /tmp hides it from auto-reload!
    with open(lock_path, "w") as f: f.write(datetime.now().isoformat())
    try: run_pipeline()
    finally:
        if os.path.exists(lock_path): os.remove(lock_path)
