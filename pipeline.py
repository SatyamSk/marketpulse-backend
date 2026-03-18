import feedparser
import pandas as pd
import numpy as np
import json
import os
import schedule
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── CONFIGURATION ─────────────────────────────────────────

RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://economictimes.indiatimes.com/small-biz/startups/rssfeeds/13357270.cms",
    "https://www.livemint.com/rss/markets",
    "https://www.business-standard.com/rss/markets-106.rss",
    "https://www.moneycontrol.com/rss/marketreports.xml",
]

SECTOR_WEIGHTS = {
    "Banking": 1.5, "Energy": 1.4, "Geopolitics": 1.4,
    "Fintech": 1.2, "IT": 1.2, "Manufacturing": 1.1,
    "Healthcare": 1.1, "FMCG": 1.0, "Startup": 0.9,
    "Retail": 0.8, "Other": 0.7,
}

# ── 1. FETCH ──────────────────────────────────────────────

def fetch_news():
    print("Fetching news from RSS feeds...")
    headlines = []
    seen_titles = set()

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:8]:
                title = entry.get("title", "").strip()
                description = entry.get("summary", entry.get("description", "")).strip()
                
                if not title or len(title) < 10 or title in seen_titles:
                    continue
                
                seen_titles.add(title)
                headlines.append({
                    "title": title,
                    "description": description[:600],
                    "source": feed.feed.get("title", "Unknown"),
                    "published": entry.get("published", datetime.now().isoformat()),
                    "url":         entry.get("link", entry.get("url", "")),
                })
        except Exception as e:
            print(f"  Feed error ({feed_url[:40]}): {e}")

    print(f"  Fetched {len(headlines)} unique headlines")
    return headlines

# ── 2. CLASSIFY (AI) ──────────────────────────────────────

def classify_headline(headline: dict) -> dict:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial news classifier for Indian markets. Return ONLY valid JSON."
                },
                {
                    "role": "user",
                    "content": f"""Classify this headline into JSON with these keys:
- sector: [Banking, IT, Energy, FMCG, Startup, Manufacturing, Healthcare, Fintech, Retail, Geopolitics, Other]
- sentiment: [positive, negative, neutral]
- sentiment_confidence: float 0.0 to 1.0
- impact_score: integer 1 to 10
- valence: float 0.0 to 1.0 (0=negative, 1=positive)
- arousal: float 0.0 to 1.0 (0=calm, 1=alarming)
- geopolitical_risk: boolean
- affected_companies: list of up to 2 Indian company names
- one_line_insight: one sharp sentence for a fund manager

Headline: {headline['title']}
Description: {headline['description'][:300]}"""
                }
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        res = json.loads(response.choices[0].message.content)
        
        # Enforce types & defaults
        res["sector"] = res.get("sector", "Other") if res.get("sector") in SECTOR_WEIGHTS else "Other"
        res["sentiment"] = res.get("sentiment", "neutral") if res.get("sentiment") in ["positive", "negative", "neutral"] else "neutral"
        res["sentiment_confidence"] = float(np.clip(res.get("sentiment_confidence", 0.7), 0, 1))
        res["impact_score"] = int(np.clip(res.get("impact_score", 5), 1, 10))
        res["valence"] = float(np.clip(res.get("valence", 0.5), 0, 1))
        res["arousal"] = float(np.clip(res.get("arousal", 0.5), 0, 1))
        res["geopolitical_risk"] = bool(res.get("geopolitical_risk", False))
        res["affected_companies"] = res.get("affected_companies", [])
        res["one_line_insight"] = str(res.get("one_line_insight", ""))
        return res

    except Exception as e:
        print(f"    Classification error: {e}")
        return {
            "sector": "Other", "sentiment": "neutral", "sentiment_confidence": 0.5,
            "impact_score": 5, "valence": 0.5, "arousal": 0.5, "geopolitical_risk": False,
            "affected_companies": [], "one_line_insight": "Classification unavailable."
        }

# ── 3. CALCULATE (PYTHON) ─────────────────────────────────

def calculate_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    
    # Headline-level Math
    sentiment_multipliers = {"positive": -0.5, "neutral": 0.0, "negative": 1.0}
    df["sentiment_num"] = df["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1}).fillna(0)
    
    df["weighted_risk_score"] = (
        df["impact_score"].astype(float) * df["sentiment"].map(sentiment_multipliers).fillna(0) * df["sector"].map(SECTOR_WEIGHTS).fillna(1.0) * df["geopolitical_risk"].apply(lambda x: 1.5 if x else 1.0)
    )
    df["weighted_risk_score"] = (df["weighted_risk_score"] * 10).clip(0, 100).round(2)

    # Z-Score Shock
# CORRECT — global z-score across all headlines
    # Uses overall mean and std so even single-sector headlines get proper scores
    global_mean = df["impact_score"].astype(float).mean()
    global_std  = df["impact_score"].astype(float).std()
    if global_std == 0 or pd.isna(global_std):
        global_std = 1.0
    
    df["z_score"] = ((df["impact_score"].astype(float) - global_mean) / global_std).round(2)
    
    # Python determines shock status — pure deterministic thresholds
    df["shock_status"] = df["z_score"].apply(
        lambda z: "Major Shock" if z > 2.0
        else "Shock"       if z > 1.0
        else "Watch"       if z > 0.5
        else "Normal"
    )
    # Save shock summary for API
    shock_summary = df[df["shock_status"].isin(["Shock", "Major Shock"])][["title", "sector", "sentiment", "impact_score", "z_score", "shock_status", "one_line_insight"]].sort_values("z_score", ascending=False)
    shock_summary.to_csv(os.path.join(DATA_DIR, "shock_headlines.csv"), index=False)
    print(f"  Shocks detected: {len(shock_summary)} ({len(df[df['shock_status']=='Major Shock'])} major)")

    # Sector-level Math
    sector_rows = []
    for sector, group in df.groupby("sector"):
        total = len(group)
        pos = (group["sentiment"] == "positive").sum()
        neg = (group["sentiment"] == "negative").sum()
        
        nss = round(((pos / total) - (neg / total)) * 100, 1) if total > 0 else 0.0
        
        total_impact = group["impact_score"].sum()
        iws = round((group["sentiment_num"] * group["impact_score"]).sum() / total_impact * 100, 1) if total_impact > 0 else 0.0
        
        conf_weight = (group["sentiment_confidence"] * group["impact_score"]).sum()
        cws = round((group["sentiment_num"] * group["sentiment_confidence"] * group["impact_score"]).sum() / conf_weight * 100, 1) if conf_weight > 0 else 0.0
        
        csi = round(nss * 0.25 + iws * 0.50 + cws * 0.25, 1)
        avg_risk = round(group["weighted_risk_score"].mean(), 1)
        avg_impact = round(group["impact_score"].mean(), 1)

        sector_rows.append({
            "sector": sector,
            "avg_weighted_risk": avg_risk,
            "sentiment_nss": nss,
            "impact_weighted_sentiment": iws,
            "confidence_weighted_sentiment": cws,
            "composite_sentiment_index": csi,
            "risk_level": "HIGH" if avg_risk >= 50 else "MEDIUM" if avg_risk >= 25 else "LOW",
            "avg_impact": avg_impact,
            "total_mentions": total,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": total - pos - neg,
            "geopolitical_flags": group["geopolitical_risk"].sum(),
            "benchmark_index": round(avg_risk * 0.5 + avg_impact * 3 + (100 - nss) * 0.2, 1),
            "sector_weight": SECTOR_WEIGHTS.get(sector, 1.0),
            "divergence": round(abs(nss - iws), 1),
            "divergence_flag": "High Divergence" if abs(nss - iws) > 30 else "Normal",
            "valence": round(group["valence"].mean(), 2) if "valence" in group else 0.5,
            "arousal": round(group["arousal"].mean(), 2) if "arousal" in group else 0.5,
        })

    sector_df = pd.DataFrame(sector_rows)
    
    # BCG Classification
    if not sector_df.empty:
        med_imp, med_risk = sector_df["avg_impact"].median(), sector_df["avg_weighted_risk"].median()
        sector_df["sector_classification"] = sector_df.apply(
            lambda r: "Watch Closely" if r["avg_impact"] >= med_imp and r["avg_weighted_risk"] >= med_risk else
                      "Opportunity" if r["avg_impact"] >= med_imp else
                      "Monitor Risk" if r["avg_weighted_risk"] >= med_risk else "Low Priority", axis=1
        )
    
    # Velocity Mapping
    master_path = os.path.join(DATA_DIR, "master_sector_scores.csv")
    if os.path.exists(master_path):
        history = pd.read_csv(master_path)
        if not history.empty and "composite_sentiment_index" in history.columns:
            last_csi = history.groupby("sector")["composite_sentiment_index"].last().to_dict()
            sector_df["sentiment_velocity"] = sector_df.apply(
                lambda r: round(r["composite_sentiment_index"] - last_csi.get(r["sector"], r["composite_sentiment_index"]), 1), axis=1
            )
    
    if "sentiment_velocity" not in sector_df.columns:
        sector_df["sentiment_velocity"] = 0.0

    return df, sector_df

# ── 4. EXECUTE & SAVE ─────────────────────────────────────

def run_pipeline():
    print(f"\n{'='*50}\nMARKETPULSE PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'='*50}")
    
    headlines = fetch_news()
    if not headlines: return

    print(f"\nClassifying {len(headlines)} headlines...")
    analyzed = [{**h, **classify_headline(h)} for h in headlines]
    
    print("Running analytics...")
    headlines_df, sector_df = calculate_metrics(pd.DataFrame(analyzed))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_date = datetime.now().strftime("%Y-%m-%d")

    # Snapshot Saves
    headlines_df.to_csv(os.path.join(DATA_DIR, f"headlines_analyzed_{timestamp}.csv"), index=False)
    sector_df.to_csv(os.path.join(DATA_DIR, f"sector_benchmark_{timestamp}.csv"), index=False)
    headlines_df.to_csv(os.path.join(DATA_DIR, "latest_headlines.csv"), index=False)
    sector_df.to_csv(os.path.join(DATA_DIR, "latest_sectors.csv"), index=False)

    # Master Appends
    headlines_df["run_date"] = run_date
    sector_df["run_date"] = run_date
    
    for df, filename in [(headlines_df, "master_headlines.csv"), (sector_df, "master_sector_scores.csv")]:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            pd.concat([pd.read_csv(path), df], ignore_index=True).to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)

    # Trend Calculations
    master_sec_path = os.path.join(DATA_DIR, "master_sector_scores.csv")
    if os.path.exists(master_sec_path):
        master = pd.read_csv(master_sec_path)
        trend = master.groupby(["run_date", "sector"])["composite_sentiment_index"].mean().reset_index()
        trend["csi_3day_ma"] = trend.groupby("sector")["composite_sentiment_index"].transform(lambda x: x.rolling(3, min_periods=1).mean()).round(2)
        trend.to_csv(os.path.join(DATA_DIR, "sector_trend_analysis.csv"), index=False)

    print("\nPIPELINE COMPLETE.")

if __name__ == "__main__":
    run_pipeline()
    schedule.every(1).hours.do(run_pipeline)
    while True:
        schedule.run_pending()
        time.sleep(60)
