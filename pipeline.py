import feedparser
import pandas as pd
import numpy as np
import json
import os
import sys
import hashlib
import concurrent.futures
import traceback
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv
from dateutil import parser as dateparser
from github import Github

load_dotenv()

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Your new Data Repository
GITHUB_REPO = "SatyamSk/MarketPulseAIData"

RSS_FEEDS = [
    ("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",      "ET Markets"),
    ("https://economictimes.indiatimes.com/economy/rssfeeds/1373380680.cms",      "ET Economy"),
    ("https://economictimes.indiatimes.com/tech/rssfeeds/13357263.cms",           "ET Tech"),
    ("https://www.livemint.com/rss/markets",   "Livemint Markets"),
    ("https://www.business-standard.com/rss/markets-106.rss",        "BS Markets"),
    ("https://www.moneycontrol.com/rss/latestnews.xml",    "MC Latest News"),
    ("https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",   "PIB Economy"),
    ("https://www.rbi.org.in/scripts/rss.aspx",                   "RBI"),
    ("https://www.sebi.gov.in/sebi_data/rss/sebi_news.xml",       "SEBI"),
    ("https://feeds.reuters.com/reuters/INbusinessNews", "Reuters India"),
] # Truncated for brevity, keep your full list of 37 here!

SECTOR_WEIGHTS = {"Banking": 1.5, "Energy": 1.4, "IT": 1.2, "Fintech": 1.2, "Manufacturing": 1.1, "Healthcare": 1.1, "FMCG": 1.0, "Startup": 0.9, "Retail": 0.8, "Other": 0.7}
CATALYST_WEIGHTS = {"government_contract": 1.7, "policy_change": 1.6, "pib_announcement": 1.6, "rbi_action": 1.5, "sebi_action": 1.5, "fii_flow": 1.4, "capex_announcement": 1.4, "earnings": 1.3, "sector_tailwind": 1.2, "regulatory": 1.1, "management_change": 1.0, "global_event": 0.9, "other": 0.7}
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
    print(f"  Fetching — {max_per_feed} per feed...")
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
    except Exception as e:
        return {"sector": "Other", "sentiment": "neutral", "sentiment_confidence": 0.5, "impact_score": 5, "valence": 0.5, "arousal": 0.5, "geopolitical_risk": False, "affected_companies": [], "second_order_beneficiaries": [], "catalyst_type": "other", "price_direction": "neutral", "time_horizon": "intraday", "conviction": "low", "macro_sensitivity": "medium", "one_line_insight": "AI error.", "signal_reason": "", "contrarian_flag": False, "contrarian_reason": ""}

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
            except Exception as e: 
                pass
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
        momentum_score = round((float((group.get("price_direction", pd.Series([])).eq("bullish")).mean()) - float((group.get("price_direction", pd.Series([])).eq("bearish")).mean())) * 100, 1)

        sector_rows.append({
            "sector": sector, "avg_weighted_risk": avg_risk, "sentiment_nss": nss, "composite_sentiment_index": csi,
            "sentiment_velocity": 0.0, "risk_level": "HIGH" if avg_risk >= 50 else "MEDIUM" if avg_risk >= 25 else "LOW",
            "avg_impact": avg_impact, "momentum_score": momentum_score,
            "divergence_flag": "High Divergence" if abs(nss - iws) > 30 else "Normal",
            "sector_classification": "Watch Closely" if avg_impact >= 5 and avg_risk >= 25 else "Monitor Risk",
            "investment_signal": "BUY BIAS" if csi > 30 and avg_risk < 25 and momentum_score > 20 else "NEUTRAL"
        })
    return df, pd.DataFrame(sector_rows)

def calculate_market_stress_index(headlines_df: pd.DataFrame, sector_df: pd.DataFrame) -> dict:
    if headlines_df.empty: return {"msi": 0, "level": "Low"}
    risk_comp = min(float(sector_df["avg_weighted_risk"].mean()), 100) if not sector_df.empty else 0
    shock_comp = min(headlines_df["shock_status"].isin(["Major Shock", "Shock"]).sum() / len(headlines_df) * 300, 100)
    msi = round(risk_comp * 0.60 + shock_comp * 0.40, 1)
    return {"msi": msi, "level": "Critical" if msi >= 75 else "High" if msi >= 50 else "Elevated" if msi >= 30 else "Low"}


# ==========================================
# GITHUB OVERRIDE SYSTEM
# ==========================================
def push_to_github(repo, file_path, content_str, commit_message):
    try:
        contents = repo.get_contents(file_path, ref="main")
        repo.update_file(contents.path, commit_message, content_str, contents.sha, branch="main")
    except Exception:
        repo.create_file(file_path, commit_message, content_str, branch="main")

def save_all(headlines_df: pd.DataFrame, sector_df: pd.DataFrame, msi: dict):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("  [!] ERROR: GITHUB_TOKEN is missing. Cannot upload to GitHub.")
        return
        
    try:
        print(f"  Uploading directly to GitHub Repository ({GITHUB_REPO})...")
        g = Github(token)
        repo = g.get_repo(GITHUB_REPO)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Overwrite Main Files
        push_to_github(repo, "latest_headlines.csv", headlines_df.to_csv(index=False), f"Update headlines {timestamp}")
        push_to_github(repo, "latest_sectors.csv", sector_df.to_csv(index=False), f"Update sectors {timestamp}")
        push_to_github(repo, "latest_msi.json", json.dumps(msi, indent=2), f"Update MSI {timestamp}")
        push_to_github(repo, "pipeline_status.json", json.dumps({"last_run": datetime.now().isoformat()}), f"Update Status {timestamp}")

        # 2. Extract Shocks and Overwrite
        shock_cols = [c for c in ["title", "sector", "sentiment", "impact_score", "z_score", "shock_status", "one_line_insight", "geopolitical_risk", "url"] if c in headlines_df.columns]
        shock_df = headlines_df[headlines_df["shock_status"].isin(["Major Shock", "Shock", "Watch"])]
        if not shock_df.empty:
            push_to_github(repo, "shock_headlines.csv", shock_df[shock_cols].to_csv(index=False), f"Update shocks {timestamp}")

        print("  ✓ GitHub Data Override Complete! The API can now read the fresh files.")
    except Exception as e:
        print(f"  [!] CRITICAL GITHUB UPLOAD ERROR: {e}")
        traceback.print_exc()

def run_pipeline():
    start_time = datetime.now()
    print(f"\n{'='*60}\nMARKETPULSE PIPELINE — {start_time.strftime('%Y-%m-%d %H:%M')}\n{'='*60}")
    
    try:
        headlines = fetch_news()
        if not headlines: 
            return
        
        analyzed_headlines = process_all_headlines(headlines) 
        if not analyzed_headlines:
            return

        df = pd.DataFrame(analyzed_headlines)
        headlines_df, sector_df = calculate_metrics(df)
        msi = calculate_market_stress_index(headlines_df, sector_df)
        
        save_all(headlines_df, sector_df, msi)
        print("  ✅ PIPELINE COMPLETE.\n")
    except Exception as e:
        print(f"  [!] FATAL PIPELINE ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    lock_path = "/tmp/pipeline.lock" 
    with open(lock_path, "w") as f: f.write(datetime.now().isoformat())
    try: 
        run_pipeline()
    finally:
        if os.path.exists(lock_path): os.remove(lock_path)
