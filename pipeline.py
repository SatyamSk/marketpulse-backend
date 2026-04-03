import feedparser
import pandas as pd
import numpy as np
import json
import os
import sys
import hashlib
import concurrent.futures
import traceback
from datetime import datetime, timezone, timedelta
from openai import OpenAI
from dotenv import load_dotenv
from dateutil import parser as dateparser
from github import Github

load_dotenv()

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# GITHUB CONFIG
GITHUB_REPO = "SatyamSk/MarketPulseAIData"

# STRICT IST TIMEZONE DEFINITION
IST = timezone(timedelta(hours=5, minutes=30))

RSS_FEEDS = [
    ("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",      "ET Markets"),
    ("https://economictimes.indiatimes.com/economy/rssfeeds/1373380680.cms",      "ET Economy"),
    ("https://economictimes.indiatimes.com/tech/rssfeeds/13357263.cms",           "ET Tech"),
    ("https://economictimes.indiatimes.com/small-biz/startups/rssfeeds/13357270.cms","ET Startups"),
    ("https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms",       "ET Industry"),
    ("https://www.livemint.com/rss/markets",   "Livemint Markets"),
    ("https://www.livemint.com/rss/companies", "Livemint Companies"),
    ("https://www.livemint.com/rss/economy",   "Livemint Economy"),
    ("https://www.business-standard.com/rss/markets-106.rss",        "BS Markets"),
    ("https://www.business-standard.com/rss/economy-policy-101.rss", "BS Economy"),
    ("https://www.moneycontrol.com/rss/latestnews.xml",    "MC Latest News"),
    ("https://www.financialexpress.com/market/feed/",  "Financial Express Markets"),
    ("https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",   "PIB Economy"),
    ("https://www.rbi.org.in/scripts/rss.aspx",                   "RBI"),
    ("https://www.sebi.gov.in/sebi_data/rss/sebi_news.xml",       "SEBI"),
    ("https://feeds.reuters.com/reuters/INbusinessNews", "Reuters India"),
] 

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
    now_ist = datetime.now(IST)
    print(f"  Fetching — {max_per_feed} per feed. Strict IST Horizon applied...")
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
                
                # 1. PARSE TIME AND CONVERT TO IST
                publish_dt = parse_publish_time(entry)
                publish_dt_ist = publish_dt.astimezone(IST)
                
                # 2. STRICT TIME HORIZON FILTER (Today or Yesterday ONLY)
                days_old = (now_ist.date() - publish_dt_ist.date()).days
                if days_old > 1:
                    continue # Brutally drop old news
                    
                seen.update([title, content_hash])
                hours_old = max(0, (now_ist - publish_dt_ist).total_seconds() / 3600)

                headlines.append({
                    "title": title, "description": desc[:800], "source": feed_label,
                    "source_url": feed_url, "published": publish_dt_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
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
    print(f"  Classifying {len(headlines)} fresh IST headlines using Threading...")
    analyzed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_h = {executor.submit(classify_headline, h): h for h in headlines}
        for future in concurrent.futures.as_completed(future_to_h):
            h = future_to_h[future]
            try:
                analyzed.append({**h, **future.result()})
            except Exception: pass
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
        total  = len(group)
        pos    = int((group["sentiment"] == "positive").sum())
        neg    = int((group["sentiment"] == "negative").sum())
        neu    = int((group["sentiment"] == "neutral").sum())
        nss    = round(((pos / total) - (neg / total)) * 100, 1) if total > 0 else 0.0
        total_impact = float(group["impact_score"].astype(float).sum())
        iws    = round((group["sentiment_num"] * group["impact_score"].astype(float)).sum() / total_impact * 100, 1) if total_impact > 0 else 0.0
        csi    = round(nss * 0.40 + iws * 0.60, 1)
        avg_risk   = round(float(group["weighted_risk_score"].mean()), 1)
        avg_impact = round(float(group["impact_score"].astype(float).mean()), 1)
        divergence = round(abs(nss - iws), 1)
        momentum_score = round((float((group.get("price_direction", pd.Series([])).eq("bullish")).mean()) - float((group.get("price_direction", pd.Series([])).eq("bearish")).mean())) * 100, 1)
        contrarian_count = int(group.get("contrarian_flag", pd.Series([False])).apply(bool).sum())
        govt_signals     = int(group.get("is_govt_source", pd.Series([False])).apply(bool).sum())

        sector_rows.append({
            "sector":                    sector,
            "avg_weighted_risk":         avg_risk,
            "sentiment_nss":             nss,
            "impact_weighted_sentiment": iws,
            "composite_sentiment_index": csi,
            "sentiment_velocity":        0.0,
            "risk_level":                "HIGH" if avg_risk >= 45 else "MEDIUM" if avg_risk >= 20 else "LOW",
            "avg_impact":                avg_impact,
            "total_mentions":            total,
            "positive_count":            pos,
            "negative_count":            neg,
            "neutral_count":             neu,
            "momentum_score":            momentum_score,
            "divergence":                divergence,
            "divergence_flag":           "High Divergence" if divergence > 30 else "Normal",
            "govt_signals":              govt_signals,
            "contrarian_count":          contrarian_count,
            "geopolitical_flags":        int(group["geopolitical_risk"].apply(bool).sum()),
            "valence":                   round(float(group.get("valence", pd.Series([0.5])).astype(float).mean()), 2),
            "arousal":                   round(float(group.get("arousal",  pd.Series([0.5])).astype(float).mean()), 2),
            "sector_classification":     "Watch Closely",
            "investment_signal":         "NEUTRAL",
        })

    sector_df = pd.DataFrame(sector_rows)
    if sector_df.empty:
        return df, sector_df

    # BCG classification relative to today's median
    med_impact = sector_df["avg_impact"].median()
    med_risk   = sector_df["avg_weighted_risk"].median()

    def bcg_classify(row):
        hi_i = row["avg_impact"]        >= med_impact
        hi_r = row["avg_weighted_risk"] >= med_risk
        if hi_i and hi_r:       return "Watch Closely"
        elif hi_i and not hi_r: return "Opportunity"
        elif not hi_i and hi_r: return "Monitor Risk"
        else:                   return "Low Priority"

    sector_df["sector_classification"] = sector_df.apply(bcg_classify, axis=1)

    # Investment signal — pure Python deterministic
    def investment_signal(row):
        csi  = row["composite_sentiment_index"]
        risk = row["avg_weighted_risk"]
        mom  = row["momentum_score"]
        div  = row["divergence_flag"]
        vel  = row["sentiment_velocity"]
        cont = row["contrarian_count"]
        if csi > 30 and risk < 25 and mom > 15:    return "BUY BIAS"
        if csi < -25 or (risk > 50 and mom < -20): return "AVOID"
        if div == "High Divergence" and csi > 0:    return "CAUTION"
        if mom > 10 and csi > 0 and vel >= 0:       return "IMPROVING"
        if cont >= 2:                               return "CONTRARIAN WATCH"
        return "NEUTRAL"

    sector_df["investment_signal"] = sector_df.apply(investment_signal, axis=1)

    # Velocity: compare CSI against previous run saved on GitHub
    try:
        import requests as _req, io as _io
        prev_url = f"https://raw.githubusercontent.com/{os.getenv('GITHUB_REPO', 'SatyamSk/MarketPulseAIData')}/main/latest_sectors.csv"
        r = _req.get(prev_url, timeout=6)
        if r.status_code == 200:
            prev_df = pd.read_csv(_io.StringIO(r.text))
            if "composite_sentiment_index" in prev_df.columns and "sector" in prev_df.columns:
                prev_csi = dict(zip(prev_df["sector"], pd.to_numeric(prev_df["composite_sentiment_index"], errors="coerce")))
                for idx, row in sector_df.iterrows():
                    prev = prev_csi.get(row["sector"])
                    if prev is not None and not pd.isna(prev):
                        sector_df.at[idx, "sentiment_velocity"] = round(float(row["composite_sentiment_index"]) - float(prev), 1)
        sector_df["investment_signal"] = sector_df.apply(investment_signal, axis=1)
    except Exception as e:
        print(f"  Velocity calc skipped: {e}")

    return df, sector_df

def calculate_market_stress_index(headlines_df: pd.DataFrame, sector_df: pd.DataFrame) -> dict:
    if headlines_df.empty: return {"msi": 0, "level": "Low"}
    risk_comp = min(float(sector_df["avg_weighted_risk"].mean()), 100) if not sector_df.empty else 0
    shock_comp = min(headlines_df["shock_status"].isin(["Major Shock", "Shock"]).sum() / len(headlines_df) * 300, 100)
    msi = round(risk_comp * 0.60 + shock_comp * 0.40, 1)
    return {"msi": msi, "level": "Critical" if msi >= 75 else "High" if msi >= 50 else "Elevated" if msi >= 30 else "Low"}

def push_to_github(repo, file_path, content_str, commit_message):
    try:
        contents = repo.get_contents(file_path, ref="main")
        repo.update_file(contents.path, commit_message, content_str, contents.sha, branch="main")
    except Exception:
        repo.create_file(file_path, commit_message, content_str, branch="main")

def save_all(headlines_df: pd.DataFrame, sector_df: pd.DataFrame, msi: dict):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("  [!] ERROR: GITHUB_TOKEN is missing.")
        return
    try:
        print(f"  Uploading directly to GitHub Repository ({GITHUB_REPO})...")
        g = Github(token)
        repo = g.get_repo(GITHUB_REPO)
        timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")

        push_to_github(repo, "latest_headlines.csv", headlines_df.to_csv(index=False), f"Update headlines {timestamp}")
        push_to_github(repo, "latest_sectors.csv", sector_df.to_csv(index=False), f"Update sectors {timestamp}")
        push_to_github(repo, "latest_msi.json", json.dumps(msi, indent=2), f"Update MSI {timestamp}")
        push_to_github(repo, "pipeline_status.json", json.dumps({"last_run": timestamp}), f"Update Status {timestamp}")

        shock_cols = [c for c in ["title", "sector", "sentiment", "impact_score", "z_score", "shock_status", "one_line_insight", "geopolitical_risk", "url"] if c in headlines_df.columns]
        shock_df = headlines_df[headlines_df["shock_status"].isin(["Major Shock", "Shock", "Watch"])]
        if not shock_df.empty:
            push_to_github(repo, "shock_headlines.csv", shock_df[shock_cols].to_csv(index=False), f"Update shocks {timestamp}")

        print("  ✓ GitHub Data Override Complete!")
    except Exception as e:
        print(f"  [!] CRITICAL GITHUB UPLOAD ERROR: {e}")
        traceback.print_exc()

def run_pipeline():
    start_time = datetime.now(IST)
    print(f"\n{'='*60}\nMARKETPULSE PIPELINE — {start_time.strftime('%Y-%m-%d %H:%M IST')}\n{'='*60}")
    try:
        headlines = fetch_news()
        if not headlines: return
        analyzed_headlines = process_all_headlines(headlines) 
        if not analyzed_headlines: return

        df = pd.DataFrame(analyzed_headlines)
        headlines_df, sector_df = calculate_metrics(df)
        msi = calculate_market_stress_index(headlines_df, sector_df)
        
        save_all(headlines_df, sector_df, msi)
        print("  ✅ PIPELINE COMPLETE.\n")
    except Exception as e:
        print(f"  [!] FATAL PIPELINE ERROR: {e}")

if __name__ == "__main__":
    lock_path = "/tmp/pipeline.lock" 
    with open(lock_path, "w") as f: f.write(datetime.now().isoformat())
    try: run_pipeline()
    finally:
        if os.path.exists(lock_path): os.remove(lock_path)
