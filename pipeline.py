import feedparser
import pandas as pd
import numpy as np
import json
import os
import sys
import schedule
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════

RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://economictimes.indiatimes.com/small-biz/startups/rssfeeds/13357270.cms",
    "https://www.livemint.com/rss/markets",
    "https://www.business-standard.com/rss/markets-106.rss",
    "https://www.moneycontrol.com/rss/marketreports.xml",
]

# Sector weights — Python determines systemic importance, not AI
# Higher weight = more impact on Indian economy when things go wrong
SECTOR_WEIGHTS = {
    "Banking":       1.5,   # Systemic — collapses cascade economy-wide
    "Energy":        1.4,   # Input cost for every sector
    "Geopolitics":   1.4,   # Cross-sector contagion
    "Fintech":       1.2,   # Digital payments backbone
    "IT":            1.2,   # Largest export sector
    "Manufacturing": 1.1,   # Employment and exports
    "Healthcare":    1.1,   # Post-COVID elevated importance
    "FMCG":          1.0,   # Baseline consumer demand
    "Startup":       0.9,   # Venture sentiment indicator
    "Retail":        0.8,   # Consumer discretionary
    "Other":         0.7,   # Catch-all
}

VALID_SECTORS    = set(SECTOR_WEIGHTS.keys())
VALID_SENTIMENTS = {"positive", "negative", "neutral"}


# ══════════════════════════════════════════════════════════
# STEP 1 — FETCH NEWS (pure Python, zero AI)
# ══════════════════════════════════════════════════════════

def fetch_news() -> list[dict]:
    """
    Pull headlines from RSS feeds.
    Deduplicates by title. Captures URL for clickable links in frontend.
    No AI involved — pure HTTP fetch and parse.
    """
    print("  Fetching news from RSS feeds...")
    headlines  = []
    seen       = set()

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:8]:
                title       = entry.get("title", "").strip()
                description = entry.get("summary", entry.get("description", "")).strip()
                url         = entry.get("link", entry.get("url", "")).strip()

                # Skip too-short or duplicate titles
                if not title or len(title) < 10:
                    continue
                key = title[:80].lower()
                if key in seen:
                    continue
                seen.add(key)

                headlines.append({
                    "title":       title,
                    "description": description[:600],
                    "source":      feed.feed.get("title", "Unknown"),
                    "published":   entry.get("published", datetime.now().isoformat()),
                    "url":         url,
                })
        except Exception as e:
            print(f"    Feed error ({feed_url[:45]}...): {e}")

    print(f"  Fetched {len(headlines)} unique headlines from {len(RSS_FEEDS)} sources")
    return headlines


# ══════════════════════════════════════════════════════════
# STEP 2 — AI CLASSIFICATION (only AI role in entire pipeline)
# AI reads headlines and returns labels.
# It does NOT calculate any scores — that is Python's job.
# ══════════════════════════════════════════════════════════

def classify_headline(headline: dict) -> dict:
    """
    Send one headline to AI. Get back structured labels only.
    AI job: classify. Python job: calculate everything else.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial news classifier for Indian markets. "
                        "Return ONLY valid JSON. Be precise and conservative. "
                        "If unsure about a field, use the most neutral/conservative value."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""Classify this headline and return JSON with exactly these keys:

- sector: one of [{", ".join(sorted(VALID_SECTORS))}]
- sentiment: one of [positive, negative, neutral]
- sentiment_confidence: float 0.0 to 1.0 (how certain you are about sentiment)
- impact_score: integer 1 to 10 (significance for Indian equity markets)
- valence: float 0.0 to 1.0 (0 = very negative, 1 = very positive)
- arousal: float 0.0 to 1.0 (0 = routine/calm, 1 = alarming/urgent)
- geopolitical_risk: boolean (true if event could affect India via geo channel)
- affected_companies: list of up to 2 specific Indian company names, [] if none
- one_line_insight: one sharp sentence a fund manager needs to hear right now

Headline: {headline["title"]}
Description: {headline["description"][:300]}""",
                },
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        res = json.loads(response.choices[0].message.content)

        # Python enforces schema — AI output is never trusted blindly
        sector    = res.get("sector", "Other")
        sentiment = res.get("sentiment", "neutral")

        return {
            "sector":               sector if sector in VALID_SECTORS else "Other",
            "sentiment":            sentiment if sentiment in VALID_SENTIMENTS else "neutral",
            "sentiment_confidence": float(np.clip(res.get("sentiment_confidence", 0.7), 0.0, 1.0)),
            "impact_score":         int(np.clip(res.get("impact_score", 5), 1, 10)),
            "valence":              float(np.clip(res.get("valence", 0.5), 0.0, 1.0)),
            "arousal":              float(np.clip(res.get("arousal", 0.5), 0.0, 1.0)),
            "geopolitical_risk":    bool(res.get("geopolitical_risk", False)),
            "affected_companies":   res.get("affected_companies", []),
            "one_line_insight":     str(res.get("one_line_insight", ""))[:300],
        }

    except Exception as e:
        print(f"    Classification error: {e}")
        return {
            "sector": "Other", "sentiment": "neutral",
            "sentiment_confidence": 0.5, "impact_score": 5,
            "valence": 0.5, "arousal": 0.5,
            "geopolitical_risk": False,
            "affected_companies": [], "one_line_insight": "",
        }


# ══════════════════════════════════════════════════════════
# STEP 3 — ALL CALCULATIONS (pure Python, zero AI)
# Every number on the dashboard comes from here.
# ══════════════════════════════════════════════════════════

def calculate_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    # ── Sentiment numeric mapping ─────────────────────────
    # Used in all downstream calculations
    sentiment_num_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["sentiment_num"] = df["sentiment"].map(sentiment_num_map).fillna(0)

    # ── Weighted Risk Score per headline ──────────────────
    # Formula: impact × sentiment_multiplier × sector_weight × geo_bonus
    # sentiment_multiplier: negative = 1.0 (full risk), neutral = 0, positive = -0.5 (reduces risk)
    # geo_bonus: geopolitical headlines get 50% extra weight
    # Normalized to 0-100 scale
    sentiment_risk_map = {"positive": -0.5, "neutral": 0.0, "negative": 1.0}
    df["weighted_risk_score"] = (
        df["impact_score"].astype(float)
        * df["sentiment"].map(sentiment_risk_map).fillna(0)
        * df["sector"].map(SECTOR_WEIGHTS).fillna(1.0)
        * df["geopolitical_risk"].apply(lambda x: 1.5 if bool(x) else 1.0)
    )
    df["weighted_risk_score"] = (df["weighted_risk_score"] * 10).clip(0, 100).round(2)

    # ── Z-Score Shock Detection ───────────────────────────
    # GLOBAL z-score across all headlines (not per-sector)
    # Reason: per-sector std collapses to 0 when sector has <3 headlines,
    # making every z-score 0 and every headline "Normal" — useless.
    # Global std gives meaningful differentiation across all 20-40 headlines.
    global_mean = float(df["impact_score"].astype(float).mean())
    global_std  = float(df["impact_score"].astype(float).std())
    if global_std == 0 or pd.isna(global_std):
        global_std = 1.0

    df["z_score"] = (
        (df["impact_score"].astype(float) - global_mean) / global_std
    ).round(2)

    # Pure Python thresholds — deterministic, same data always gives same result
    df["shock_status"] = df["z_score"].apply(
        lambda z: "Major Shock" if z > 2.0
        else      "Shock"       if z > 1.0
        else      "Watch"       if z > 0.5
        else      "Normal"
    )

    # ── Sector-level Calculations ─────────────────────────
    sector_rows = []

    for sector, group in df.groupby("sector"):
        group = group.copy()
        total = len(group)
        pos   = int((group["sentiment"] == "positive").sum())
        neg   = int((group["sentiment"] == "negative").sum())
        neu   = int((group["sentiment"] == "neutral").sum())

        # Net Sentiment Score (NSS) — adapted from NPS formula
        # (%positive - %negative) × 100 · Range: -100 to +100
        # Simple and fast but treats all headlines as equal weight
        nss = round(((pos / total) - (neg / total)) * 100, 1) if total > 0 else 0.0

        # Impact-Weighted Sentiment (IWS)
        # Severe headlines count more than mild ones
        # A negative impact-9 headline moves the score more than a negative impact-2
        total_impact = float(group["impact_score"].astype(float).sum())
        iws = round(
            (group["sentiment_num"] * group["impact_score"].astype(float)).sum()
            / total_impact * 100, 1
        ) if total_impact > 0 else 0.0

        # Confidence-Weighted Sentiment (CWS)
        # AI's uncertain classifications carry less weight
        # High-confidence labels influence the score more
        conf_weight = float(
            (group["sentiment_confidence"].astype(float) * group["impact_score"].astype(float)).sum()
        )
        cws = round(
            (group["sentiment_num"] * group["sentiment_confidence"].astype(float) * group["impact_score"].astype(float)).sum()
            / conf_weight * 100, 1
        ) if conf_weight > 0 else 0.0

        # Composite Sentiment Index (CSI) — most reliable single number
        # 25% NSS (broad coverage) + 50% IWS (severity-adjusted) + 25% CWS (confidence-adjusted)
        csi = round(nss * 0.25 + iws * 0.50 + cws * 0.25, 1)

        avg_risk   = round(float(group["weighted_risk_score"].mean()), 1)
        avg_impact = round(float(group["impact_score"].astype(float).mean()), 1)

        # Sentiment Divergence — gap between NSS and IWS
        # High divergence: many mild positives masking one severe negative (hidden risk)
        divergence = round(abs(nss - iws), 1)

        sector_rows.append({
            "sector":                        sector,
            "avg_weighted_risk":             avg_risk,
            "sentiment_nss":                 nss,
            "impact_weighted_sentiment":     iws,
            "confidence_weighted_sentiment": cws,
            "composite_sentiment_index":     csi,
            "sentiment_velocity":            0.0,   # filled below from history
            "risk_level":                    "HIGH" if avg_risk >= 50 else "MEDIUM" if avg_risk >= 25 else "LOW",
            "avg_impact":                    avg_impact,
            "total_mentions":                total,
            "positive_count":                pos,
            "negative_count":                neg,
            "neutral_count":                 neu,
            "geopolitical_flags":            int(group["geopolitical_risk"].apply(bool).sum()),
            "benchmark_index":               round(avg_risk * 0.5 + avg_impact * 3 + (100 - nss) * 0.2, 1),
            "sector_weight":                 SECTOR_WEIGHTS.get(sector, 1.0),
            "divergence":                    divergence,
            "divergence_flag":               "High Divergence" if divergence > 30 else "Normal",
            "valence":                       round(float(group["valence"].astype(float).mean()), 2) if "valence" in group.columns else 0.5,
            "arousal":                       round(float(group["arousal"].astype(float).mean()), 2) if "arousal" in group.columns else 0.5,
        })

    sector_df = pd.DataFrame(sector_rows)

    if sector_df.empty:
        return df, sector_df

    # ── BCG Matrix Classification ─────────────────────────
    # Sectors plotted on avg_impact vs avg_weighted_risk
    # Quadrant determined by whether each is above/below median
    # Pure Python logic — no AI involved
    med_impact = sector_df["avg_impact"].median()
    med_risk   = sector_df["avg_weighted_risk"].median()

    def bcg_classify(row):
        hi_impact = row["avg_impact"]         >= med_impact
        hi_risk   = row["avg_weighted_risk"]  >= med_risk
        if hi_impact and hi_risk:     return "Watch Closely"
        elif hi_impact and not hi_risk: return "Opportunity"
        elif not hi_impact and hi_risk: return "Monitor Risk"
        else:                          return "Low Priority"

    sector_df["sector_classification"] = sector_df.apply(bcg_classify, axis=1)

    # ── Sentiment Velocity ────────────────────────────────
    # Rate of change of CSI from previous run
    # Positive velocity = sentiment improving (potential buy signal)
    # Negative velocity = sentiment deteriorating (caution)
    master_path = os.path.join(DATA_DIR, "master_sector_scores.csv")
    if os.path.exists(master_path):
        try:
            history = pd.read_csv(master_path)
            if not history.empty and "composite_sentiment_index" in history.columns:
                last_csi = (
                    history.groupby("sector")["composite_sentiment_index"]
                    .last()
                    .to_dict()
                )
                sector_df["sentiment_velocity"] = sector_df.apply(
                    lambda r: round(
                        r["composite_sentiment_index"]
                        - last_csi.get(r["sector"], r["composite_sentiment_index"]),
                        1,
                    ),
                    axis=1,
                )
        except Exception as e:
            print(f"  Velocity calc error: {e}")

    if "sentiment_velocity" not in sector_df.columns:
        sector_df["sentiment_velocity"] = 0.0

    return df, sector_df


# ══════════════════════════════════════════════════════════
# STEP 4 — SAVE (pure Python)
# ══════════════════════════════════════════════════════════

def save_all(headlines_df: pd.DataFrame, sector_df: pd.DataFrame):
    """Save all output files. API reads from latest_*.csv files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_date  = datetime.now().strftime("%Y-%m-%d")

    # ── Snapshot files (timestamped archive) ─────────────
    headlines_df.to_csv(
        os.path.join(DATA_DIR, f"headlines_analyzed_{timestamp}.csv"), index=False
    )
    sector_df.to_csv(
        os.path.join(DATA_DIR, f"sector_benchmark_{timestamp}.csv"), index=False
    )

    # ── Latest files (API reads these) ───────────────────
    headlines_df.to_csv(os.path.join(DATA_DIR, "latest_headlines.csv"), index=False)
    sector_df.to_csv(os.path.join(DATA_DIR, "latest_sectors.csv"), index=False)

    # ── Shock headlines (API serves separately) ──────────
    shock_cols = [
        "title", "sector", "sentiment", "impact_score",
        "z_score", "shock_status", "one_line_insight",
        "geopolitical_risk", "url", "source",
    ]
    shock_cols_present = [c for c in shock_cols if c in headlines_df.columns]
    shock_df = (
        headlines_df[headlines_df["shock_status"].isin(["Major Shock", "Shock", "Watch"])]
        [shock_cols_present]
        .sort_values("z_score", ascending=False)
    )
    shock_df.to_csv(os.path.join(DATA_DIR, "shock_headlines.csv"), index=False)

    # ── Master historical files (for velocity + trend) ───
    headlines_copy = headlines_df.copy()
    sector_copy    = sector_df.copy()
    headlines_copy["run_date"] = run_date
    sector_copy["run_date"]    = run_date

    for df_to_save, filename in [
        (headlines_copy, "master_headlines.csv"),
        (sector_copy,    "master_sector_scores.csv"),
    ]:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            try:
                existing = pd.read_csv(path)
                combined = pd.concat([existing, df_to_save], ignore_index=True)
                combined.to_csv(path, index=False)
            except Exception as e:
                print(f"  Master append error ({filename}): {e}")
                df_to_save.to_csv(path, index=False)
        else:
            df_to_save.to_csv(path, index=False)

    # ── Trend Analysis (3-day moving average of CSI) ─────
    master_sec_path = os.path.join(DATA_DIR, "master_sector_scores.csv")
    if os.path.exists(master_sec_path):
        try:
            master = pd.read_csv(master_sec_path)
            if "composite_sentiment_index" in master.columns and "run_date" in master.columns:
                trend = (
                    master
                    .groupby(["run_date", "sector"])["composite_sentiment_index"]
                    .mean()
                    .reset_index()
                )
                trend["csi_3day_ma"] = (
                    trend
                    .groupby("sector")["composite_sentiment_index"]
                    .transform(lambda x: x.rolling(3, min_periods=1).mean())
                    .round(2)
                )
                trend.to_csv(
                    os.path.join(DATA_DIR, "sector_trend_analysis.csv"), index=False
                )
        except Exception as e:
            print(f"  Trend calc error: {e}")


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════

def run_pipeline():
    print(f"\n{'='*55}")
    print(f"MARKETPULSE PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")

    # ── Step 1: Fetch ─────────────────────────────────────
    headlines = fetch_news()
    if not headlines:
        print("  No headlines fetched. Aborting run.")
        return

    # ── Step 2: AI classifies each headline ──────────────
    print(f"\n  Classifying {len(headlines)} headlines with AI...")
    analyzed = []
    for i, h in enumerate(headlines):
        labels = classify_headline(h)
        analyzed.append({**h, **labels})
        print(f"    [{i+1:2d}/{len(headlines)}] [{labels['sector']:12}] [{labels['sentiment']:8}] {h['title'][:55]}...")

    # ── Step 3: Python calculates all metrics ────────────
    print("\n  Calculating all metrics (Python only)...")
    df = pd.DataFrame(analyzed)
    headlines_df, sector_df = calculate_metrics(df)

    # ── Step 4: Save all files ───────────────────────────
    print("  Saving all files...")
    save_all(headlines_df, sector_df)

    # ── Summary ──────────────────────────────────────────
    major_shocks = int((headlines_df["shock_status"] == "Major Shock").sum())
    shocks       = int((headlines_df["shock_status"] == "Shock").sum())
    watches      = int((headlines_df["shock_status"] == "Watch").sum())
    geo_flags    = int(headlines_df["geopolitical_risk"].apply(bool).sum())

    avg_nss  = float(sector_df["sentiment_nss"].mean())
    avg_risk = float(sector_df["avg_weighted_risk"].mean())
    avg_csi  = float(sector_df["composite_sentiment_index"].mean())

    if avg_csi > 20 and avg_risk < 25:
        regime = "Risk On"
    elif avg_csi < -20 and avg_risk > 45:
        regime = "Panic"
    elif avg_csi > 0 and avg_risk > 35:
        regime = "Complacent"
    else:
        regime = "Risk Off"

    print(f"\n{'─'*55}")
    print(f"  Headlines   : {len(headlines_df)}")
    print(f"  Sectors     : {len(sector_df)}")
    print(f"  Geo flags   : {geo_flags}")
    print(f"  Shocks      : {major_shocks} major · {shocks} shock · {watches} watch")
    print(f"  Avg NSS     : {avg_nss:+.1f}")
    print(f"  Avg CSI     : {avg_csi:+.1f}")
    print(f"  Avg Risk    : {avg_risk:.1f}")
    print(f"  Regime      : {regime}")
    print(f"\n  SECTOR SNAPSHOT:")
    for _, row in sector_df.sort_values("benchmark_index", ascending=False).iterrows():
        print(
            f"  {row['sector']:14} | Risk {row['avg_weighted_risk']:5.1f} "
            f"| NSS {row['sentiment_nss']:+6.1f} "
            f"| CSI {row['composite_sentiment_index']:+6.1f} "
            f"| Vel {row['sentiment_velocity']:+5.1f} "
            f"| {row['risk_level']:6} | {row['sector_classification']}"
        )
    print(f"{'─'*55}")
    print("  PIPELINE COMPLETE.\n")


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# --once flag: run once and exit (used by API trigger)
# no flag:     run once then schedule every hour
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Write lock file so API can detect pipeline is running
    lock_path = os.path.join(DATA_DIR, "pipeline.lock")
    with open(lock_path, "w") as f:
        f.write(datetime.now().isoformat())

    try:
        run_pipeline()
    finally:
        # Always remove lock even if pipeline errors
        if os.path.exists(lock_path):
            os.remove(lock_path)

    # --once: exit after single run (API-triggered mode)
    if "--once" in sys.argv:
        sys.exit(0)

    # Default: schedule hourly runs
    print("  Scheduling hourly runs. Press Ctrl+C to stop.")
    schedule.every(1).hours.do(run_pipeline)
    while True:
        schedule.run_pending()
        time.sleep(60)
