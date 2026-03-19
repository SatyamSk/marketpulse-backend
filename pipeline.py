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

# ── 15 unique feeds across 8 different sources ────────────
# Multiple feeds per source = different topic coverage
# If one feed goes down, others from same source still work
RSS_FEEDS = [
    # Economic Times — Markets, Economy, Tech, Startups
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://economictimes.indiatimes.com/economy/rssfeeds/1373380680.cms",
    "https://economictimes.indiatimes.com/tech/rssfeeds/13357263.cms",
    "https://economictimes.indiatimes.com/small-biz/startups/rssfeeds/13357270.cms",

    # Livemint — Markets, Companies, Economy
    "https://www.livemint.com/rss/markets",
    "https://www.livemint.com/rss/companies",
    "https://www.livemint.com/rss/economy",

    # Business Standard — Markets, Economy, Finance
    "https://www.business-standard.com/rss/markets-106.rss",
    "https://www.business-standard.com/rss/economy-policy-101.rss",
    "https://www.business-standard.com/rss/finance-103.rss",

    # Moneycontrol — Market Reports + Latest News
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.moneycontrol.com/rss/latestnews.xml",

    # Financial Express — Markets
    "https://www.financialexpress.com/market/feed/",

    # Inc42 — Indian Startups + Fintech
    "https://inc42.com/feed/",

    # Reuters India — Business News
    "https://feeds.reuters.com/reuters/INbusinessNews",
]

# Geopolitics removed — it is an event type, not a market sector
# Geopolitical headlines are flagged via geopolitical_risk=True
# and get a 1.5x risk multiplier regardless of their sector
SECTOR_WEIGHTS = {
    "Banking":       1.5,
    "Energy":        1.4,
    "IT":            1.2,
    "Fintech":       1.2,
    "Manufacturing": 1.1,
    "Healthcare":    1.1,
    "FMCG":          1.0,
    "Startup":       0.9,
    "Retail":        0.8,
    "Other":         0.7,
}

VALID_SECTORS    = set(SECTOR_WEIGHTS.keys())
VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def get_max_per_feed() -> int:
    for arg in sys.argv:
        if arg.startswith("--max-per-feed="):
            try:
                return max(3, min(20, int(arg.split("=")[1])))
            except ValueError:
                pass
    return 8


# ══════════════════════════════════════════════════════════
# STEP 1 — FETCH (pure Python)
# ══════════════════════════════════════════════════════════

def fetch_news() -> list[dict]:
    max_per_feed = get_max_per_feed()
    print(f"  Fetching — {max_per_feed} per feed · {len(RSS_FEEDS)} sources · "
          f"~{max_per_feed * len(RSS_FEEDS)} possible...")

    headlines  = []
    seen       = set()
    feed_stats = []

    for feed_url in RSS_FEEDS:
        try:
            feed  = feedparser.parse(feed_url)
            count = 0

            # Skip if feed returned nothing useful
            if not feed.entries:
                print(f"    EMPTY   {feed_url[:60]}")
                feed_stats.append((feed_url[:50], 0, "empty"))
                continue

            for entry in feed.entries:
                if count >= max_per_feed:
                    break

                title       = entry.get("title", "").strip()
                description = entry.get("summary", entry.get("description", "")).strip()
                url         = entry.get("link", entry.get("url", "")).strip()

                if not title or len(title) < 10:
                    continue

                # Deduplicate on exact title
                if title in seen:
                    continue

                seen.add(title)
                headlines.append({
                    "title":       title,
                    "description": description[:600],
                    "source":      feed.feed.get("title", "Unknown"),
                    "published":   entry.get("published", datetime.now().isoformat()),
                    "url":         url,
                })
                count += 1

            source_name = feed.feed.get("title", feed_url[:40])
            print(f"    OK      {source_name[:45]:45} → {count} headlines")
            feed_stats.append((source_name[:45], count, "ok"))

        except Exception as e:
            print(f"    FAILED  {feed_url[:60]} — {e}")
            feed_stats.append((feed_url[:50], 0, f"error: {e}"))

    # Summary
    ok_feeds    = len([f for f in feed_stats if f[2] == "ok"])
    failed_feeds = len([f for f in feed_stats if f[2] != "ok" and f[2] != "empty"])
    empty_feeds  = len([f for f in feed_stats if f[2] == "empty"])

    print(f"\n  ── Feed Summary ──────────────────────────────")
    print(f"  OK: {ok_feeds} · Empty: {empty_feeds} · Failed: {failed_feeds}")
    print(f"  Total unique headlines fetched: {len(headlines)}")
    print(f"  ─────────────────────────────────────────────")

    return headlines


# ══════════════════════════════════════════════════════════
# STEP 2 — AI CLASSIFICATION (only AI role)
# AI classifies sector, sentiment, impact etc.
# Everything else is Python.
# ══════════════════════════════════════════════════════════

def classify_headline(headline: dict) -> dict:
    sectors_list = ", ".join(sorted(VALID_SECTORS))
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial news classifier for Indian equity markets. "
                        "Return ONLY valid JSON. Be precise. "
                        "Geopolitics is NOT a sector — classify geopolitical headlines "
                        "under the actual market sector most affected "
                        "(e.g. Energy for oil/Hormuz, Banking for sanctions, IT for chip shortage). "
                        "Set geopolitical_risk=true for any headline involving international events."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""Classify this headline and return JSON with exactly these keys:

- sector: one of [{sectors_list}]
- sentiment: one of [positive, negative, neutral]
- sentiment_confidence: float 0.0 to 1.0
- impact_score: integer 1 to 10 (significance for Indian equity markets)
- valence: float 0.0 to 1.0 (0=very negative, 1=very positive)
- arousal: float 0.0 to 1.0 (0=calm/routine, 1=alarming/urgent)
- geopolitical_risk: boolean (true if involves international/geopolitical events)
- affected_companies: list of up to 2 specific Indian company names, [] if none
- one_line_insight: one sharp sentence a fund manager needs right now

Headline: {headline["title"]}
Description: {headline["description"][:300]}""",
                },
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        res = json.loads(response.choices[0].message.content)

        sector    = res.get("sector", "Other")
        sentiment = res.get("sentiment", "neutral")

        # If AI still returns Geopolitics despite instructions, remap to Other
        if sector == "Geopolitics" or sector not in VALID_SECTORS:
            sector = "Other"

        return {
            "sector":               sector,
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
# STEP 3 — CALCULATIONS (pure Python, zero AI)
# ══════════════════════════════════════════════════════════

def calculate_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    # Sentiment numeric
    df["sentiment_num"] = df["sentiment"].map(
        {"positive": 1, "neutral": 0, "negative": -1}
    ).fillna(0)

    # Weighted Risk Score per headline
    sentiment_risk_map = {"positive": -0.5, "neutral": 0.0, "negative": 1.0}
    df["weighted_risk_score"] = (
        df["impact_score"].astype(float)
        * df["sentiment"].map(sentiment_risk_map).fillna(0)
        * df["sector"].map(SECTOR_WEIGHTS).fillna(1.0)
        * df["geopolitical_risk"].apply(lambda x: 1.5 if bool(x) else 1.0)
    )
    df["weighted_risk_score"] = (df["weighted_risk_score"] * 10).clip(0, 100).round(2)

    # Z-Score — GLOBAL across all headlines
    # Per-sector std collapses to 0 with few headlines per sector
    global_mean = float(df["impact_score"].astype(float).mean())
    global_std  = float(df["impact_score"].astype(float).std())
    if global_std == 0 or pd.isna(global_std):
        global_std = 1.0

    df["z_score"] = (
        (df["impact_score"].astype(float) - global_mean) / global_std
    ).round(2)

    df["shock_status"] = df["z_score"].apply(
        lambda z: "Major Shock" if z > 2.0
        else      "Shock"       if z > 1.0
        else      "Watch"       if z > 0.5
        else      "Normal"
    )

    # Sector-level calculations
    sector_rows = []
    for sector, group in df.groupby("sector"):
        group = group.copy()
        total = len(group)
        pos   = int((group["sentiment"] == "positive").sum())
        neg   = int((group["sentiment"] == "negative").sum())
        neu   = int((group["sentiment"] == "neutral").sum())

        # NSS — Net Sentiment Score (adapted from NPS)
        nss = round(((pos / total) - (neg / total)) * 100, 1) if total > 0 else 0.0

        # Impact-Weighted Sentiment
        total_impact = float(group["impact_score"].astype(float).sum())
        iws = round(
            (group["sentiment_num"] * group["impact_score"].astype(float)).sum()
            / total_impact * 100, 1
        ) if total_impact > 0 else 0.0

        # Confidence-Weighted Sentiment
        conf_weight = float(
            (group["sentiment_confidence"].astype(float)
             * group["impact_score"].astype(float)).sum()
        )
        cws = round(
            (group["sentiment_num"]
             * group["sentiment_confidence"].astype(float)
             * group["impact_score"].astype(float)).sum()
            / conf_weight * 100, 1
        ) if conf_weight > 0 else 0.0

        # Composite Sentiment Index
        csi = round(nss * 0.25 + iws * 0.50 + cws * 0.25, 1)

        avg_risk   = round(float(group["weighted_risk_score"].mean()), 1)
        avg_impact = round(float(group["impact_score"].astype(float).mean()), 1)
        divergence = round(abs(nss - iws), 1)

        sector_rows.append({
            "sector":                        sector,
            "avg_weighted_risk":             avg_risk,
            "sentiment_nss":                 nss,
            "impact_weighted_sentiment":     iws,
            "confidence_weighted_sentiment": cws,
            "composite_sentiment_index":     csi,
            "sentiment_velocity":            0.0,
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

    # BCG Classification
    med_impact = sector_df["avg_impact"].median()
    med_risk   = sector_df["avg_weighted_risk"].median()

    def bcg_classify(row: pd.Series) -> str:
        hi_i = row["avg_impact"]        >= med_impact
        hi_r = row["avg_weighted_risk"] >= med_risk
        if hi_i and hi_r:      return "Watch Closely"
        elif hi_i and not hi_r: return "Opportunity"
        elif not hi_i and hi_r: return "Monitor Risk"
        else:                   return "Low Priority"

    sector_df["sector_classification"] = sector_df.apply(bcg_classify, axis=1)

    # Velocity from history
    master_path = os.path.join(DATA_DIR, "master_sector_scores.csv")
    if os.path.exists(master_path):
        try:
            history = pd.read_csv(master_path)
            if not history.empty and "composite_sentiment_index" in history.columns:
                last_csi = (
                    history.groupby("sector")["composite_sentiment_index"]
                    .last().to_dict()
                )
                sector_df["sentiment_velocity"] = sector_df.apply(
                    lambda r: round(
                        r["composite_sentiment_index"]
                        - last_csi.get(r["sector"], r["composite_sentiment_index"]),
                        1,
                    ), axis=1,
                )
        except Exception as e:
            print(f"  Velocity error: {e}")

    if "sentiment_velocity" not in sector_df.columns:
        sector_df["sentiment_velocity"] = 0.0

    return df, sector_df


# ══════════════════════════════════════════════════════════
# STEP 4 — SAVE (pure Python)
# ══════════════════════════════════════════════════════════

def save_all(headlines_df: pd.DataFrame, sector_df: pd.DataFrame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_date  = datetime.now().strftime("%Y-%m-%d")

    # Timestamped snapshots
    headlines_df.to_csv(
        os.path.join(DATA_DIR, f"headlines_analyzed_{timestamp}.csv"), index=False
    )
    sector_df.to_csv(
        os.path.join(DATA_DIR, f"sector_benchmark_{timestamp}.csv"), index=False
    )

    # Latest files — API reads these
    headlines_df.to_csv(os.path.join(DATA_DIR, "latest_headlines.csv"), index=False)
    sector_df.to_csv(os.path.join(DATA_DIR, "latest_sectors.csv"), index=False)

    # Shock summary
    shock_cols = [c for c in [
        "title", "sector", "sentiment", "impact_score",
        "z_score", "shock_status", "one_line_insight",
        "geopolitical_risk", "url", "source",
    ] if c in headlines_df.columns]

    headlines_df[
        headlines_df["shock_status"].isin(["Major Shock", "Shock", "Watch"])
    ][shock_cols].sort_values("z_score", ascending=False).to_csv(
        os.path.join(DATA_DIR, "shock_headlines.csv"), index=False
    )

    # Master historical files
    headlines_copy = headlines_df.copy()
    sector_copy    = sector_df.copy()
    headlines_copy["run_date"] = run_date
    sector_copy["run_date"]    = run_date

    for df_save, filename in [
        (headlines_copy, "master_headlines.csv"),
        (sector_copy,    "master_sector_scores.csv"),
    ]:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            try:
                combined = pd.concat(
                    [pd.read_csv(path), df_save], ignore_index=True
                )
                combined.to_csv(path, index=False)
            except Exception as e:
                print(f"  Master append error ({filename}): {e}")
                df_save.to_csv(path, index=False)
        else:
            df_save.to_csv(path, index=False)

    # Trend analysis
    master_path = os.path.join(DATA_DIR, "master_sector_scores.csv")
    if os.path.exists(master_path):
        try:
            master = pd.read_csv(master_path)
            if "composite_sentiment_index" in master.columns and "run_date" in master.columns:
                trend = (
                    master
                    .groupby(["run_date", "sector"])["composite_sentiment_index"]
                    .mean().reset_index()
                )
                trend["csi_3day_ma"] = (
                    trend.groupby("sector")["composite_sentiment_index"]
                    .transform(lambda x: x.rolling(3, min_periods=1).mean())
                    .round(2)
                )
                trend.to_csv(
                    os.path.join(DATA_DIR, "sector_trend_analysis.csv"), index=False
                )
        except Exception as e:
            print(f"  Trend error: {e}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def run_pipeline():
    print(f"\n{'='*55}")
    print(f"MARKETPULSE PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")

    headlines = fetch_news()
    if not headlines:
        print("  No headlines fetched. Aborting.")
        return

    print(f"\n  Classifying {len(headlines)} headlines...")
    analyzed = []
    for i, h in enumerate(headlines):
        labels = classify_headline(h)
        analyzed.append({**h, **labels})
        print(
            f"    [{i+1:2d}/{len(headlines)}] "
            f"[{labels['sector']:12}] "
            f"[geo:{str(labels['geopolitical_risk'])[0]}] "
            f"[{labels['sentiment']:8}] "
            f"{h['title'][:50]}..."
        )

    print("\n  Calculating metrics...")
    df = pd.DataFrame(analyzed)
    headlines_df, sector_df = calculate_metrics(df)

    print("  Saving files...")
    save_all(headlines_df, sector_df)

    # Summary
    major  = int((headlines_df["shock_status"] == "Major Shock").sum())
    shocks = int((headlines_df["shock_status"] == "Shock").sum())
    geo    = int(headlines_df["geopolitical_risk"].apply(bool).sum())
    avg_nss  = float(sector_df["sentiment_nss"].mean())
    avg_csi  = float(sector_df["composite_sentiment_index"].mean())
    avg_risk = float(sector_df["avg_weighted_risk"].mean())

    if avg_csi > 20 and avg_risk < 20:    regime = "Risk On"
    elif avg_csi < -20 and avg_risk > 35: regime = "Panic"
    elif avg_csi > 0 and avg_risk > 25:   regime = "Complacent"
    else:                                  regime = "Risk Off"

    print(f"\n{'─'*55}")
    print(f"  Headlines  : {len(headlines_df)}")
    print(f"  Sectors    : {len(sector_df)}")
    print(f"  Geo flags  : {geo}")
    print(f"  Shocks     : {major} major · {shocks} shock")
    print(f"  Avg NSS    : {avg_nss:+.1f}")
    print(f"  Avg CSI    : {avg_csi:+.1f}")
    print(f"  Avg Risk   : {avg_risk:.1f}")
    print(f"  Regime     : {regime}")
    print(f"\n  SECTOR SNAPSHOT:")
    for _, row in sector_df.sort_values("benchmark_index", ascending=False).iterrows():
        print(
            f"  {row['sector']:14} "
            f"| Risk {row['avg_weighted_risk']:5.1f} "
            f"| NSS {row['sentiment_nss']:+6.1f} "
            f"| CSI {row['composite_sentiment_index']:+6.1f} "
            f"| Vel {row['sentiment_velocity']:+5.1f} "
            f"| {row['risk_level']:6} "
            f"| {row['sector_classification']}"
        )
    print(f"{'─'*55}")
    print("  PIPELINE COMPLETE.\n")


if __name__ == "__main__":
    lock_path = os.path.join(DATA_DIR, "pipeline.lock")
    with open(lock_path, "w") as f:
        f.write(datetime.now().isoformat())

    try:
        run_pipeline()
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)

    if "--once" in sys.argv:
        sys.exit(0)

    print("  Scheduling hourly runs. Ctrl+C to stop.")
    schedule.every(1).hours.do(run_pipeline)
    while True:
        schedule.run_pending()
        time.sleep(60)
