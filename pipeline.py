import feedparser
import pandas as pd
import numpy as np
import json
import os
import sys
import hashlib
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv
from dateutil import parser as dateparser

load_dotenv()

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════

RSS_FEEDS = [
    # ── Economic Times ────────────────────────────────────
    ("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",      "ET Markets"),
    ("https://economictimes.indiatimes.com/economy/rssfeeds/1373380680.cms",      "ET Economy"),
    ("https://economictimes.indiatimes.com/tech/rssfeeds/13357263.cms",           "ET Tech"),
    ("https://economictimes.indiatimes.com/small-biz/startups/rssfeeds/13357270.cms","ET Startups"),
    ("https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms",       "ET Industry"),

    # ── Livemint ──────────────────────────────────────────
    ("https://www.livemint.com/rss/markets",   "Livemint Markets"),
    ("https://www.livemint.com/rss/companies", "Livemint Companies"),
    ("https://www.livemint.com/rss/economy",   "Livemint Economy"),
    ("https://www.livemint.com/rss/politics",  "Livemint Politics"),

    # ── Business Standard ─────────────────────────────────
    ("https://www.business-standard.com/rss/markets-106.rss",         "BS Markets"),
    ("https://www.business-standard.com/rss/economy-policy-101.rss",  "BS Economy"),
    ("https://www.business-standard.com/rss/finance-103.rss",         "BS Finance"),
    ("https://www.business-standard.com/rss/companies-101.rss",       "BS Companies"),

    # ── Moneycontrol ──────────────────────────────────────
    ("https://www.moneycontrol.com/rss/marketreports.xml", "MC Market Reports"),
    ("https://www.moneycontrol.com/rss/latestnews.xml",    "MC Latest News"),
    ("https://www.moneycontrol.com/rss/business.xml",      "MC Business"),

    # ── Financial Express ─────────────────────────────────
    ("https://www.financialexpress.com/market/feed/",  "Financial Express Markets"),
    ("https://www.financialexpress.com/economy/feed/", "Financial Express Economy"),

    # ── Hindu BusinessLine ────────────────────────────────
    ("https://www.thehindubusinessline.com/markets/?service=rss",  "BL Markets"),
    ("https://www.thehindubusinessline.com/economy/?service=rss",  "BL Economy"),
    ("https://www.thehindubusinessline.com/companies/?service=rss","BL Companies"),

    # ── Government & Regulatory (PIB, RBI, SEBI) ─────────
    ("https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",   "PIB Economy"),
    ("https://pib.gov.in/RssMain.aspx?ModId=37&Lang=1&Regid=3",  "PIB Commerce"),
    ("https://pib.gov.in/RssMain.aspx?ModId=25&Lang=1&Regid=3",  "PIB Finance"),
    ("https://pib.gov.in/RssMain.aspx?ModId=3&Lang=1&Regid=3",   "PIB Infrastructure"),
    ("https://pib.gov.in/RssMain.aspx?ModId=14&Lang=1&Regid=3",  "PIB Defence"),
    ("https://www.rbi.org.in/scripts/rss.aspx",                   "RBI"),
    ("https://www.sebi.gov.in/sebi_data/rss/sebi_news.xml",       "SEBI"),

    # ── Startups & Fintech ────────────────────────────────
    ("https://inc42.com/feed/",              "Inc42"),
    ("https://entrackr.com/feed/",           "Entrackr"),
    ("https://yourstory.com/feed",           "YourStory"),

    # ── Sector Specific ───────────────────────────────────
    ("https://mercomindia.com/feed/",        "Mercom India — Solar/Energy"),
    ("https://www.constructionworld.in/feed","Construction World — Infra"),
    ("https://www.pharmacybiz.net/feed/",    "PharmaBiz — Healthcare"),
    ("https://www.cio.in/rss.xml",           "CIO India — IT/Tech"),

    # ── Global Impact on India ────────────────────────────
    ("https://feeds.reuters.com/reuters/INbusinessNews", "Reuters India"),
    ("https://feeds.bbci.co.uk/news/business/rss.xml",  "BBC Business"),
]

# Sector weights — Banking most systemic, Other least
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

# Catalyst importance weights — government actions most reliable
CATALYST_WEIGHTS = {
    "government_contract":   1.7,
    "policy_change":         1.6,
    "pib_announcement":      1.6,
    "rbi_action":            1.5,
    "sebi_action":           1.5,
    "fii_flow":              1.4,
    "capex_announcement":    1.4,
    "earnings":              1.3,
    "sector_tailwind":       1.2,
    "regulatory":            1.1,
    "management_change":     1.0,
    "global_event":          0.9,
    "other":                 0.7,
}

# Signal half-life in hours — how fast signal decays
SIGNAL_HALF_LIFE = {
    "intraday":         4,
    "swing_2_5days":    48,
    "positional_weeks": 168,
}

VALID_SECTORS    = set(SECTOR_WEIGHTS.keys())
VALID_SENTIMENTS = {"positive", "negative", "neutral"}


# ══════════════════════════════════════════════════════════
# STEP 1 — FETCH (pure Python)
# ══════════════════════════════════════════════════════════

def get_max_per_feed() -> int:
    for arg in sys.argv:
        if arg.startswith("--max-per-feed="):
            try:
                return max(3, min(100, int(arg.split("=")[1]))) # Increased to 100
            except ValueError:
                pass
    return 25   # Default 25 × 37 feeds ≈ 925 headlines

def parse_publish_time(entry: dict) -> datetime:
    """Parse RSS publish time — fallback to now if unparseable."""
    for field in ["published", "updated", "created"]:
        val = entry.get(field)
        if val:
            try:
                dt = dateparser.parse(val)
                if dt:
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
            except Exception:
                pass
    return datetime.now(timezone.utc)

def fetch_news() -> list[dict]:
    max_per_feed = get_max_per_feed()
    total_possible = max_per_feed * len(RSS_FEEDS)
    print(f"  Fetching — {max_per_feed} per feed · {len(RSS_FEEDS)} sources · "
          f"~{total_possible} possible...")

    headlines  = []
    seen       = set()
    ok_count   = 0
    fail_count = 0

    for feed_url, feed_label in RSS_FEEDS:
        try:
            feed  = feedparser.parse(feed_url)
            count = 0

            if not feed.entries:
                print(f"    EMPTY   {feed_label}")
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

                # Content hash for near-duplicate detection
                content_hash = hashlib.md5(
                    title[:60].lower().encode()
                ).hexdigest()[:8]
                if content_hash in seen:
                    continue

                seen.add(title)
                seen.add(content_hash)

                publish_dt = parse_publish_time(entry)
                hours_old  = max(0, (
                    datetime.now(timezone.utc) - publish_dt
                ).total_seconds() / 3600)

                headlines.append({
                    "title":       title,
                    "description": description[:800],
                    "source":      feed_label,
                    "source_url":  feed_url,
                    "published":   publish_dt.isoformat(),
                    "hours_old":   round(hours_old, 1),
                    "url":         url,
                    # Tag govt/regulatory sources for extra weight
                    "is_govt_source": any(
                        x in feed_label for x in ["PIB", "RBI", "SEBI"]
                    ),
                })
                count += 1

            print(f"    OK   {feed_label:35} → {count}")
            ok_count += 1

        except Exception as e:
            print(f"    FAIL {feed_label:35} → {e}")
            fail_count += 1

    print(f"\n  Sources: {ok_count} OK · {fail_count} failed")
    print(f"  Total unique headlines: {len(headlines)}")
    return headlines


# ══════════════════════════════════════════════════════════
# STEP 2 — AI CLASSIFICATION (only AI role)
# ══════════════════════════════════════════════════════════

def classify_headline(headline: dict) -> dict:
    sectors_list = ", ".join(sorted(VALID_SECTORS))
    is_govt = headline.get("is_govt_source", False)

    # Extra instruction for govt sources
    govt_note = (
        "This is an OFFICIAL GOVERNMENT/REGULATORY source (PIB/RBI/SEBI). "
        "These announcements are high-conviction catalysts. "
        "Give impact_score >= 7 unless clearly routine. "
    ) if is_govt else ""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior equity research analyst at a top Indian brokerage. "
                        "You find NON-OBVIOUS opportunities — not just headline stocks but "
                        "second and third order beneficiaries. "
                        "Example: a highway contract benefits the contractor "
                        "BUT ALSO cement, steel, equipment, logistics companies. "
                        "A PIB defence announcement benefits defence PSUs AND "
                        "their component suppliers AND logistics firms. "
                        "Think in supply chains. Think in ecosystems. "
                        f"{govt_note}"
                        "Geopolitics is NOT a sector — use the actual affected market sector. "
                        "Return ONLY valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""Analyze this headline deeply and return JSON with these exact keys:

- sector: one of [{sectors_list}]
- sentiment: one of [positive, negative, neutral]
- sentiment_confidence: float 0.0-1.0
- impact_score: integer 1-10 (market significance for Indian equities)
- valence: float 0.0-1.0 (0=very negative, 1=very positive)
- arousal: float 0.0-1.0 (0=calm/routine, 1=alarming/urgent)
- geopolitical_risk: boolean
- affected_companies: list of up to 4 specific Indian companies directly affected
- second_order_beneficiaries: list of up to 4 companies indirectly benefiting
- catalyst_type: one of [{", ".join(CATALYST_WEIGHTS.keys())}]
- price_direction: one of [bullish, bearish, neutral]
- time_horizon: one of [intraday, swing_2_5days, positional_weeks]
- conviction: one of [high, medium, low]
- macro_sensitivity: one of [high, medium, low]
  (how sensitive is this to macro conditions like rates, crude, dollar)
- one_line_insight: sharp one sentence for a fund manager
- signal_reason: one sentence explaining the price direction call
- contrarian_flag: boolean
  (true if this news seems bullish/bearish but the REAL implication is opposite)
- contrarian_reason: string, explain if contrarian_flag is true, else empty string

Headline: {headline["title"]}
Description: {headline["description"][:400]}""",
                },
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        res = json.loads(response.choices[0].message.content)

        sector = res.get("sector", "Other")
        if sector not in VALID_SECTORS:
            sector = "Other"

        return {
            "sector":                   sector,
            "sentiment":                res.get("sentiment", "neutral") if res.get("sentiment") in VALID_SENTIMENTS else "neutral",
            "sentiment_confidence":     float(np.clip(res.get("sentiment_confidence", 0.7), 0.0, 1.0)),
            "impact_score":             int(np.clip(res.get("impact_score", 5), 1, 10)),
            "valence":                  float(np.clip(res.get("valence", 0.5), 0.0, 1.0)),
            "arousal":                  float(np.clip(res.get("arousal", 0.5), 0.0, 1.0)),
            "geopolitical_risk":        bool(res.get("geopolitical_risk", False)),
            "affected_companies":       res.get("affected_companies", []),
            "second_order_beneficiaries": res.get("second_order_beneficiaries", []),
            "catalyst_type":            res.get("catalyst_type", "other"),
            "price_direction":          res.get("price_direction", "neutral"),
            "time_horizon":             res.get("time_horizon", "intraday"),
            "conviction":               res.get("conviction", "low"),
            "macro_sensitivity":        res.get("macro_sensitivity", "medium"),
            "one_line_insight":         str(res.get("one_line_insight", ""))[:300],
            "signal_reason":            str(res.get("signal_reason", ""))[:300],
            "contrarian_flag":          bool(res.get("contrarian_flag", False)),
            "contrarian_reason":        str(res.get("contrarian_reason", ""))[:300],
        }

    except Exception as e:
        print(f"    Classification error: {e}")
        return {
            "sector": "Other", "sentiment": "neutral",
            "sentiment_confidence": 0.5, "impact_score": 5,
            "valence": 0.5, "arousal": 0.5, "geopolitical_risk": False,
            "affected_companies": [], "second_order_beneficiaries": [],
            "catalyst_type": "other", "price_direction": "neutral",
            "time_horizon": "intraday", "conviction": "low",
            "macro_sensitivity": "medium",
            "one_line_insight": "", "signal_reason": "",
            "contrarian_flag": False, "contrarian_reason": "",
        }


# ══════════════════════════════════════════════════════════
# STEP 3 — ALL CALCULATIONS (pure Python, zero AI)
# ══════════════════════════════════════════════════════════

def calculate_signal_decay(hours_old: float, time_horizon: str) -> float:
    """
    Signal decays over time. A 4-hour-old intraday signal
    is much weaker than a fresh one. Positional signals decay slower.
    Returns 0.0 (fully decayed) to 1.0 (fresh).
    """
    half_life = SIGNAL_HALF_LIFE.get(time_horizon, 4)
    return float(np.exp(-0.693 * hours_old / half_life))

def calculate_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    # ── Numeric mappings ──────────────────────────────────
    df["sentiment_num"] = df["sentiment"].map(
        {"positive": 1, "neutral": 0, "negative": -1}
    ).fillna(0)

    # ── Weighted Risk Score ───────────────────────────────
    sentiment_risk_map = {"positive": -0.5, "neutral": 0.0, "negative": 1.0}
    df["weighted_risk_score"] = (
        df["impact_score"].astype(float)
        * df["sentiment"].map(sentiment_risk_map).fillna(0)
        * df["sector"].map(SECTOR_WEIGHTS).fillna(1.0)
        * df["geopolitical_risk"].apply(lambda x: 1.5 if bool(x) else 1.0)
        * df["is_govt_source"].apply(lambda x: 1.3 if bool(x) else 1.0)
    )
    df["weighted_risk_score"] = (df["weighted_risk_score"] * 10).clip(0, 100).round(2)

    # ── Signal Decay ──────────────────────────────────────
    df["signal_decay"] = df.apply(
        lambda r: calculate_signal_decay(
            float(r.get("hours_old", 0)),
            str(r.get("time_horizon", "intraday"))
        ), axis=1
    ).round(3)

    # ── Recency-Weighted Impact ───────────────────────────
    # Fresh news counts more than stale news
    df["recency_weighted_impact"] = (
        df["impact_score"].astype(float) * df["signal_decay"]
    ).round(2)

    # ── Catalyst Weight ───────────────────────────────────
    df["catalyst_weight"] = df["catalyst_type"].map(CATALYST_WEIGHTS).fillna(0.7)

    # ── Z-Score — GLOBAL across all headlines ────────────
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

    # ── Sector-level Calculations ─────────────────────────
    sector_rows = []

    for sector, group in df.groupby("sector"):
        group = group.copy()
        total = len(group)
        pos   = int((group["sentiment"] == "positive").sum())
        neg   = int((group["sentiment"] == "negative").sum())
        neu   = int((group["sentiment"] == "neutral").sum())

        # NSS
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

        # Recency-Weighted Sentiment — newer news counts more
        recency_weight = float(group["recency_weighted_impact"].sum())
        rws = round(
            (group["sentiment_num"] * group["recency_weighted_impact"]).sum()
            / recency_weight * 100, 1
        ) if recency_weight > 0 else 0.0

        # Composite Sentiment Index
        # 20% NSS + 40% IWS + 20% CWS + 20% RWS
        csi = round(nss * 0.20 + iws * 0.40 + cws * 0.20 + rws * 0.20, 1)

        avg_risk   = round(float(group["weighted_risk_score"].mean()), 1)
        avg_impact = round(float(group["impact_score"].astype(float).mean()), 1)
        divergence = round(abs(nss - iws), 1)

        # Catalyst Diversity — are signals from multiple catalyst types?
        unique_catalysts   = group["catalyst_type"].nunique()
        catalyst_diversity = round(min(unique_catalysts / 3, 1.0), 2)

        # Govt signal presence — PIB/RBI/SEBI in this sector?
        govt_signals = int(group.get("is_govt_source", pd.Series([False])).apply(bool).sum())

        # Contrarian signals count
        contrarian_count = int(group.get("contrarian_flag", pd.Series([False])).apply(bool).sum())

        # Macro sensitivity — majority vote
        macro_counts = group.get("macro_sensitivity", pd.Series([])).value_counts()
        macro_sensitivity = macro_counts.index[0] if not macro_counts.empty else "medium"

        # Signal strength — combines conviction, impact, catalyst weight, recency
        conviction_map = {"high": 1.0, "medium": 0.65, "low": 0.35}
        avg_conviction = float(
            group.get("conviction", pd.Series(["low"]))
            .map(conviction_map).fillna(0.35).mean()
        )
        avg_catalyst_weight = float(group["catalyst_weight"].mean())
        avg_decay = float(group["signal_decay"].mean())

        signal_strength = round(
            avg_conviction * avg_catalyst_weight * avg_decay
            * (avg_impact / 10) * 100, 1
        )

        # Momentum score — direction consistency across headlines
        bullish_pct = float((group.get("price_direction", pd.Series([])) == "bullish").mean())
        bearish_pct = float((group.get("price_direction", pd.Series([])) == "bearish").mean())
        momentum_score = round((bullish_pct - bearish_pct) * 100, 1)

        sector_rows.append({
            "sector":                        sector,
            "avg_weighted_risk":             avg_risk,
            "sentiment_nss":                 nss,
            "impact_weighted_sentiment":     iws,
            "confidence_weighted_sentiment": cws,
            "recency_weighted_sentiment":    rws,
            "composite_sentiment_index":     csi,
            "sentiment_velocity":            0.0,
            "risk_level":                    "HIGH" if avg_risk >= 50 else "MEDIUM" if avg_risk >= 25 else "LOW",
            "avg_impact":                    avg_impact,
            "total_mentions":                total,
            "positive_count":                pos,
            "negative_count":                neg,
            "neutral_count":                 neu,
            "geopolitical_flags":            int(group["geopolitical_risk"].apply(bool).sum()),
            "govt_signals":                  govt_signals,
            "contrarian_count":              contrarian_count,
            "catalyst_diversity":            catalyst_diversity,
            "macro_sensitivity":             macro_sensitivity,
            "signal_strength":               signal_strength,
            "momentum_score":                momentum_score,
            "benchmark_index":               round(avg_risk * 0.5 + avg_impact * 3 + (100 - nss) * 0.2, 1),
            "sector_weight":                 SECTOR_WEIGHTS.get(sector, 1.0),
            "divergence":                    divergence,
            "divergence_flag":               "High Divergence" if divergence > 30 else "Normal",
            "valence":                       round(float(group["valence"].astype(float).mean()), 2),
            "arousal":                       round(float(group["arousal"].astype(float).mean()), 2),
        })

    sector_df = pd.DataFrame(sector_rows)

    if sector_df.empty:
        return df, sector_df

    # ── BCG Classification ────────────────────────────────
    med_impact = sector_df["avg_impact"].median()
    med_risk   = sector_df["avg_weighted_risk"].median()

    def bcg_classify(row: pd.Series) -> str:
        hi_i = row["avg_impact"]        >= med_impact
        hi_r = row["avg_weighted_risk"] >= med_risk
        if hi_i and hi_r:       return "Watch Closely"
        elif hi_i and not hi_r: return "Opportunity"
        elif not hi_i and hi_r: return "Monitor Risk"
        else:                   return "Low Priority"

    sector_df["sector_classification"] = sector_df.apply(bcg_classify, axis=1)

    # ── Investment Signal ─────────────────────────────────
    def investment_signal(row: pd.Series) -> str:
        csi  = row["composite_sentiment_index"]
        risk = row["avg_weighted_risk"]
        vel  = row["sentiment_velocity"]
        sig  = row["signal_strength"]
        mom  = row["momentum_score"]
        if csi > 30 and risk < 25 and mom > 20:   return "BUY BIAS"
        if csi < -20 or (risk > 50 and mom < -20): return "AVOID"
        if row["divergence_flag"] == "High Divergence": return "CAUTION"
        if mom > 10 and csi > 0:                   return "IMPROVING"
        if row["contrarian_count"] > 2:            return "CONTRARIAN WATCH"
        return "NEUTRAL"

    sector_df["investment_signal"] = sector_df.apply(investment_signal, axis=1)

    # ── Velocity from history ─────────────────────────────
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


def calculate_market_stress_index(
    headlines_df: pd.DataFrame,
    sector_df: pd.DataFrame
) -> dict:
    """
    Market Stress Index — single number 0-100.
    Combines risk, shock frequency, geo flags, divergence, contrarian signals.
    Pure Python. Deterministic.
    """
    total = len(headlines_df)
    if total == 0:
        return {"msi": 0, "level": "Low", "components": {}}

    # Component 1 — Average weighted risk (0-100 → weight 30%)
    avg_risk = float(sector_df["avg_weighted_risk"].mean()) if not sector_df.empty else 0
    risk_component = min(avg_risk, 100)

    # Component 2 — Shock frequency (% of Major+Shock headlines → weight 25%)
    shock_pct = (
        headlines_df["shock_status"].isin(["Major Shock", "Shock"]).sum()
        / total * 100
    )
    shock_component = min(shock_pct * 3, 100)

    # Component 3 — Geo risk concentration (→ weight 20%)
    geo_pct = headlines_df["geopolitical_risk"].apply(bool).sum() / total * 100
    geo_component = min(geo_pct * 2, 100)

    # Component 4 — Divergence prevalence (→ weight 15%)
    if not sector_df.empty:
        div_count = (sector_df["divergence_flag"] == "High Divergence").sum()
        div_component = min(div_count / len(sector_df) * 100 * 2, 100)
    else:
        div_component = 0

    # Component 5 — Negative sentiment dominance (→ weight 10%)
    neg_pct = (headlines_df["sentiment"] == "negative").sum() / total * 100
    neg_component = min(neg_pct, 100)

    msi = round(
        risk_component  * 0.30
        + shock_component * 0.25
        + geo_component   * 0.20
        + div_component   * 0.15
        + neg_component   * 0.10,
        1
    )

    level = (
        "Critical" if msi >= 75 else
        "High"     if msi >= 50 else
        "Elevated" if msi >= 30 else
        "Low"
    )

    return {
        "msi": msi,
        "level": level,
        "components": {
            "risk_score":       round(risk_component, 1),
            "shock_frequency":  round(shock_component, 1),
            "geo_exposure":     round(geo_component, 1),
            "divergence_risk":  round(div_component, 1),
            "negative_dominance": round(neg_component, 1),
        }
    }


# ══════════════════════════════════════════════════════════
# STEP 4 — SAVE (pure Python)
# ══════════════════════════════════════════════════════════

def save_all(headlines_df: pd.DataFrame, sector_df: pd.DataFrame, msi: dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_date  = datetime.now().strftime("%Y-%m-%d")

    # Latest files — API reads these
    headlines_df.to_csv(os.path.join(DATA_DIR, "latest_headlines.csv"), index=False)
    sector_df.to_csv(os.path.join(DATA_DIR,    "latest_sectors.csv"),   index=False)

    # Market stress index
    import json as _json
    with open(os.path.join(DATA_DIR, "latest_msi.json"), "w") as f:
        _json.dump(msi, f)

    # Shock headlines
    shock_cols = [c for c in [
        "title", "sector", "sentiment", "impact_score", "z_score",
        "shock_status", "one_line_insight", "geopolitical_risk",
        "url", "source", "hours_old", "catalyst_type", "signal_decay",
    ] if c in headlines_df.columns]

    headlines_df[
        headlines_df["shock_status"].isin(["Major Shock", "Shock", "Watch"])
    ][shock_cols].sort_values("z_score", ascending=False).to_csv(
        os.path.join(DATA_DIR, "shock_headlines.csv"), index=False
    )

    # Govt/PIB headlines separately
    if "is_govt_source" in headlines_df.columns:
        govt_cols = [c for c in [
            "title", "sector", "sentiment", "impact_score",
            "catalyst_type", "one_line_insight", "url",
            "source", "hours_old", "affected_companies",
        ] if c in headlines_df.columns]
        headlines_df[
            headlines_df["is_govt_source"].apply(bool)
        ][govt_cols].to_csv(
            os.path.join(DATA_DIR, "govt_headlines.csv"), index=False
        )

    # Timestamped snapshots
    headlines_df.to_csv(
        os.path.join(DATA_DIR, f"headlines_analyzed_{timestamp}.csv"), index=False
    )
    sector_df.to_csv(
        os.path.join(DATA_DIR, f"sector_benchmark_{timestamp}.csv"), index=False
    )

    # Master historical files
    for df_save, filename in [
        (headlines_df.assign(run_date=run_date), "master_headlines.csv"),
        (sector_df.assign(run_date=run_date),    "master_sector_scores.csv"),
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
            if "composite_sentiment_index" in master.columns:
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
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"MARKETPULSE PIPELINE — {start_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    # Step 1 — Fetch
    headlines = fetch_news()
    if not headlines:
        print("  No headlines fetched. Aborting.")
        return

    # Step 2 — AI classifies
    print(f"\n  Classifying {len(headlines)} headlines...")
    analyzed = []
    for i, h in enumerate(headlines):
        labels = classify_headline(h)
        analyzed.append({**h, **labels})
        geo_tag = "🌍" if labels["geopolitical_risk"] else "  "
        govt_tag = "🏛" if h.get("is_govt_source") else "  "
        print(
            f"    [{i+1:3d}/{len(headlines)}] "
            f"{geo_tag}{govt_tag} "
            f"[{labels['sector']:12}] "
            f"[{labels['sentiment']:8}] "
            f"[{labels['catalyst_type']:20}] "
            f"{h['title'][:45]}..."
        )

    # Step 3 — Python calculates everything
    print("\n  Calculating metrics...")
    df = pd.DataFrame(analyzed)
    headlines_df, sector_df = calculate_metrics(df)

    print("  Calculating Market Stress Index...")
    msi = calculate_market_stress_index(headlines_df, sector_df)

    # Step 4 — Save
    print("  Saving all files...")
    save_all(headlines_df, sector_df, msi)

    # Summary
    elapsed     = (datetime.now() - start_time).seconds
    major       = int((headlines_df["shock_status"] == "Major Shock").sum())
    shocks      = int((headlines_df["shock_status"] == "Shock").sum())
    geo         = int(headlines_df["geopolitical_risk"].apply(bool).sum())
    govt        = int(headlines_df.get("is_govt_source", pd.Series([False])).apply(bool).sum())
    contrarian  = int(headlines_df.get("contrarian_flag", pd.Series([False])).apply(bool).sum())
    avg_csi     = float(sector_df["composite_sentiment_index"].mean()) if not sector_df.empty else 0
    avg_risk    = float(sector_df["avg_weighted_risk"].mean()) if not sector_df.empty else 0

    if avg_csi > 20 and avg_risk < 20:    regime = "Risk On"
    elif avg_csi < -20 and avg_risk > 35: regime = "Panic"
    elif avg_csi > 0 and avg_risk > 25:   regime = "Complacent"
    else:                                  regime = "Risk Off"

    print(f"\n{'─'*60}")
    print(f"  Headlines    : {len(headlines_df)}")
    print(f"  Sectors      : {len(sector_df)}")
    print(f"  Geo flags    : {geo}")
    print(f"  Govt/PIB     : {govt}")
    print(f"  Contrarian   : {contrarian}")
    print(f"  Shocks       : {major} major · {shocks} shock")
    print(f"  Avg CSI      : {avg_csi:+.1f}")
    print(f"  Avg Risk     : {avg_risk:.1f}")
    print(f"  MSI          : {msi['msi']} ({msi['level']})")
    print(f"  Regime       : {regime}")
    print(f"  Time taken   : {elapsed}s")
    print(f"\n  SECTOR SNAPSHOT:")
    for _, row in sector_df.sort_values("benchmark_index", ascending=False).iterrows():
        print(
            f"  {row['sector']:14} "
            f"| Risk {row['avg_weighted_risk']:5.1f} "
            f"| CSI {row['composite_sentiment_index']:+6.1f} "
            f"| Vel {row['sentiment_velocity']:+5.1f} "
            f"| Sig {row['signal_strength']:5.1f} "
            f"| Mom {row['momentum_score']:+5.1f} "
            f"| {row['investment_signal']}"
        )
    print(f"{'─'*60}")
    print("  PIPELINE COMPLETE.\n")


# ══════════════════════════════════════════════════════════
# ENTRY POINT — runs once and exits always
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    lock_path = os.path.join(DATA_DIR, "pipeline.lock")

    with open(lock_path, "w") as f:
        f.write(datetime.now().isoformat())

    try:
        run_pipeline()
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)

    sys.exit(0)
