"""
MarketPulse AI — Macro Data Fetcher
Fetches real-time macro indicators that the headline pipeline was BLIND to:
  - Brent Crude Oil price & daily change
  - INR/USD exchange rate
  - India VIX (fear index)
  - US 10Y Treasury yield
  - Gold price
  - FII/DII flow direction (estimated from sector index moves)

This data is injected into the GPT classification prompt AND used to 
override regime classification when macro signals are extreme.

Data source: Yahoo Finance (free, reliable, no API key needed).
"""

import json
import os
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

# Thresholds for regime override
CRUDE_PANIC_THRESHOLD = 5.0      # >5% daily spike = crisis
CRUDE_HIGH_THRESHOLD = 3.0       # >3% daily spike = elevated
INR_PANIC_THRESHOLD = 1.5        # >1.5% daily depreciation = crisis
VIX_PANIC_THRESHOLD = 22.0       # VIX above 22 = elevated fear
VIX_CRISIS_THRESHOLD = 28.0      # VIX above 28 = panic territory


def fetch_yahoo_quote(symbol):
    """Fetch latest quote from Yahoo Finance v8 API."""
    try:
        import urllib.request
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())

        result = data["chart"]["result"][0]
        meta = result.get("meta", {})
        quotes = result["indicators"]["quote"][0]
        timestamps = result.get("timestamp", [])

        if not timestamps or not quotes["close"]:
            return None

        # Latest data point
        latest_close = None
        prev_close = None
        for i in range(len(timestamps) - 1, -1, -1):
            if quotes["close"][i] is not None:
                if latest_close is None:
                    latest_close = quotes["close"][i]
                    latest_open = quotes["open"][i] if quotes["open"][i] else latest_close
                elif prev_close is None:
                    prev_close = quotes["close"][i]
                    break

        if latest_close is None:
            return None

        prev = prev_close or latest_open or latest_close
        change_pct = round(((latest_close - prev) / prev) * 100, 2) if prev else 0

        return {
            "price": round(latest_close, 2),
            "prev_close": round(prev, 2),
            "change_pct": change_pct,
            "currency": meta.get("currency", ""),
        }
    except Exception as e:
        print(f"  [!] Yahoo fetch failed for {symbol}: {e}")
        return None


def fetch_all_macro_data():
    """
    Fetch all macro indicators. Returns a dict with all data + risk flags.
    This is the critical data the old pipeline was missing.
    """
    print("  Fetching macro data (Crude, INR, VIX, Gold, US10Y)...")

    macro = {
        "crude_oil": None,
        "inr_usd": None,
        "india_vix": None,
        "gold": None,
        "us_10y": None,
        "fetched_at": datetime.now(IST).isoformat(),
        "risk_flags": [],
        "macro_risk_score": 0,
        "macro_regime_override": None,
    }

    # 1. Brent Crude Oil
    crude = fetch_yahoo_quote("BZ=F")
    if crude:
        macro["crude_oil"] = crude
        if abs(crude["change_pct"]) >= CRUDE_PANIC_THRESHOLD:
            macro["risk_flags"].append(f"CRUDE SHOCK: {crude['change_pct']:+.1f}% (${crude['price']}/bbl)")
        elif abs(crude["change_pct"]) >= CRUDE_HIGH_THRESHOLD:
            macro["risk_flags"].append(f"CRUDE ELEVATED: {crude['change_pct']:+.1f}% (${crude['price']}/bbl)")
        print(f"    Brent Crude: ${crude['price']}/bbl ({crude['change_pct']:+.1f}%)")

    # 2. INR/USD
    inr = fetch_yahoo_quote("INR=X")
    if inr:
        macro["inr_usd"] = inr
        # For INR, higher number = depreciation (bad)
        if inr["change_pct"] >= INR_PANIC_THRESHOLD:
            macro["risk_flags"].append(f"INR CRISIS: ₹{inr['price']}/USD ({inr['change_pct']:+.1f}% depreciation)")
        elif inr["change_pct"] >= 0.8:
            macro["risk_flags"].append(f"INR WEAK: ₹{inr['price']}/USD ({inr['change_pct']:+.1f}%)")
        print(f"    INR/USD: ₹{inr['price']} ({inr['change_pct']:+.1f}%)")

    # 3. India VIX
    vix = fetch_yahoo_quote("^INDIAVIX")
    if vix:
        macro["india_vix"] = vix
        if vix["price"] >= VIX_CRISIS_THRESHOLD:
            macro["risk_flags"].append(f"VIX PANIC: {vix['price']:.1f} (CRISIS level)")
        elif vix["price"] >= VIX_PANIC_THRESHOLD:
            macro["risk_flags"].append(f"VIX ELEVATED: {vix['price']:.1f} ({vix['change_pct']:+.1f}%)")
        print(f"    India VIX: {vix['price']:.1f} ({vix['change_pct']:+.1f}%)")

    # 4. Gold
    gold = fetch_yahoo_quote("GC=F")
    if gold:
        macro["gold"] = gold
        if abs(gold["change_pct"]) >= 2.0:
            macro["risk_flags"].append(f"GOLD FLIGHT: ${gold['price']}/oz ({gold['change_pct']:+.1f}%)")
        print(f"    Gold: ${gold['price']}/oz ({gold['change_pct']:+.1f}%)")

    # 5. US 10Y Treasury
    us10y = fetch_yahoo_quote("^TNX")
    if us10y:
        macro["us_10y"] = us10y
        if us10y["price"] >= 5.0:
            macro["risk_flags"].append(f"US YIELD SPIKE: {us10y['price']:.2f}% (EM capital flight risk)")
        print(f"    US 10Y: {us10y['price']:.2f}% ({us10y['change_pct']:+.1f}%)")

    # ── CALCULATE MACRO RISK SCORE ─────────────────────────────────
    score = 0
    
    if crude:
        score += min(abs(crude["change_pct"]) * 8, 30)  # Max 30 from crude
    if inr:
        score += min(abs(inr["change_pct"]) * 12, 25)   # Max 25 from INR
    if vix:
        if vix["price"] >= VIX_CRISIS_THRESHOLD:
            score += 25
        elif vix["price"] >= VIX_PANIC_THRESHOLD:
            score += 15
        elif vix["price"] >= 16:
            score += 8
    if gold and gold["change_pct"] > 1.5:
        score += 10  # Flight to safety signal
    if us10y and us10y["price"] >= 5.0:
        score += 10

    macro["macro_risk_score"] = round(min(score, 100), 1)

    # ── REGIME OVERRIDE ────────────────────────────────────────────
    # If macro signals are extreme, override the headline-based regime
    if macro["macro_risk_score"] >= 60:
        macro["macro_regime_override"] = "Panic"
        print(f"    ⚠ MACRO OVERRIDE → PANIC (score: {macro['macro_risk_score']})")
    elif macro["macro_risk_score"] >= 35:
        macro["macro_regime_override"] = "Risk Off"
        print(f"    ⚠ MACRO OVERRIDE → RISK OFF (score: {macro['macro_risk_score']})")
    elif macro["macro_risk_score"] >= 20:
        macro["macro_regime_override"] = "Complacent"
    
    flags_count = len(macro["risk_flags"])
    print(f"    Macro risk score: {macro['macro_risk_score']}/100, {flags_count} risk flag(s)")

    return macro


def format_macro_context_for_gpt(macro):
    """
    Format macro data as text context to inject into GPT classification prompts.
    This is the missing data that caused the 4.5/10 audit score.
    """
    lines = ["REAL-TIME MACRO CONTEXT (weight heavily):"]
    
    if macro.get("crude_oil"):
        c = macro["crude_oil"]
        lines.append(f"  Brent Crude: ${c['price']}/bbl ({c['change_pct']:+.1f}% today)")
    if macro.get("inr_usd"):
        i = macro["inr_usd"]
        lines.append(f"  INR/USD: ₹{i['price']} ({i['change_pct']:+.1f}% today)")
    if macro.get("india_vix"):
        v = macro["india_vix"]
        lines.append(f"  India VIX: {v['price']:.1f} ({v['change_pct']:+.1f}% today)")
    if macro.get("gold"):
        g = macro["gold"]
        lines.append(f"  Gold: ${g['price']}/oz ({g['change_pct']:+.1f}%)")
    if macro.get("us_10y"):
        u = macro["us_10y"]
        lines.append(f"  US 10Y Yield: {u['price']:.2f}%")
    
    if macro.get("risk_flags"):
        lines.append(f"  ⚠ RISK FLAGS: {'; '.join(macro['risk_flags'])}")
    
    lines.append(f"  Macro Risk Score: {macro.get('macro_risk_score', 0)}/100")
    
    if macro.get("macro_regime_override"):
        lines.append(f"  ⚠ MACRO REGIME OVERRIDE: {macro['macro_regime_override']}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    data = fetch_all_macro_data()
    print("\n" + format_macro_context_for_gpt(data))
    print(f"\nRaw data: {json.dumps(data, indent=2, default=str)}")
