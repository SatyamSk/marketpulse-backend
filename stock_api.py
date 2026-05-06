"""
Indian Stock Market API Client
Source: http://65.0.104.9/ (0xramm/Indian-Stock-Market-API)

Fetches real-time NSE/BSE prices for stocks mentioned in headlines.
No API key required.
"""

from __future__ import annotations

import requests
from typing import Any, Dict, List, Optional
from functools import lru_cache
import time
import re

STOCK_API_BASE = "http://65.0.104.9"
TIMEOUT = 8  # seconds

# ── Top Nifty-50 stocks by sector ──────────────────────────────────
# When an agent identifies an affected sector, we map it to key stocks
SECTOR_STOCKS: Dict[str, List[str]] = {
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
    "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
    "Energy": ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "ADANIGREEN"],
    "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP"],
    "FMCG": ["ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA", "DABUR"],
    "Metals": ["TATASTEEL", "HINDALCO", "JSWSTEEL", "VEDL", "COALINDIA"],
    "Infra": ["LARSENTOUBRO", "ULTRACEMCO", "ADANIENT", "ADANIPORTS", "GRASIM"],
    "Finance": ["BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "ICICIPRULI"],
    "Telecom": ["BHARTIARTL", "IDEA"],
    "Realty": ["DLF", "GODREJPROP", "OBEROIRLTY"],
    "Defence": ["HAL", "BEL", "BHEL"],
    "Media": ["ZEEL", "PVR"],
}

# Flatten for quick lookup
ALL_TRACKED_SYMBOLS = set()
for stocks in SECTOR_STOCKS.values():
    ALL_TRACKED_SYMBOLS.update(stocks)

# Key indices / bellwethers — always fetch these
BELLWETHER_STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC"]


def fetch_stock(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch a single stock's real-time data."""
    try:
        r = requests.get(
            f"{STOCK_API_BASE}/stock",
            params={"symbol": symbol, "res": "num"},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "success":
                return data.get("data", {})
    except Exception:
        pass
    return None


def fetch_stocks_batch(symbols: List[str]) -> List[Dict[str, Any]]:
    """Fetch multiple stocks in one API call."""
    if not symbols:
        return []
    try:
        r = requests.get(
            f"{STOCK_API_BASE}/stock/list",
            params={"symbols": ",".join(symbols), "res": "num"},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "success":
                return data.get("stocks", [])
    except Exception:
        pass
    return []


def search_stock(query: str) -> List[Dict[str, Any]]:
    """Search for stocks by company name."""
    try:
        r = requests.get(
            f"{STOCK_API_BASE}/search",
            params={"q": query},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "success":
                return data.get("results", [])
    except Exception:
        pass
    return []


def get_market_snapshot() -> Dict[str, Any]:
    """
    Get a real-time snapshot of key market bellwethers.
    Returns price, change, percent_change for top stocks.
    """
    stocks = fetch_stocks_batch(BELLWETHER_STOCKS)
    snapshot = []
    for s in stocks:
        snapshot.append({
            "symbol": s.get("symbol", ""),
            "name": s.get("company_name", ""),
            "price": s.get("last_price", 0),
            "change": s.get("change", 0),
            "pct_change": s.get("percent_change", 0),
            "volume": s.get("volume", 0),
            "sector": s.get("sector", ""),
        })
    return {"bellwethers": snapshot, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def get_sector_stocks(sector: str) -> List[Dict[str, Any]]:
    """Get real-time data for stocks in a specific sector."""
    # Fuzzy match sector name
    sector_lower = sector.lower().strip()
    matched_key = None
    for key in SECTOR_STOCKS:
        if key.lower() in sector_lower or sector_lower in key.lower():
            matched_key = key
            break
    
    if not matched_key:
        return []
    
    symbols = SECTOR_STOCKS[matched_key]
    return fetch_stocks_batch(symbols)


def extract_mentioned_stocks(headlines: List[str]) -> List[str]:
    """
    Extract stock symbols mentioned in headlines.
    Returns unique symbols that we can fetch.
    """
    # Company name → symbol mapping
    name_to_symbol = {
        "reliance": "RELIANCE", "tcs": "TCS", "infosys": "INFY", "infy": "INFY",
        "hdfc bank": "HDFCBANK", "hdfc": "HDFCBANK", "icici bank": "ICICIBANK",
        "icici": "ICICIBANK", "sbi": "SBIN", "state bank": "SBIN",
        "kotak": "KOTAKBANK", "axis bank": "AXISBANK", "axis": "AXISBANK",
        "wipro": "WIPRO", "hcl tech": "HCLTECH", "hcl": "HCLTECH",
        "tech mahindra": "TECHM", "maruti": "MARUTI", "tata motors": "TATAMOTORS",
        "bajaj auto": "BAJAJ-AUTO", "bajaj finance": "BAJFINANCE",
        "sun pharma": "SUNPHARMA", "dr reddy": "DRREDDY", "cipla": "CIPLA",
        "itc": "ITC", "hindustan unilever": "HINDUNILVR", "hul": "HINDUNILVR",
        "nestle": "NESTLEIND", "britannia": "BRITANNIA",
        "tata steel": "TATASTEEL", "hindalco": "HINDALCO", "jsw steel": "JSWSTEEL",
        "vedanta": "VEDL", "coal india": "COALINDIA",
        "l&t": "LARSENTOUBRO", "larsen": "LARSENTOUBRO",
        "ultratech": "ULTRACEMCO", "adani": "ADANIENT", "adani ports": "ADANIPORTS",
        "bharti airtel": "BHARTIARTL", "airtel": "BHARTIARTL",
        "ntpc": "NTPC", "power grid": "POWERGRID", "ongc": "ONGC",
        "dlf": "DLF", "hal": "HAL", "bhel": "BHEL", "bel": "BEL",
        "titan": "TITAN", "asian paints": "ASIANPAINT",
        "bajaj finserv": "BAJAJFINSV", "hdfc life": "HDFCLIFE",
        "sbi life": "SBILIFE",
    }
    
    found = set()
    combined = " ".join(headlines).lower()
    
    # Sort by length descending to match longer names first (e.g., "hdfc bank" before "hdfc")
    for name in sorted(name_to_symbol, key=len, reverse=True):
        if name in combined:
            found.add(name_to_symbol[name])
    
    return list(found)[:15]  # Cap at 15 to avoid API overload


def get_stocks_for_today(headlines: List[Dict], sector_data: List[Dict]) -> Dict[str, Any]:
    """
    Main entry point for the Today page.
    
    1. Fetch bellwether stocks (always shown)
    2. Extract stocks mentioned in today's headlines
    3. Fetch top movers from most-affected sectors
    4. Generate BUY/HOLD/SELL/LONG/SHORT recommendations
    5. Return a combined view
    """
    # Build sentiment context from headlines
    headline_sentiment = _build_headline_sentiment(headlines)
    sector_sentiment = _build_sector_sentiment(sector_data)
    
    # 1. Bellwethers
    bellwethers = fetch_stocks_batch(BELLWETHER_STOCKS)
    bellwether_list = []
    for s in bellwethers:
        sym = s.get("symbol", "")
        pct = s.get("percent_change", 0) or 0
        rec = _generate_recommendation(sym, pct, headline_sentiment, sector_sentiment, s)
        bellwether_list.append({
            "symbol": sym,
            "name": _short_name(s.get("company_name", "")),
            "price": s.get("last_price", 0),
            "change": s.get("change", 0),
            "pct_change": round(pct, 2),
            "sector": s.get("sector", ""),
            **rec,
        })
    
    # 2. Headlines-mentioned stocks
    titles = [h.get("title", "") for h in headlines if isinstance(h, dict)]
    mentioned_symbols = extract_mentioned_stocks(titles)
    mentioned_symbols = [s for s in mentioned_symbols if s not in BELLWETHER_STOCKS]
    
    mentioned_stocks = []
    if mentioned_symbols:
        raw = fetch_stocks_batch(mentioned_symbols[:10])
        for s in raw:
            sym = s.get("symbol", "")
            pct = s.get("percent_change", 0) or 0
            rec = _generate_recommendation(sym, pct, headline_sentiment, sector_sentiment, s)
            mentioned_stocks.append({
                "symbol": sym,
                "name": _short_name(s.get("company_name", "")),
                "price": s.get("last_price", 0),
                "change": s.get("change", 0),
                "pct_change": round(pct, 2),
                "sector": s.get("sector", ""),
                "reason": "Mentioned in today's headlines",
                **rec,
            })
    
    # 3. Top affected sector stocks
    sector_stocks = []
    if sector_data:
        sorted_sectors = sorted(
            sector_data,
            key=lambda x: abs(float(x.get("composite_sentiment_index", 0))),
            reverse=True,
        )
        for sec in sorted_sectors[:2]:
            sec_name = sec.get("sector", "")
            stocks_in_sec = get_sector_stocks(sec_name)
            for s in stocks_in_sec[:3]:
                sym = s.get("symbol", "")
                if sym not in BELLWETHER_STOCKS and sym not in [m["symbol"] for m in mentioned_stocks]:
                    pct = s.get("percent_change", 0) or 0
                    rec = _generate_recommendation(sym, pct, headline_sentiment, sector_sentiment, s)
                    sector_stocks.append({
                        "symbol": sym,
                        "name": _short_name(s.get("company_name", "")),
                        "price": s.get("last_price", 0),
                        "change": s.get("change", 0),
                        "pct_change": round(pct, 2),
                        "sector": s.get("sector", ""),
                        "reason": f"Top stock in {sec_name}",
                        **rec,
                    })
    
    return {
        "bellwethers": bellwether_list,
        "mentioned": mentioned_stocks,
        "sector_picks": sector_stocks,
        "total_tracked": len(bellwether_list) + len(mentioned_stocks) + len(sector_stocks),
    }


def _build_headline_sentiment(headlines: List[Dict]) -> Dict[str, Dict]:
    """Build per-symbol sentiment from headlines."""
    sentiment_map: Dict[str, Dict] = {}
    for h in headlines:
        if not isinstance(h, dict):
            continue
        companies = h.get("affected_companies", [])
        if isinstance(companies, str):
            companies = [c.strip() for c in companies.split(",") if c.strip()]
        sentiment = h.get("sentiment", "neutral")
        impact = float(h.get("impact_score", 5) or 5)
        direction = h.get("price_direction", "neutral")
        
        for company in companies:
            company_upper = company.upper().strip()
            if company_upper not in sentiment_map:
                sentiment_map[company_upper] = {"pos": 0, "neg": 0, "total": 0, "impact_sum": 0, "directions": []}
            sentiment_map[company_upper]["total"] += 1
            sentiment_map[company_upper]["impact_sum"] += impact
            sentiment_map[company_upper]["directions"].append(direction)
            if sentiment == "positive":
                sentiment_map[company_upper]["pos"] += 1
            elif sentiment == "negative":
                sentiment_map[company_upper]["neg"] += 1
    return sentiment_map


def _build_sector_sentiment(sector_data: List[Dict]) -> Dict[str, float]:
    """Build per-sector sentiment score."""
    sector_map = {}
    for s in sector_data:
        if isinstance(s, dict):
            name = s.get("sector", "")
            csi = float(s.get("composite_sentiment_index", 0) or 0)
            sector_map[name.lower()] = csi
    return sector_map


def _generate_recommendation(
    symbol: str, pct_change: float,
    headline_sentiment: Dict, sector_sentiment: Dict,
    stock_data: Dict
) -> Dict[str, Any]:
    """
    Generate a recommendation for a stock based on:
    1. Price momentum (current day change)
    2. Headline sentiment (if mentioned)
    3. Sector sentiment
    
    Returns: {signal, signal_label, signal_reason, conviction}
    """
    score = 0.0  # -100 to +100 scale
    reasons = []
    
    # 1. Price momentum (weight: 30%)
    if pct_change > 2.0:
        score += 25
        reasons.append(f"Strong momentum (+{pct_change:.1f}%)")
    elif pct_change > 0.5:
        score += 15
        reasons.append(f"Positive momentum (+{pct_change:.1f}%)")
    elif pct_change < -2.0:
        score -= 25
        reasons.append(f"Sharp decline ({pct_change:.1f}%)")
    elif pct_change < -0.5:
        score -= 15
        reasons.append(f"Under pressure ({pct_change:.1f}%)")
    
    # 2. Headline sentiment (weight: 40%)
    sym_upper = symbol.upper()
    if sym_upper in headline_sentiment:
        hs = headline_sentiment[sym_upper]
        if hs["total"] > 0:
            net = (hs["pos"] - hs["neg"]) / hs["total"]
            avg_impact = hs["impact_sum"] / hs["total"]
            score += net * 40 * (avg_impact / 10)
            if hs["pos"] > hs["neg"]:
                reasons.append(f"Positive news ({hs['pos']}/{hs['total']} bullish headlines)")
            elif hs["neg"] > hs["pos"]:
                reasons.append(f"Negative news ({hs['neg']}/{hs['total']} bearish headlines)")
            # Check direction predictions
            bullish_dirs = sum(1 for d in hs["directions"] if d == "bullish")
            bearish_dirs = sum(1 for d in hs["directions"] if d == "bearish")
            if bullish_dirs > bearish_dirs:
                score += 10
            elif bearish_dirs > bullish_dirs:
                score -= 10
    
    # 3. Sector sentiment (weight: 30%)
    stock_sector = (stock_data.get("sector", "") or "").lower()
    if stock_sector and stock_sector in sector_sentiment:
        csi = sector_sentiment[stock_sector]
        if csi > 0.3:
            score += 20
            reasons.append(f"Sector tailwind ({stock_sector})")
        elif csi < -0.3:
            score -= 20
            reasons.append(f"Sector headwind ({stock_sector})")
    
    # Map score to signal
    if score >= 30:
        signal, label = "BUY", "Buy"
    elif score >= 15:
        signal, label = "LONG", "Go Long"
    elif score > -15:
        signal, label = "HOLD", "Hold"
    elif score > -30:
        signal, label = "SHORT", "Short"
    else:
        signal, label = "SELL", "Sell"
    
    # Conviction
    abs_score = abs(score)
    if abs_score >= 40:
        conviction = "high"
    elif abs_score >= 20:
        conviction = "medium"
    else:
        conviction = "low"
    
    return {
        "signal": signal,
        "signal_label": label,
        "signal_reason": "; ".join(reasons) if reasons else "No strong signals",
        "conviction": conviction,
        "score": round(score, 1),
    }


def _short_name(name: str) -> str:
    """Shorten company names for UI display."""
    replacements = {
        " Limited": "", " Ltd": "", " Corporation": " Corp",
        " Industries": " Ind", " Technologies": " Tech",
        " Consultancy Services": "", " Information Technology": " IT",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.strip()

