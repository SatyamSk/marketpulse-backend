"""
MarketPulse AI — Autonomous Agent Engine
Uses OpenAI function calling to let GPT decide what data to fetch.
The agent THINKS about what it needs, calls tools, and reasons holistically.
"""

import json, os, traceback
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Any

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
IST = timezone(timedelta(hours=5, minutes=30))
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(DATA_DIR, "pipeline_live.log")


def _log(line: str):
    """Append a single line to the live pipeline log (powers SSE stream)."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8", errors="ignore") as f:
            f.write(line.rstrip("\n") + "\n")
    except Exception:
        pass

# ── TOOL DEFINITIONS (what the agent CAN do) ──────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_market_price",
            "description": "Fetch real-time price data for any financial instrument from Yahoo Finance. Use for stocks, indices, commodities, currencies, bonds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Yahoo Finance symbol. Examples: ^NSEI (Nifty50), ^NSEBANK (BankNifty), BZ=F (Brent Crude), INR=X (INR/USD), ^INDIAVIX (India VIX), GC=F (Gold), ^TNX (US 10Y), RELIANCE.NS, HDFCBANK.NS, ^GSPC (S&P500), ^DJI (Dow Jones)"},
                    "reason": {"type": "string", "description": "Why you need this data"}
                },
                "required": ["symbol", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_macro_snapshot",
            "description": "Fetch macro indicators that drive Indian intraday risk (Brent crude, INR/USD, India VIX, Gold, US 10Y) and return risk flags + override regime if extreme.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_rss_headlines",
            "description": "Fetch latest news headlines from Indian financial RSS feeds. Returns titles, sources, and publish times.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Which sources to fetch. Options: ET_Markets, ET_Economy, Livemint, BS_Markets, Moneycontrol, Reuters, PIB, RBI, SEBI, ALL"
                    },
                    "max_per_source": {"type": "integer", "description": "Max headlines per source (3-20)"}
                },
                "required": ["sources"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_previous_predictions",
            "description": "Get MarketPulse's previous predictions and their accuracy. Use to understand what we got right/wrong recently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "How many days of history (1-30)"}
                },
                "required": ["days"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_financial",
            "description": "Search the web for latest financial news or data. Use when RSS feeds might miss breaking news, or when you need specific data not available via price feeds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query. Be specific. Example: 'India FII DII flows today', 'Brent crude OPEC latest', 'RBI policy rate decision'"},
                    "reason": {"type": "string", "description": "Why this search matters for the analysis"}
                },
                "required": ["query", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_headlines_batch",
            "description": "Send a batch of headlines to GPT for sector classification and sentiment analysis. Returns structured data for each headline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "headlines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of headline titles to analyze (max 30 per batch)"
                    },
                    "macro_context": {"type": "string", "description": "Current macro context to inform the analysis (crude price, INR level, VIX, etc.)"}
                },
                "required": ["headlines", "macro_context"]
            }
        }
    }
]


# ── TOOL IMPLEMENTATIONS ──────────────────────────────────────────

RSS_MAP = {
    "ET_Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "ET_Economy": "https://economictimes.indiatimes.com/economy/rssfeeds/1373380680.cms",
    "ET_Tech": "https://economictimes.indiatimes.com/tech/rssfeeds/13357263.cms",
    "ET_Startups": "https://economictimes.indiatimes.com/small-biz/startups/rssfeeds/13357270.cms",
    "ET_Industry": "https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms",
    "Livemint_Markets": "https://www.livemint.com/rss/markets",
    "Livemint_Companies": "https://www.livemint.com/rss/companies",
    "Livemint_Economy": "https://www.livemint.com/rss/economy",
    "BS_Markets": "https://www.business-standard.com/rss/markets-106.rss",
    "BS_Economy": "https://www.business-standard.com/rss/economy-policy-101.rss",
    "Moneycontrol": "https://www.moneycontrol.com/rss/latestnews.xml",
    "FinancialExpress": "https://www.financialexpress.com/market/feed/",
    "Reuters": "https://feeds.reuters.com/reuters/INbusinessNews",
    "PIB": "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",
    "RBI": "https://www.rbi.org.in/scripts/rss.aspx",
    "SEBI": "https://www.sebi.gov.in/sebi_data/rss/sebi_news.xml",
}

def _exec_fetch_market_price(symbol, reason=""):
    """Fetch price from Yahoo Finance."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode())
        result = data["chart"]["result"][0]
        quotes = result["indicators"]["quote"][0]
        meta = result.get("meta", {})
        # Find latest valid close
        close_val = None
        prev_val = None
        for i in range(len(quotes["close"])-1, -1, -1):
            if quotes["close"][i] is not None:
                if close_val is None:
                    close_val = quotes["close"][i]
                    open_val = quotes["open"][i] or close_val
                elif prev_val is None:
                    prev_val = quotes["close"][i]
                    break
        if close_val is None:
            return json.dumps({"error": f"No data for {symbol}"})
        prev = prev_val or open_val
        change = round(((close_val - prev) / prev) * 100, 2) if prev else 0
        return json.dumps({
            "symbol": symbol,
            "name": meta.get("shortName", symbol),
            "price": round(close_val, 2),
            "prev_close": round(prev, 2),
            "change_pct": change,
            "currency": meta.get("currency", ""),
            "direction": "up" if change > 0.1 else "down" if change < -0.1 else "flat"
        })
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch {symbol}: {str(e)[:100]}"})


def _exec_fetch_rss(sources, max_per_source=10):
    """Fetch RSS headlines."""
    import feedparser
    from dateutil import parser as dateparser
    
    if "ALL" in sources:
        sources = list(RSS_MAP.keys())
    
    now = datetime.now(IST)
    results = []
    for src in sources:
        url = RSS_MAP.get(src)
        if not url:
            continue
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                if count >= max_per_source:
                    break
                title = entry.get("title", "").strip()
                if len(title) < 10:
                    continue
                # Parse time
                pub_str = entry.get("published", entry.get("updated", ""))
                try:
                    dt = dateparser.parse(pub_str)
                    if dt and dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    dt_ist = dt.astimezone(IST) if dt else now
                    days_old = (now.date() - dt_ist.date()).days
                    if days_old > 1:
                        continue
                    hours = round((now - dt_ist).total_seconds() / 3600, 1)
                except:
                    hours = 0
                results.append({"title": title, "source": src, "hours_old": hours, "url": entry.get("link", "")})
                count += 1
        except Exception as e:
            results.append({"error": f"{src}: {str(e)[:80]}"})
    
    # Don't hard-cap to 100; user controls max_per_source. Still keep a safety cap.
    headlines = [r for r in results if isinstance(r, dict) and r.get("title")]
    return json.dumps({"count": len(headlines), "headlines": headlines[:600]})

def _exec_fetch_macro_snapshot():
    from macro_fetcher import fetch_all_macro_data
    macro = fetch_all_macro_data()
    return json.dumps(macro, default=str)


def _exec_get_predictions(days=7):
    """Get past predictions from database."""
    try:
        from database import get_accuracy_stats
        stats = get_accuracy_stats(days)
        return json.dumps(stats, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "predictions": []})


def _exec_web_search(query, reason=""):
    """Simple web search via DuckDuckGo (free, no API key)."""
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query + ' site:economictimes.com OR site:livemint.com OR site:moneycontrol.com')}"
        req = urllib.request.Request(search_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        # Extract snippets (basic parsing)
        import re
        snippets = re.findall(r'class="result__snippet">(.*?)</a>', html, re.DOTALL)
        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)
        results = []
        for i in range(min(5, len(titles))):
            t = re.sub(r'<[^>]+>', '', titles[i]).strip()
            s = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
            results.append({"title": t, "snippet": s})
        return json.dumps({"query": query, "results": results})
    except Exception as e:
        return json.dumps({"query": query, "error": str(e)[:100], "results": []})


def _exec_analyze_batch(headlines, macro_context=""):
    """Analyze headlines with macro context injected."""
    if not headlines:
        return json.dumps({"analyzed": []})
    
    batch_text = "\n".join([f"{i+1}. {h}" for i, h in enumerate(headlines[:30])])
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"""You are a senior Indian equity analyst. Classify each headline.
{macro_context}

IMPORTANT: Weight your analysis against the macro context above. If crude is spiking, Energy sector headlines should reflect higher impact. If INR is crashing, IT (exporters) might benefit while import-heavy sectors suffer.

For each headline, return JSON with: sector (Banking/Energy/IT/Fintech/Manufacturing/Healthcare/FMCG/Startup/Retail/Other), sentiment (positive/negative/neutral), impact_score (1-10), price_direction (bullish/bearish/neutral), catalyst_type, one_line_insight (max 200 chars), affected_companies (list), geopolitical_risk (bool).

Return a JSON object with key "analyzed" containing an array of objects."""},
                {"role": "user", "content": f"Analyze these headlines:\n{batch_text}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=4000,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return json.dumps({"error": str(e)[:200], "analyzed": []})


# ── TOOL DISPATCHER ───────────────────────────────────────────────

def execute_tool(name, args):
    """Execute a tool by name with given arguments."""
    if name == "fetch_market_price":
        return _exec_fetch_market_price(args.get("symbol"), args.get("reason", ""))
    elif name == "fetch_macro_snapshot":
        return _exec_fetch_macro_snapshot()
    elif name == "fetch_rss_headlines":
        return _exec_fetch_rss(args.get("sources", ["ALL"]), args.get("max_per_source", 10))
    elif name == "get_previous_predictions":
        return _exec_get_predictions(args.get("days", 7))
    elif name == "web_search_financial":
        import urllib.parse
        return _exec_web_search(args.get("query"), args.get("reason", ""))
    elif name == "analyze_headlines_batch":
        return _exec_analyze_batch(args.get("headlines", []), args.get("macro_context", ""))
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


# ── THE AGENT ─────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are MarketPulse AI — an autonomous market intelligence agent for Indian intraday traders.

Your personality: a calm, ruthless, 20-year experienced India intraday professional (risk-first).
You do NOT hallucinate. If you lack a datapoint, you explicitly fetch it using tools.

YOUR MISSION
Produce the most accurate, actionable pre-market / intraday intelligence possible for *today* in the Indian context.

You have tools. Use them strategically and in this order:
1) Fetch macro snapshot FIRST (crude, INR, India VIX, gold, US10Y) via fetch_macro_snapshot
2) Check global risk tone using fetch_market_price (S&P500, Dow, Nasdaq, DXY proxy if needed, Brent again if needed)
3) Pull Indian RSS headlines (ALL) and then analyze in batches WITH the macro context
4) If there is a macro shock / unusual move / large gap risk, use web_search_financial to confirm cause (wars, RBI, OPEC, sanctions, policy headlines)
5) Check your own recent accuracy (get_previous_predictions) and calibrate confidence down if you’ve been wrong recently

MENTAL CHECKLIST (India intraday, non-negotiable):
- Macro shock check: crude/INR/VIX dominates everything. If extreme → override the regime.
- Event calendar logic: RBI/US Fed/BoJ, inflation prints, budget/policy, major earnings day, monthly expiry/weekly expiry context.
- Flows logic (proxy if needed): risk-off abroad + INR weakness usually means FII selling pressure in India.
- Sector rotation logic: crude up → OMC/paint/aviation negative; INR down → IT exporters positive, oil marketing & importers negative; high yields → growth/expensive names hit.
- Risk plan: define “what would invalidate this view” (one clear line).
- Output must be tradeable: bias + risk + what to watch + what to avoid.

CRITICAL RULES
- Macro overrides headlines when extreme.
- A 5% crude spike matters more than 100 positive headlines.
- INR at fresh highs is systemic risk; say it loudly.
- VIX above 22 = fear; bias toward Risk Off and wider stop logic.
- Always build causality chains (driver → transmission → sector winners/losers).
- Always state confidence and WHY (data quality).

After gathering data, produce a structured JSON output with:
{
  "regime": "Risk On" | "Risk Off" | "Complacent" | "Panic",
  "regime_confidence": 0-100,
  "nifty_direction": "bullish" | "bearish" | "neutral",
  "nifty_reasoning": "...",
  "macro_summary": "...",
  "risk_flags": ["..."],
  "sector_signals": {"Banking": "BUY BIAS" | "AVOID" | "CAUTION" | "IMPROVING" | "CONTRARIAN WATCH" | "NEUTRAL", ...},
  "top_insight": "single most important actionable insight",
  "invalidations": ["one or two concrete invalidation conditions"],
  "data_quality": "high" | "medium" | "low",
  "headlines_analyzed": number,
  "agent_reasoning": "step-by-step reasoning"
}"""


def run_agent(max_iterations=8):
    """
    Run the autonomous agent loop.
    The agent decides what tools to call and reasons about the results.
    """
    now = datetime.now(IST)
    _log(f"{'='*60}")
    _log(f"MARKETPULSE AGENT — {now.strftime('%Y-%m-%d %H:%M IST')}")
    _log(f"{'='*60}")
    
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Today is {now.strftime('%A, %d %B %Y')}. Indian market hours: 9:15 AM - 3:30 PM IST. Current time: {now.strftime('%H:%M IST')}. Produce today's complete market intelligence. Start by checking macro conditions, then headlines, then synthesize everything."}
    ]
    
    iteration = 0
    all_tool_results = {}
    
    while iteration < max_iterations:
        iteration += 1
        _log(f"\n  ── Agent iteration {iteration}/{max_iterations} ──")
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=4000,
            )
        except Exception as e:
            _log(f"  [!] Agent API error: {e}")
            break
        
        msg = response.choices[0].message
        messages.append(msg)
        
        # Check if agent wants to call tools
        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                _log(f"    🔧 {fn_name}({json.dumps(fn_args)[:200]})")
                
                result = execute_tool(fn_name, fn_args)
                all_tool_results[f"{fn_name}_{iteration}"] = result
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result[:8000]  # Truncate very large results
                })
        else:
            # Agent is done — it produced its final analysis
            final_content = msg.content
            _log(f"\n  ✅ Agent completed in {iteration} iterations")
            
            # Try to parse as JSON
            try:
                # Find JSON in the response
                import re
                json_match = re.search(r'\{[\s\S]*\}', final_content)
                if json_match:
                    result = json.loads(json_match.group())
                    result["iterations"] = iteration
                    result["timestamp"] = now.isoformat()
                    result["tool_calls"] = list(all_tool_results.keys())
                    result["_tool_results"] = all_tool_results
                    return result
            except json.JSONDecodeError:
                pass
            
            # If not valid JSON, wrap the text response
            return {
                "regime": "Risk Off",
                "regime_confidence": 50,
                "nifty_direction": "neutral",
                "agent_reasoning": final_content,
                "iterations": iteration,
                "timestamp": now.isoformat(),
                "_tool_results": all_tool_results,
                "raw_response": True
            }
    
    _log(f"  [!] Agent hit max iterations ({max_iterations})")
    return {"error": "Max iterations reached", "iterations": max_iterations}


def run_agent_pipeline(max_per_source: int = 14):
    """
    Run the full agent pipeline:
    1. Agent gathers data and produces analysis
    2. Results saved to database
    3. GitHub backup
    """
    from database import (
        create_pipeline_run, complete_pipeline_run, save_prediction
    )
    
    run_id = create_pipeline_run()
    _log(f"  Pipeline run #{run_id}")
    
    # Run the "brain" agent (reasoning + optional web verification). Data collection is enforced below.
    result = run_agent(max_iterations=10)
    
    if "error" in result and not result.get("regime"):
        _log(f"  [!] Agent failed: {result.get('error')}")
        return result

    # ── Build dashboard dataset from tool outputs ───────────────────
    tool_results = result.get("_tool_results", {}) or {}

    macro: dict[str, Any] = {}
    rss_headlines: list[dict[str, Any]] = []
    analyzed: list[dict[str, Any]] = []

    for key, raw in tool_results.items():
        if key.startswith("fetch_macro_snapshot_"):
            try:
                macro = json.loads(raw)
            except Exception:
                macro = {}
        if key.startswith("fetch_rss_headlines_"):
            try:
                payload = json.loads(raw)
                rss_headlines = [h for h in payload.get("headlines", []) if isinstance(h, dict) and h.get("title")]
            except Exception:
                rss_headlines = []
        if key.startswith("analyze_headlines_batch_"):
            try:
                payload = json.loads(raw)
                analyzed = payload.get("analyzed", []) if isinstance(payload, dict) else []
            except Exception:
                analyzed = []

    # If agent didn't call the tools in the expected order, do the minimum required here.
    if not macro:
        from macro_fetcher import fetch_all_macro_data
        macro = fetch_all_macro_data()
    if not rss_headlines:
        rss_payload = json.loads(_exec_fetch_rss(["ALL"], max_per_source=int(max_per_source)))
        rss_headlines = [h for h in rss_payload.get("headlines", []) if h.get("title")]
    if not analyzed:
        from macro_fetcher import format_macro_context_for_gpt
        macro_context = format_macro_context_for_gpt(macro)
        titles = [h.get("title", "") for h in rss_headlines if h.get("title")]

        _log(f"  RSS fetched: {len(titles)} headlines across {len(RSS_MAP)} sources (max_per_source={max_per_source})")
        analyzed_all: list[dict[str, Any]] = []
        for i in range(0, len(titles), 30):
            batch = titles[i:i+30]
            _log(f"  Analyzing headlines batch {i//30+1}: {len(batch)} items")
            analyzed_payload = json.loads(_exec_analyze_batch(batch, macro_context))
            batch_analyzed = analyzed_payload.get("analyzed", []) if isinstance(analyzed_payload, dict) else []
            if not isinstance(batch_analyzed, list):
                batch_analyzed = []
            analyzed_all.extend(batch_analyzed)
        analyzed = analyzed_all

    # Merge analyzed results back onto raw RSS headlines by index
    merged_rows: list[dict[str, Any]] = []
    for i, h in enumerate(rss_headlines[: min(len(rss_headlines), len(analyzed))]):
        a = analyzed[i] if i < len(analyzed) else {}
        merged_rows.append({
            "title": h.get("title", ""),
            "description": "",
            "source": h.get("source", ""),
            "source_url": "",
            "published": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
            "hours_old": float(h.get("hours_old", 0) or 0),
            "url": h.get("url", ""),
            "is_govt_source": bool(str(h.get("source", "")).upper() in ("PIB", "RBI", "SEBI")),
            # analyzed fields (dashboard schema expects these)
            "sector": a.get("sector", "Other"),
            "sentiment": a.get("sentiment", "neutral"),
            "sentiment_confidence": float(a.get("sentiment_confidence", 0.7) or 0.7),
            "impact_score": float(a.get("impact_score", 5) or 5),
            "valence": float(a.get("valence", 0.5) or 0.5),
            "arousal": float(a.get("arousal", 0.5) or 0.5),
            "geopolitical_risk": bool(a.get("geopolitical_risk", False)),
            "affected_companies": a.get("affected_companies", []),
            "second_order_beneficiaries": a.get("second_order_beneficiaries", []),
            "catalyst_type": a.get("catalyst_type", "other"),
            "price_direction": a.get("price_direction", "neutral"),
            "time_horizon": a.get("time_horizon", "intraday"),
            "conviction": a.get("conviction", "low"),
            "macro_sensitivity": a.get("macro_sensitivity", "medium"),
            "one_line_insight": a.get("one_line_insight", ""),
            "signal_reason": a.get("signal_reason", ""),
            "contrarian_flag": bool(a.get("contrarian_flag", False)),
            "contrarian_reason": a.get("contrarian_reason", ""),
            "source_reliability": 1.0,
        })

    headlines_df = pd.DataFrame(merged_rows)

    # Use the same metric engine as the original pipeline so the UI stays consistent
    from pipeline import calculate_metrics, calculate_market_stress_index, save_all
    scored_headlines_df, sector_df = calculate_metrics(headlines_df) if not headlines_df.empty else (headlines_df, pd.DataFrame())
    msi = calculate_market_stress_index(scored_headlines_df, sector_df) if not scored_headlines_df.empty else {"msi": 0, "level": "Low"}

    save_all(scored_headlines_df, sector_df, msi, run_id)

    avg_nss = float(sector_df["composite_sentiment_index"].mean()) if not sector_df.empty else 0.0
    avg_risk = float(sector_df["avg_weighted_risk"].mean()) if not sector_df.empty else 0.0

    # Macro override (if extreme) should override regime label
    macro_override = macro.get("macro_regime_override")
    regime = macro_override or result.get("regime") or "Risk Off"

    complete_pipeline_run(
        run_id,
        int(len(scored_headlines_df)),
        float(avg_nss),
        float(avg_risk),
        str(regime),
        float(macro.get("macro_risk_score", 0) or 0),
        str(msi.get("level", "Low")),
    )
    
    # Save prediction
    sector_signals = result.get("sector_signals", {})
    today = datetime.now(IST).strftime("%Y-%m-%d")
    save_prediction(today, regime, avg_nss, avg_risk, sector_signals)
    
    # Save agent result as JSON for the API to serve
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    _log(f"\n  ✅ Agent pipeline complete. Regime: {regime} (confidence: {result.get('regime_confidence', '?')}%)")
    _log(f"  Headlines saved: {len(scored_headlines_df)} | sectors saved: {len(sector_df)} | MSI: {msi.get('msi')} ({msi.get('level')})")
    
    return result


if __name__ == "__main__":
    result = run_agent_pipeline()
    print(f"\nFinal result: {json.dumps(result, indent=2, default=str)[:2000]}")
