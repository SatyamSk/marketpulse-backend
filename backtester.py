"""
MarketPulse AI — Backtester
Compares morning predictions against actual Nifty returns.
Uses Yahoo Finance for reliable, free daily close data.
"""

import json
import os
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))


def fetch_nifty_data(date_str=None):
    """
    Fetch Nifty 50 OHLC data from Yahoo Finance.
    Returns dict with open, close, change_pct for the given date.
    """
    try:
        import urllib.request
        import json as _json
        
        # Yahoo Finance v8 API for Nifty 50
        symbol = "^NSEI"
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
        
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        
        with urllib.request.urlopen(req, timeout=10) as response:
            data = _json.loads(response.read().decode())
        
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quotes = result["indicators"]["quote"][0]
        
        # Find the matching date or latest
        target_date = date_str or datetime.now(IST).strftime("%Y-%m-%d")
        
        for i, ts in enumerate(timestamps):
            dt = datetime.fromtimestamp(ts, tz=IST)
            if dt.strftime("%Y-%m-%d") == target_date:
                open_price = round(quotes["open"][i], 2)
                close_price = round(quotes["close"][i], 2)
                change_pct = round(((close_price - open_price) / open_price) * 100, 2)
                return {
                    "date": target_date,
                    "open": open_price,
                    "close": close_price,
                    "change_pct": change_pct,
                    "direction": "bullish" if change_pct > 0.1 else "bearish" if change_pct < -0.1 else "neutral"
                }
        
        # If exact date not found, use latest
        if timestamps:
            i = -1
            open_price = round(quotes["open"][i], 2)
            close_price = round(quotes["close"][i], 2)
            change_pct = round(((close_price - open_price) / open_price) * 100, 2)
            dt = datetime.fromtimestamp(timestamps[i], tz=IST)
            return {
                "date": dt.strftime("%Y-%m-%d"),
                "open": open_price,
                "close": close_price,
                "change_pct": change_pct,
                "direction": "bullish" if change_pct > 0.1 else "bearish" if change_pct < -0.1 else "neutral"
            }
        
        return None
    except Exception as e:
        print(f"  [!] Yahoo Finance fetch failed: {e}")
        return None


def fetch_sector_index_data():
    """
    Fetch sector index data (Bank Nifty, Nifty IT, etc.) from Yahoo Finance.
    Returns dict of sector → direction.
    """
    sector_symbols = {
        "Banking": "^NSEBANK",
        "IT": "^CNXIT",
        "Energy": "^CNXENERGY",
        "FMCG": "^CNXFMCG",
    }
    
    results = {}
    for sector, symbol in sector_symbols.items():
        try:
            import urllib.request
            import json as _json
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=2d"
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = _json.loads(response.read().decode())
            
            result = data["chart"]["result"][0]
            quotes = result["indicators"]["quote"][0]
            
            if quotes["open"] and quotes["close"]:
                open_p = quotes["open"][-1]
                close_p = quotes["close"][-1]
                if open_p and close_p:
                    change = round(((close_p - open_p) / open_p) * 100, 2)
                    results[sector] = {
                        "change_pct": change,
                        "direction": "bullish" if change > 0.1 else "bearish" if change < -0.1 else "neutral"
                    }
        except Exception as e:
            print(f"  [!] Sector index fetch failed for {sector}: {e}")
    
    return results


def run_backtest():
    """
    Main backtest function. Called daily after market close (3:30 PM IST).
    
    1. Fetches actual Nifty data for today
    2. Compares against morning's predicted regime
    3. Updates prediction accuracy
    4. Recalibrates source reliability
    5. Recalibrates dynamic sector weights
    """
    from database import (
        get_db, get_accuracy_stats, update_prediction_accuracy,
        update_source_reliability, update_dynamic_weight,
        get_dynamic_weights
    )
    
    today = datetime.now(IST).strftime("%Y-%m-%d")
    print(f"\n{'='*60}\nBACKTEST — {today}\n{'='*60}")
    
    # 1. Fetch actual market data
    nifty = fetch_nifty_data(today)
    if not nifty:
        print("  [!] Could not fetch Nifty data. Skipping backtest.")
        return
    
    print(f"  Nifty: Open={nifty['open']}, Close={nifty['close']}, "
          f"Change={nifty['change_pct']}%, Direction={nifty['direction']}")
    
    # 2. Get today's prediction
    with get_db() as conn:
        prediction = conn.execute(
            "SELECT * FROM predictions WHERE date = ?", (today,)
        ).fetchone()
    
    if not prediction:
        print("  [!] No prediction found for today. Skipping.")
        return
    
    # 3. Check regime accuracy
    predicted_regime = prediction["predicted_regime"]
    actual_direction = nifty["direction"]
    
    # Regime → expected direction mapping
    regime_to_direction = {
        "Risk On": "bullish",
        "Complacent": "bullish",
        "Risk Off": "bearish",
        "Panic": "bearish",
    }
    expected_direction = regime_to_direction.get(predicted_regime, "neutral")
    was_correct = (expected_direction == actual_direction) or (
        expected_direction == "neutral" and abs(nifty["change_pct"]) < 0.15
    )
    
    print(f"  Predicted: {predicted_regime} → {expected_direction}")
    print(f"  Actual: {actual_direction} ({nifty['change_pct']}%)")
    print(f"  Result: {'✓ CORRECT' if was_correct else '✗ WRONG'}")
    
    # 4. Check sector accuracy
    sector_data = fetch_sector_index_data()
    sector_accuracy = {}
    
    try:
        sector_signals = json.loads(prediction["sector_signals"]) if prediction["sector_signals"] else {}
    except (json.JSONDecodeError, TypeError):
        sector_signals = {}
    
    for sector, predicted_signal in sector_signals.items():
        if sector in sector_data:
            actual = sector_data[sector]["direction"]
            predicted_dir = "bullish" if predicted_signal in ["BUY BIAS", "IMPROVING"] else \
                            "bearish" if predicted_signal == "AVOID" else "neutral"
            correct = (predicted_dir == actual)
            sector_accuracy[sector] = {
                "predicted": predicted_signal,
                "actual_direction": actual,
                "actual_change": sector_data[sector]["change_pct"],
                "correct": correct
            }
    
    # 5. Calculate overall accuracy
    accuracy_items = [v["correct"] for v in sector_accuracy.values()]
    overall = round(sum(accuracy_items) / len(accuracy_items) * 100, 1) if accuracy_items else 0
    
    # 6. Update prediction record
    update_prediction_accuracy(
        today, nifty["open"], nifty["close"], nifty["change_pct"],
        was_correct, sector_accuracy, overall
    )
    
    # 7. Update source reliability based on headline accuracy
    with get_db() as conn:
        # Get today's headlines
        run = conn.execute(
            "SELECT id FROM pipeline_runs WHERE DATE(started_at) = ? ORDER BY id DESC LIMIT 1",
            (today,)
        ).fetchone()
        if run:
            headlines = conn.execute(
                "SELECT source, price_direction, sector FROM headlines WHERE pipeline_run_id = ?",
                (run["id"],)
            ).fetchall()
            for h in headlines:
                h_sector = h["sector"]
                h_predicted = h["price_direction"]
                # Check against actual sector movement
                if h_sector in sector_data:
                    actual = sector_data[h_sector]["direction"]
                    was_h_correct = (h_predicted == actual)
                    update_source_reliability(h["source"], was_h_correct)
    
    # 8. Recalibrate dynamic sector weights (weekly)
    _recalibrate_sector_weights()
    
    print(f"  ✅ Backtest complete. Regime: {'✓' if was_correct else '✗'}, "
          f"Sector accuracy: {overall}%\n")


def _recalibrate_sector_weights():
    """
    Recalibrate sector weights based on accumulated accuracy data.
    Runs weekly — adjusts weights based on which sectors we predict most accurately.
    """
    from database import get_db, update_dynamic_weight
    
    with get_db() as conn:
        # Check if we have at least 7 days of data
        count = conn.execute(
            "SELECT COUNT(*) as c FROM predictions WHERE was_regime_correct IS NOT NULL"
        ).fetchone()["c"]
        
        if count < 7:
            print("  [i] Not enough data for weight recalibration (need 7+ days).")
            return
        
        # Get sector-level accuracy from sector_accuracy JSON
        rows = conn.execute(
            """SELECT sector_accuracy FROM predictions 
               WHERE sector_accuracy IS NOT NULL 
               ORDER BY date DESC LIMIT 30"""
        ).fetchall()
    
    sector_scores = {}
    for row in rows:
        try:
            sa = json.loads(row["sector_accuracy"]) if isinstance(row["sector_accuracy"], str) else row["sector_accuracy"]
            if not sa:
                continue
            for sector, data in sa.items():
                if sector not in sector_scores:
                    sector_scores[sector] = {"correct": 0, "total": 0}
                sector_scores[sector]["total"] += 1
                if data.get("correct"):
                    sector_scores[sector]["correct"] += 1
        except (json.JSONDecodeError, TypeError):
            continue
    
    # Base weights (Nifty 50 approximate sector weights)
    nifty_weights = {
        "Banking": 1.5, "Energy": 1.4, "IT": 1.2, "Fintech": 1.2,
        "Healthcare": 1.1, "Manufacturing": 1.1, "FMCG": 1.0,
        "Startup": 0.9, "Retail": 0.8, "Other": 0.7
    }
    
    for sector, scores in sector_scores.items():
        if scores["total"] >= 5:
            accuracy = scores["correct"] / scores["total"]
            base = nifty_weights.get(sector, 1.0)
            # Adjust: high accuracy → boost weight, low accuracy → reduce
            adjustment = (accuracy - 0.5) * 0.6  # -0.3 to +0.3 range
            new_weight = round(max(0.5, min(2.0, base + adjustment)), 2)
            update_dynamic_weight("sector", sector, new_weight, "backtest_calibration")
            print(f"    {sector}: {base} → {new_weight} (accuracy: {accuracy:.0%})")


if __name__ == "__main__":
    run_backtest()
