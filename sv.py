import os
import json
import time
from datetime import datetime

import requests
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
SYMBOL = "BTCUSDT"
BASE_URL = "https://fapi.binance.com"

TELEGRAM_BOT_TOKEN = "8299632218:AAGJwtvLMtJj69Jewdv3H9tL2RCfO0VvVUY"
TELEGRAM_CHAT_ID = "6060782678"

CHECK_INTERVAL_SECONDS = 60
STATE_FILE = "signal_state.json"
SIGNALS_FILE = "signals.json"

# Bộ lọc signal
RR_MIN = 1.5
ATR_SL_MULTIPLIER = 1.2
MIN_VOLUME_RATIO = 1.05

RSI_BUY_MIN = 45
RSI_BUY_MAX = 62
RSI_SELL_MIN = 38
RSI_SELL_MAX = 55

# Chỉ gửi signal nếu confidence >= mức này
MIN_CONFIDENCE_TO_SEND = 68

# Nếu true: đóng lệnh ngay khi chạm TP1/TP2/TP3
# Nếu false: có thể sửa tiếp logic trailing / partial
CLOSE_ON_FIRST_TP_HIT = True

# =========================================================
# HELPERS
# =========================================================
def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def load_json_file(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_state():
    return load_json_file(STATE_FILE, {})

def save_state(state: dict):
    save_json_file(STATE_FILE, state)

def load_signals():
    return load_json_file(SIGNALS_FILE, [])

def save_signals(signals):
    save_json_file(SIGNALS_FILE, signals)

def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID \
       or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN" \
       or TELEGRAM_CHAT_ID == "YOUR_TELEGRAM_CHAT_ID":
        log("Telegram chưa cấu hình.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }

    try:
        r = requests.post(url, data=payload, timeout=15)
        if r.status_code != 200:
            log(f"Telegram lỗi: {r.status_code} - {r.text}")
    except Exception as e:
        log(f"Lỗi gửi Telegram: {e}")

# =========================================================
# BINANCE DATA
# =========================================================
def get_klines(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)

    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    ]
    for c in numeric_cols:
        df[c] = df[c].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df

def get_mark_price(symbol: str) -> float:
    url = f"{BASE_URL}/fapi/v1/premiumIndex"
    params = {"symbol": symbol}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    return safe_float(data.get("markPrice"))

def get_open_interest(symbol: str) -> float:
    url = f"{BASE_URL}/fapi/v1/openInterest"
    params = {"symbol": symbol}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    return safe_float(data.get("openInterest"))

def get_funding_rate(symbol: str) -> float:
    url = f"{BASE_URL}/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": 1}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data:
        return 0.0
    return safe_float(data[-1].get("fundingRate"))

# =========================================================
# INDICATORS
# =========================================================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)

    df["rsi14"] = rsi(df["close"], 14)

    macd_line, signal_line, hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist

    df["atr14"] = atr(df, 14)
    df["vol_sma20"] = df["volume"].rolling(20).mean()

    return df

# =========================================================
# MARKET STRUCTURE
# =========================================================
def detect_h1_trend(df_h1: pd.DataFrame) -> str:
    last = df_h1.iloc[-1]
    prev = df_h1.iloc[-2]

    bullish_structure = last["close"] > prev["high"] or (
        last["close"] > last["ema20"] > last["ema50"] > last["ema200"]
    )

    bearish_structure = last["close"] < prev["low"] or (
        last["close"] < last["ema20"] < last["ema50"] < last["ema200"]
    )

    if last["close"] > last["ema50"] and last["close"] > last["ema200"] and bullish_structure:
        return "UP"
    if last["close"] < last["ema50"] and last["close"] < last["ema200"] and bearish_structure:
        return "DOWN"
    return "SIDEWAYS"

def recent_levels(df: pd.DataFrame, lookback: int = 48):
    recent = df.tail(lookback)
    support = recent["low"].min()
    resistance = recent["high"].max()
    return support, resistance

def volume_ok(last_row: pd.Series) -> bool:
    if pd.isna(last_row["vol_sma20"]) or last_row["vol_sma20"] == 0:
        return False
    return last_row["volume"] >= last_row["vol_sma20"] * MIN_VOLUME_RATIO

def funding_bias(funding: float) -> str:
    if funding > 0.0008:
        return "LONG_CROWDED"
    if funding < -0.0008:
        return "SHORT_CROWDED"
    return "NEUTRAL"

# =========================================================
# SIGNAL LOGIC
# =========================================================
def generate_signal(df_h1: pd.DataFrame, df_m15: pd.DataFrame, mark_price: float,
                    funding: float, oi_now: float, oi_prev: float):
    trend = detect_h1_trend(df_h1)
    h1_support, h1_resistance = recent_levels(df_h1, lookback=72)

    m15 = df_m15.copy()
    last = m15.iloc[-1]
    prev = m15.iloc[-2]

    oi_change_pct = 0.0
    if oi_prev > 0:
        oi_change_pct = ((oi_now - oi_prev) / oi_prev) * 100.0

    crowd = funding_bias(funding)

    near_ema20 = abs(last["close"] - last["ema20"]) <= last["atr14"] * 0.5
    near_ema50 = abs(last["close"] - last["ema50"]) <= last["atr14"] * 0.5
    pullback_zone = near_ema20 or near_ema50

    macd_up = last["macd_hist"] > prev["macd_hist"] and last["macd_hist"] > -5
    macd_down = last["macd_hist"] < prev["macd_hist"] and last["macd_hist"] < 5

    bullish_candle = last["close"] > last["open"] and last["close"] > prev["high"]
    bearish_candle = last["close"] < last["open"] and last["close"] < prev["low"]

    vol_is_ok = volume_ok(last)

    dist_to_res = h1_resistance - mark_price
    dist_to_sup = mark_price - h1_support

    if trend == "UP":
        if pullback_zone \
           and RSI_BUY_MIN <= last["rsi14"] <= RSI_BUY_MAX \
           and macd_up \
           and bullish_candle \
           and vol_is_ok \
           and dist_to_res > last["atr14"] * RR_MIN:

            confidence = 70
            reasons = [
                "H1 trend tăng",
                "M15 pullback về EMA20/EMA50",
                "RSI vùng hồi đẹp",
                "MACD histogram tăng",
                "Nến xác nhận breakout",
                "Volume xác nhận"
            ]

            if oi_change_pct > 0.5:
                confidence += 5
                reasons.append("Open Interest tăng hỗ trợ xu hướng")

            if crowd == "LONG_CROWDED":
                confidence -= 8
                reasons.append("Funding dương mạnh, long đang đông")
            elif crowd == "SHORT_CROWDED":
                confidence += 4
                reasons.append("Funding âm, dễ squeeze lên")

            entry = mark_price
            sl = min(last["low"], prev["low"]) - last["atr14"] * ATR_SL_MULTIPLIER
            risk = entry - sl

            if risk <= 0:
                return None

            tp1 = entry + risk * 1.0
            tp2 = entry + risk * 1.5
            tp3 = min(entry + risk * 2.0, h1_resistance)

            rr = (tp2 - entry) / risk

            if rr >= RR_MIN:
                return {
                    "signal": "BUY",
                    "trend": trend,
                    "entry": round(entry, 2),
                    "sl": round(sl, 2),
                    "tp1": round(tp1, 2),
                    "tp2": round(tp2, 2),
                    "tp3": round(tp3, 2),
                    "rr": round(rr, 2),
                    "confidence": max(min(confidence, 95), 1),
                    "reasons": reasons,
                    "funding": funding,
                    "oi_change_pct": oi_change_pct,
                    "support": round(h1_support, 2),
                    "resistance": round(h1_resistance, 2),
                }

    if trend == "DOWN":
        if pullback_zone \
           and RSI_SELL_MIN <= last["rsi14"] <= RSI_SELL_MAX \
           and macd_down \
           and bearish_candle \
           and vol_is_ok \
           and dist_to_sup > last["atr14"] * RR_MIN:

            confidence = 70
            reasons = [
                "H1 trend giảm",
                "M15 pullback về EMA20/EMA50",
                "RSI vùng hồi xuống đẹp",
                "MACD histogram giảm",
                "Nến xác nhận breakdown",
                "Volume xác nhận"
            ]

            if oi_change_pct > 0.5:
                confidence += 5
                reasons.append("Open Interest tăng hỗ trợ xu hướng")

            if crowd == "SHORT_CROWDED":
                confidence -= 8
                reasons.append("Funding âm mạnh, short đang đông")
            elif crowd == "LONG_CROWDED":
                confidence += 4
                reasons.append("Funding dương, dễ squeeze xuống")

            entry = mark_price
            sl = max(last["high"], prev["high"]) + last["atr14"] * ATR_SL_MULTIPLIER
            risk = sl - entry

            if risk <= 0:
                return None

            tp1 = entry - risk * 1.0
            tp2 = entry - risk * 1.5
            tp3 = max(entry - risk * 2.0, h1_support)

            rr = (entry - tp2) / risk

            if rr >= RR_MIN:
                return {
                    "signal": "SELL",
                    "trend": trend,
                    "entry": round(entry, 2),
                    "sl": round(sl, 2),
                    "tp1": round(tp1, 2),
                    "tp2": round(tp2, 2),
                    "tp3": round(tp3, 2),
                    "rr": round(rr, 2),
                    "confidence": max(min(confidence, 95), 1),
                    "reasons": reasons,
                    "funding": funding,
                    "oi_change_pct": oi_change_pct,
                    "support": round(h1_support, 2),
                    "resistance": round(h1_resistance, 2),
                }

    return None

# =========================================================
# SIGNAL STORAGE / WINRATE
# =========================================================
def create_signal_record(sig: dict):
    signal_id = f"{SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sig['signal']}"
    return {
        "id": signal_id,
        "symbol": SYMBOL,
        "signal": sig["signal"],
        "entry": sig["entry"],
        "sl": sig["sl"],
        "tp1": sig["tp1"],
        "tp2": sig["tp2"],
        "tp3": sig["tp3"],
        "rr": sig["rr"],
        "confidence": sig["confidence"],
        "status": "OPEN",
        "result": None,
        "tp_hit": None,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "closed_at": None
    }

def append_signal(sig: dict):
    signals = load_signals()
    signals.append(create_signal_record(sig))
    save_signals(signals)

def update_open_signals(current_price: float):
    signals = load_signals()
    changed = False
    closed_results = []

    for s in signals:
        if s["status"] != "OPEN":
            continue

        side = s["signal"]
        sl = float(s["sl"])
        tp1 = float(s["tp1"])
        tp2 = float(s["tp2"])
        tp3 = float(s["tp3"])

        if side == "BUY":
            if current_price <= sl:
                s["status"] = "CLOSED"
                s["result"] = "LOSE"
                s["tp_hit"] = 0
                s["closed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
                closed_results.append(s)

            elif current_price >= tp3:
                s["status"] = "CLOSED"
                s["result"] = "WIN"
                s["tp_hit"] = 3
                s["closed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
                closed_results.append(s)

            elif current_price >= tp2 and CLOSE_ON_FIRST_TP_HIT:
                s["status"] = "CLOSED"
                s["result"] = "WIN"
                s["tp_hit"] = 2
                s["closed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
                closed_results.append(s)

            elif current_price >= tp1 and CLOSE_ON_FIRST_TP_HIT:
                s["status"] = "CLOSED"
                s["result"] = "WIN"
                s["tp_hit"] = 1
                s["closed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
                closed_results.append(s)

        elif side == "SELL":
            if current_price >= sl:
                s["status"] = "CLOSED"
                s["result"] = "LOSE"
                s["tp_hit"] = 0
                s["closed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
                closed_results.append(s)

            elif current_price <= tp3:
                s["status"] = "CLOSED"
                s["result"] = "WIN"
                s["tp_hit"] = 3
                s["closed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
                closed_results.append(s)

            elif current_price <= tp2 and CLOSE_ON_FIRST_TP_HIT:
                s["status"] = "CLOSED"
                s["result"] = "WIN"
                s["tp_hit"] = 2
                s["closed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
                closed_results.append(s)

            elif current_price <= tp1 and CLOSE_ON_FIRST_TP_HIT:
                s["status"] = "CLOSED"
                s["result"] = "WIN"
                s["tp_hit"] = 1
                s["closed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
                closed_results.append(s)

    if changed:
        save_signals(signals)

    return closed_results

def calculate_stats():
    signals = load_signals()

    total = len(signals)
    open_count = sum(1 for s in signals if s["status"] == "OPEN")
    closed = [s for s in signals if s["status"] == "CLOSED"]

    wins = sum(1 for s in closed if s["result"] == "WIN")
    losses = sum(1 for s in closed if s["result"] == "LOSE")

    buy_closed = [s for s in closed if s["signal"] == "BUY"]
    sell_closed = [s for s in closed if s["signal"] == "SELL"]

    buy_wins = sum(1 for s in buy_closed if s["result"] == "WIN")
    sell_wins = sum(1 for s in sell_closed if s["result"] == "WIN")

    tp1_hits = sum(1 for s in closed if s.get("tp_hit") == 1)
    tp2_hits = sum(1 for s in closed if s.get("tp_hit") == 2)
    tp3_hits = sum(1 for s in closed if s.get("tp_hit") == 3)

    closed_count = len(closed)
    winrate = (wins / closed_count * 100) if closed_count > 0 else 0.0
    buy_winrate = (buy_wins / len(buy_closed) * 100) if len(buy_closed) > 0 else 0.0
    sell_winrate = (sell_wins / len(sell_closed) * 100) if len(sell_closed) > 0 else 0.0

    return {
        "total_signals": total,
        "open_signals": open_count,
        "closed_signals": closed_count,
        "wins": wins,
        "losses": losses,
        "winrate": round(winrate, 2),
        "buy_winrate": round(buy_winrate, 2),
        "sell_winrate": round(sell_winrate, 2),
        "tp1_hits": tp1_hits,
        "tp2_hits": tp2_hits,
        "tp3_hits": tp3_hits
    }

def format_stats_message(stats: dict) -> str:
    return (
        f"📊 <b>BTC SIGNAL STATS</b>\n\n"
        f"Total signals: {stats['total_signals']}\n"
        f"Open: {stats['open_signals']}\n"
        f"Closed: {stats['closed_signals']}\n"
        f"Wins: {stats['wins']}\n"
        f"Losses: {stats['losses']}\n"
        f"Winrate: <b>{stats['winrate']}%</b>\n"
        f"BUY winrate: {stats['buy_winrate']}%\n"
        f"SELL winrate: {stats['sell_winrate']}%\n"
        f"TP1 hits: {stats['tp1_hits']}\n"
        f"TP2 hits: {stats['tp2_hits']}\n"
        f"TP3 hits: {stats['tp3_hits']}"
    )

# =========================================================
# DUPLICATE FILTER
# =========================================================
def is_duplicate_signal(state: dict, sig: dict) -> bool:
    last_sig = state.get("last_signal")
    if not last_sig:
        return False

    same_side = last_sig.get("signal") == sig.get("signal")
    entry_close = abs(last_sig.get("entry", 0) - sig.get("entry", 0)) <= 80
    recent = (time.time() - state.get("last_signal_ts", 0)) < 60 * 45

    return same_side and entry_close and recent

# =========================================================
# FORMAT TELEGRAM
# =========================================================
def format_signal_message(sig: dict) -> str:
    reasons_text = "\n".join([f"• {r}" for r in sig["reasons"]])

    return (
        f"🚨 <b>{SYMBOL} INTRADAY SIGNAL</b>\n\n"
        f"📌 <b>Direction:</b> {sig['signal']}\n"
        f"📈 <b>Trend H1:</b> {sig['trend']}\n"
        f"🎯 <b>Entry:</b> {sig['entry']}\n"
        f"🛑 <b>SL:</b> {sig['sl']}\n"
        f"✅ <b>TP1:</b> {sig['tp1']}\n"
        f"✅ <b>TP2:</b> {sig['tp2']}\n"
        f"✅ <b>TP3:</b> {sig['tp3']}\n"
        f"⚖️ <b>RR:</b> {sig['rr']}\n"
        f"🧠 <b>Confidence:</b> {sig['confidence']}%\n\n"
        f"📊 <b>H1 Support:</b> {sig['support']}\n"
        f"📊 <b>H1 Resistance:</b> {sig['resistance']}\n"
        f"💸 <b>Funding:</b> {sig['funding']:.6f}\n"
        f"📦 <b>OI Change:</b> {sig['oi_change_pct']:.2f}%\n\n"
        f"🔎 <b>Reasons:</b>\n{reasons_text}\n\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

def format_closed_signal_message(s: dict) -> str:
    status_emoji = "✅" if s["result"] == "WIN" else "❌"
    return (
        f"{status_emoji} <b>{SYMBOL} SIGNAL CLOSED</b>\n\n"
        f"Direction: {s['signal']}\n"
        f"Entry: {s['entry']}\n"
        f"SL: {s['sl']}\n"
        f"TP1: {s['tp1']}\n"
        f"TP2: {s['tp2']}\n"
        f"TP3: {s['tp3']}\n"
        f"Result: <b>{s['result']}</b>\n"
        f"TP Hit: {s['tp_hit']}\n"
        f"Created: {s['created_at']}\n"
        f"Closed: {s['closed_at']}"
    )

# =========================================================
# MAIN
# =========================================================
def run_once():
    log("Đang lấy dữ liệu...")

    mark_price = get_mark_price(SYMBOL)

    closed_results = update_open_signals(mark_price)
    for item in closed_results:
        send_telegram(format_closed_signal_message(item))
        log(f"Signal đóng: {item['id']} -> {item['result']}")

    df_h1 = get_klines(SYMBOL, "1h", 300)
    df_m15 = get_klines(SYMBOL, "15m", 300)

    df_h1 = add_indicators(df_h1)
    df_m15 = add_indicators(df_m15)

    funding = get_funding_rate(SYMBOL)

    oi_prev = get_open_interest(SYMBOL)
    time.sleep(2)
    oi_now = get_open_interest(SYMBOL)

    signal = generate_signal(df_h1, df_m15, mark_price, funding, oi_now, oi_prev)

    state = load_state()

    if signal:
        log(f"Có signal: {signal['signal']} | Entry={signal['entry']} | Confidence={signal['confidence']}%")

        if signal["confidence"] >= MIN_CONFIDENCE_TO_SEND:
            if not is_duplicate_signal(state, signal):
                append_signal(signal)
                send_telegram(format_signal_message(signal))

                state["last_signal"] = signal
                state["last_signal_ts"] = time.time()
                save_state(state)

                log("Đã lưu signal và gửi Telegram.")
            else:
                log("Signal trùng gần đây, bỏ qua.")
        else:
            log(f"Signal có nhưng confidence thấp ({signal['confidence']}%), bỏ qua.")
    else:
        log("Không có signal đẹp.")

    stats = calculate_stats()
    log(
        f"Stats | Total={stats['total_signals']} | Open={stats['open_signals']} | "
        f"Closed={stats['closed_signals']} | Win={stats['wins']} | Lose={stats['losses']} | "
        f"Winrate={stats['winrate']}%"
    )

def main():
    log("Bot BTC Intraday Signal Full đang chạy...")
    send_telegram("🤖 Bot BTC Intraday Signal Full đã khởi động.")

    while True:
        try:
            run_once()
        except Exception as e:
            err = f"❌ Bot lỗi: {e}"
            log(err)
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()