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

GOLD_API_KEY = "goldapi-clvgyxsmmie3e2d-io"

TELEGRAM_BOT_TOKEN = "8299632218:AAGJwtvLMtJj69Jewdv3H9tL2RCfO0VvVUY"
TELEGRAM_CHAT_ID = "-1003815900287"

CHECK_INTERVAL_SECONDS = 60
STATE_FILE = "signal_state.json"
SIGNALS_FILE = "signals.json"

LAST_STATUS_TIME = 0
LAST_DAILY_REPORT = None

# Bộ lọc signal
RR_MIN = 1.5
ATR_SL_MULTIPLIER = 1.2
MIN_VOLUME_RATIO = 1.05

RSI_BUY_MIN = 45
RSI_BUY_MAX = 62
RSI_SELL_MIN = 38
RSI_SELL_MAX = 55

MIN_CONFIDENCE_TO_SEND = 68
CLOSE_ON_FIRST_TP_HIT = True


# =========================================================
# HELPERS
# =========================================================

def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


def send_telegram(text: str):

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }

    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        log(f"Telegram error {e}")


# =========================================================
# GOLD PRICE
# =========================================================

def get_xau_price():

    url = "https://www.goldapi.io/api/XAU/USD"

    headers = {
        "x-access-token": GOLD_API_KEY,
        "Content-Type": "application/json"
    }

    try:

        r = requests.get(url, headers=headers, timeout=10)

        if r.status_code != 200:
            return None

        data = r.json()

        return float(data["price"])

    except:
        return None


# =========================================================
# BINANCE DATA
# =========================================================

def get_klines(symbol, interval, limit=300):

    url = f"{BASE_URL}/fapi/v1/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    r = requests.get(url, params=params)

    data = r.json()

    df = pd.DataFrame(data)

    df.columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "q1", "q2", "q3", "q4", "q5"
    ]

    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df


def get_mark_price(symbol):

    url = f"{BASE_URL}/fapi/v1/premiumIndex"

    r = requests.get(url, params={"symbol": symbol})

    return float(r.json()["markPrice"])


def get_open_interest(symbol):

    url = f"{BASE_URL}/fapi/v1/openInterest"

    r = requests.get(url, params={"symbol": symbol})

    return float(r.json()["openInterest"])


def get_funding_rate(symbol):

    url = f"{BASE_URL}/fapi/v1/fundingRate"

    r = requests.get(url, params={"symbol": symbol, "limit": 1})

    data = r.json()

    if not data:
        return 0

    return float(data[-1]["fundingRate"])


# =========================================================
# INDICATORS
# =========================================================

def ema(series, period):

    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period).mean()

    avg_loss = loss.ewm(alpha=1/period).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def atr(df, period=14):

    high_low = df["high"] - df["low"]

    high_close = (df["high"] - df["close"].shift()).abs()

    low_close = (df["low"] - df["close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period).mean()


def add_indicators(df):

    df["ema20"] = ema(df["close"], 20)

    df["ema50"] = ema(df["close"], 50)

    df["ema200"] = ema(df["close"], 200)

    df["rsi"] = rsi(df["close"])

    df["atr"] = atr(df)

    return df


# =========================================================
# TREND ARROW
# =========================================================

def trend_arrow(df):

    if df.iloc[-1]["ema20"] > df.iloc[-1]["ema50"]:
        return "🟢 ↑"

    elif df.iloc[-1]["ema20"] < df.iloc[-1]["ema50"]:
        return "🔴 ↓"

    return "⚪ →"


# =========================================================
# STATS
# =========================================================

def calculate_stats():

    if not os.path.exists(SIGNALS_FILE):
        return {"wins":0,"losses":0,"total":0,"winrate":0}

    with open(SIGNALS_FILE) as f:
        signals = json.load(f)

    wins = sum(1 for s in signals if s["result"]=="WIN")

    losses = sum(1 for s in signals if s["result"]=="LOSE")

    total = wins + losses

    winrate = (wins/total*100) if total>0 else 0

    return {
        "wins":wins,
        "losses":losses,
        "total":total,
        "winrate":round(winrate,2)
    }


# =========================================================
# STATUS MESSAGE
# =========================================================

def send_status(btc_price, btc_arrow, xau_price):

    stats = calculate_stats()

    text = (
        "🤖 <b>JustinCuaFX BOT</b>\n\n"

        "🔎 Market Scan\n\n"

        f"BTCUSDT: {btc_arrow} {btc_price}\n"

        f"XAUUSD: {xau_price}\n\n"

        "📊 Stats\n"

        f"Trades: {stats['total']}\n"

        f"Wins: {stats['wins']} | Loss: {stats['losses']}\n"

        f"Winrate: <b>{stats['winrate']}%</b>\n\n"

        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    send_telegram(text)


# =========================================================
# DAILY REPORT
# =========================================================

def send_daily_report():

    stats = calculate_stats()

    text = (
        "📊 DAILY REPORT\n\n"

        f"Trades: {stats['total']}\n"

        f"Wins: {stats['wins']}\n"

        f"Loss: {stats['losses']}\n"

        f"Winrate: {stats['winrate']}%"
    )

    send_telegram(text)


# =========================================================
# MAIN LOOP
# =========================================================

def run_once():

    global LAST_STATUS_TIME
    global LAST_DAILY_REPORT

    btc_price = get_mark_price(SYMBOL)

    df = get_klines(SYMBOL,"15m")

    df = add_indicators(df)

    arrow = trend_arrow(df)

    xau_price = get_xau_price()

    if time.time() - LAST_STATUS_TIME > 60:

        send_status(round(btc_price,2),arrow,xau_price)

        LAST_STATUS_TIME = time.time()


    now = datetime.now()

    if now.hour == 23 and now.minute == 59:

        if LAST_DAILY_REPORT != now.date():

            send_daily_report()

            LAST_DAILY_REPORT = now.date()


# =========================================================
# START BOT
# =========================================================

def main():

    log("Bot JustinCuaFX started")

    send_telegram("🤖 JustinCuaFX BOT started")

    while True:

        try:

            run_once()

        except Exception as e:

            log(e)

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":

    main()
