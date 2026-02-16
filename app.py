import streamlit as st 
import yfinance as yf
import ta
import pandas as pd
import time
from datetime import datetime, timedelta 
import requests
from bs4 import BeautifulSoup
import pytz

# ================= TELEGRAM =================
BOT_TOKEN = "8527341776:AAGII0r_Badcp8sToimRbCzGPOP5lwnyZhY"
CHAT_ID = "8516458781"

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Work-Station", layout="wide")
if "trade_active" not in st.session_state:
    st.session_state.trade_active = False

if "last_signal" not in st.session_state:
    st.session_state.last_signal = None

if "pair_cooldown" not in st.session_state:
    st.session_state.pair_cooldown = {}

if "result_checked" not in st.session_state:
    st.session_state.result_checked = False

# ================= PASSWORD =================
APP_PASSWORD = "2026"

def check_password():
    if "auth" not in st.session_state:
        st.session_state.auth = False
    if not st.session_state.auth:
        st.markdown("<h2 style='text-align:center'>🔐 Secure Access</h2>", unsafe_allow_html=True)
        pwd = st.text_input("Password", type="password")
        if pwd == APP_PASSWORD:
            st.session_state.auth = True
            st.rerun()
        elif pwd:
            st.error("Incorrect password")
        st.stop()

check_password()
tv_symbol = None

# ================= STYLES =================
st.markdown("""
<style>
body { background:#0b0f14; color:white; }
.block { background:#121722; padding:22px; border-radius:16px; margin-bottom:18px; }
.center { text-align:center; }
.signal-buy { color:#22c55e; font-size:60px; font-weight:800; }
.signal-sell { color:#ef4444; font-size:60px; font-weight:800; }
.signal-wait { color:#9ca3af; font-size:48px; font-weight:700; }
.metric { color:#9ca3af; margin-top:6px; }
.small { font-size:13px; color:#9ca3af; }
</style>
""", unsafe_allow_html=True)

# ================= MARKETS =================
CURRENCIES = {
    "EUR/JPY": "EURJPY=X",
    "EUR/GBP": "EURGBP=X",
    "USD/JPY": "JPY=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/CAD": "AUDCAD=X",
    "AUD/CHF": "AUDCHF=X",
    "GBP/AUD": "GBPAUD=X",
    "EUR/USD": "EURUSD=X",
    "AUD/JPY": "AUDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "EUR/CHF": "EURCHF=X",
    "GBP/CHF": "GBPCHF=X",
    "CHF/JPY": "CHFJPY=X",
    "EUR/AUD": "EURAUD=X",
    "GBP/JPY": "GBPJPY=X",
    "EUR/CAD": "EURCAD=X",
    "USD/CAD": "CAD=X",
    "GBP/CAD": "GBPCAD=X",
    "USD/CHF": "CHF=X",
    "CAD/JPY": "CADJPY=X"
}

CRYPTO = {
    "BTC/USD":"BTC-USD","ETH/USD":"ETH-USD","BNB/USD":"BNB-USD",
    "SOL/USD":"SOL-USD","XRP/USD":"XRP-USD","ADA/USD":"ADA-USD",
    "DOGE/USD":"DOGE-USD","AVAX/USD":"AVAX-USD","DOT/USD":"DOT-USD",
    "LINK/USD":"LINK-USD","MATIC/USD":"MATIC-USD"
}

COMMODITIES = {
    "Gold":"GC=F","Silver":"SI=F","Crude Oil":"CL=F",
    "Brent Oil":"BZ=F","Natural Gas":"NG=F",
    "Copper":"HG=F","Corn":"ZC=F","Wheat":"ZW=F"
}

@st.cache_data(ttl=60)
def fetch(symbol, interval, period):
    return yf.download(symbol, interval=interval, period=period, progress=False)

def indicators(df):
    if df is None or df.empty or "Close" not in df.columns:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # 🔒 Force Series (not DataFrame)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]

    close = close.astype(float)
    high = high.astype(float)
    low = low.astype(float)

    return {
        "close": close,
        "ema20": ta.trend.ema_indicator(close, 20),
        "ema50": ta.trend.ema_indicator(close, 50),
        "ema100": ta.trend.ema_indicator(close, 100),
        "rsi": ta.momentum.rsi(close, 14),
        "macd": ta.trend.macd_diff(close),
        "atr": ta.volatility.average_true_range(high, low, close, 14),
        "adx": ta.trend.adx(high, low, close, 14)
    }

# ================= PRICE STRUCTURE (LIVE DATA) =================

def detect_structure_from_price(df, indicators):
    """
    Improved trend detection using:
    - Higher High / Higher Low logic
    - EMA alignment
    - ADX strength filter
    """

    if df is None or df.empty or indicators is None:
        return "RANGE"

    closes = df["Close"]
    highs = df["High"]
    lows = df["Low"]

    # 🔒 Ensure Series
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    if isinstance(highs, pd.DataFrame):
        highs = highs.iloc[:, 0]
    if isinstance(lows, pd.DataFrame):
        lows = lows.iloc[:, 0]

    closes = closes.astype(float)
    highs = highs.astype(float)
    lows = lows.astype(float)

    if len(closes) < 30:
        return "RANGE"

    # --- Structure (HH + HL) ---
    recent_high = float(highs.iloc[-10:].max())
    prior_high  = float(highs.iloc[-20:-10].max())

    recent_low  = float(lows.iloc[-10:].min())
    prior_low   = float(lows.iloc[-20:-10].min())

    # --- EMA Alignment ---
    ema20 = indicators["ema20"].iloc[-1]
    ema50 = indicators["ema50"].iloc[-1]
    price = closes.iloc[-1]

    # --- Strength ---
    adx = indicators["adx"].iloc[-1]

    # --- Bullish Trend ---
    if (recent_high > prior_high and
        recent_low > prior_low and
        ema20 > ema50 and
        price > ema20 and
        adx > 20):
        return "BULLISH"

    # --- Bearish Trend ---
    if (recent_high < prior_high and
        recent_low < prior_low and
        ema20 < ema50 and
        price < ema20 and
        adx > 20):
        return "BEARISH"

    return "RANGE"

def detect_phase_from_price(df, structure):
    """
    Continuation vs pullback using price direction.
    """
    if df is None or df.empty or len(df) < 6:
        return "NO_TRADE"

    closes = df["Close"]
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]

    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-5])

    if structure == "BULLISH":
        return "CONTINUATION" if last >= prev else "PULLBACK"

    if structure == "BEARISH":
        return "CONTINUATION" if last <= prev else "PULLBACK"

    return "NO_TRADE"

def detect_regime(df, indicators):
    if indicators is None:
        return "UNKNOWN"

    atr = indicators["atr"].iloc[-1]
    atr_avg = indicators["atr"].rolling(20).mean().iloc[-1]
    adx = indicators["adx"].iloc[-1]

    if adx > 25 and atr > atr_avg:
        return "STRONG_TREND"

    if adx > 20:
        return "TREND"

    if atr < atr_avg * 0.8:
        return "LOW_VOLATILITY"

    return "RANGE"

def detect_market_personality(df, indicators, sr):
    if indicators is None:
        return "UNKNOWN"

    adx = indicators["adx"].iloc[-1]
    atr = indicators["atr"].iloc[-1]
    atr_avg = indicators["atr"].rolling(20).mean().iloc[-1]

    # Count SR hits in last 30 candles
    sr_hits = 0
    closes = indicators["close"].iloc[-30:]

    recent_low  = closes.min()
    recent_high = closes.max()

    for price in closes:
        if abs(price - recent_low) / price < 0.002:
            sr_hits += 1
        if abs(price - recent_high) / price < 0.002:
            sr_hits += 1

    # --- TREND DOMINANT ---
    if adx > 25 and atr > atr_avg:
        return "TREND_DOMINANT"

    # --- MEAN REVERTING ---
    if adx < 20 and sr_hits >= 5:
        return "MEAN_REVERTING"

    # --- RANGE ---
    if 15 <= adx <= 22:
        return "RANGE_BOUND"

    return "MIXED"

def movement_quality(indicators):

    close = indicators["close"]

    recent = close.iloc[-5:]

    move = abs(recent.iloc[-1] - recent.iloc[0])
    noise = recent.diff().abs().sum()

    if noise == 0:
        return "CHAOTIC"

    smoothness = move / noise

    if smoothness > 0.55:
        return "CLEAN"

    return "CHAOTIC"

def is_range_market(indicators, sr_local):

    if indicators is None:
        return False

    adx = indicators["adx"].iloc[-1]
    atr = indicators["atr"].iloc[-1]
    atr_avg = indicators["atr"].rolling(20).mean().iloc[-1]

    # Quiet + stable = range candidate
    if adx < 22 and atr < atr_avg * 1.2:
        if sr_local["support"] or sr_local["resistance"]:
            return True

    return False

def detect_pullback_quality(df, indicators, structure):

    if indicators is None:
        return "NONE"

    close = indicators["close"]
    ema20 = indicators["ema20"]
    ema50 = indicators["ema50"]
    adx = indicators["adx"].iloc[-1]

    price = close.iloc[-1]

    # Pullback must stay inside trend
    if structure == "BULLISH":

        near_ema = (
            abs(price - ema20.iloc[-1]) / price < 0.003 or
            abs(price - ema50.iloc[-1]) / price < 0.003
        )

        if near_ema and adx > 18:
            return "HEALTHY"

        if adx < 15:
            return "WEAK"

    if structure == "BEARISH":

        near_ema = (
            abs(price - ema20.iloc[-1]) / price < 0.003 or
            abs(price - ema50.iloc[-1]) / price < 0.003
        )

        if near_ema and adx > 18:
            return "HEALTHY"

        if adx < 15:
            return "WEAK"

    return "NONE"

def classify_market_state(structure, phase):

    if structure == "BULLISH" and phase == "CONTINUATION":
        return "BUY", "Uptrend continuation", 80

    if structure == "BEARISH" and phase == "CONTINUATION":
        return "SELL", "Downtrend continuation", 80

    # Pullbacks are ignored for beginners
    if phase == "PULLBACK":
        return "WAIT", "Pullback ignored for safety", 0

    return "WAIT", "No clear structure", 0

def pair_is_on_cooldown(pair):

    if pair not in st.session_state.pair_cooldown:
        return False

    last_time = st.session_state.pair_cooldown[pair]

    # 15 minute cooldown
    cooldown = timedelta(minutes=15)

    return datetime.now() - last_time < cooldown

def scan_all_markets():

    best_trade = None
    best_score = 0
    
    for asset, symbol in CURRENCIES.items():
    
        if pair_is_on_cooldown(asset):
            continue
    
        df = fetch(symbol, "5m", "2d")
        i = indicators(df)

        if df is None or i is None:
            continue

        structure = detect_structure_from_price(df, i)
        phase = detect_phase_from_price(df, structure)
        regime = detect_regime(df, i)

        # Support / Resistance
        sr_local = {"support": False, "resistance": False}
        recent_low  = i["close"].rolling(20).min().iloc[-1]
        recent_high = i["close"].rolling(20).max().iloc[-1]
        price = i["close"].iloc[-1]

        if abs(price - recent_low) / price < 0.002:
            sr_local["support"] = True
        if abs(price - recent_high) / price < 0.002:
            sr_local["resistance"] = True

        personality = detect_market_personality(df, i, sr_local)
        movement = movement_quality(i)
        range_mode = is_range_market(i, sr_local)
        pullback = detect_pullback_quality(df, i, structure)

        if personality == "TREND_DOMINANT":
            signal, reason, confidence = classify_market_state(structure, phase)

        elif personality == "MEAN_REVERTING":
            if sr_local["support"]:
                signal, reason, confidence = "BUY", "Mean reversion bounce", 70
            elif sr_local["resistance"]:
                signal, reason, confidence = "SELL", "Mean reversion rejection", 70
            else:
                signal, reason, confidence = "WAIT", "", 0
        
        elif personality == "RANGE":
            if sr_local["support"]:
                signal, reason, confidence = "BUY", "Range support bounce", 75
            elif sr_local["resistance"]:
                signal, reason, confidence = "SELL", "Range resistance rejection", 75
            else:
                signal, reason, confidence = "WAIT", "", 0
        
        else:
            signal, reason, confidence = classify_market_state(structure, phase)

        if signal in ["BUY", "SELL"]:
        
            score = confidence   # 👈 MUST COME FIRST

            # Reward healthy pullbacks
            if pullback == "HEALTHY":
                score += 10
            
            # Avoid weak pullbacks
            if pullback == "WEAK":
                score -= 15

            # Reward clean build (simulates M3/M4 confirmation)
            if movement == "CLEAN":
                score += 10
        
            # Penalize spike behaviour
            if movement == "CHAOTIC":
                score -= 10

            adx = i["adx"].iloc[-1]
            atr = i["atr"].iloc[-1]
            atr_avg = i["atr"].rolling(20).mean().iloc[-1]

            # 1. Reward stable trend (not extreme)
            if 22 <= adx <= 35:
                score += 15

            # 2. Penalize exhaustion
            if adx > 40:
                score -= 10

            # 3. Reward controlled volatility
            if atr < atr_avg * 1.2:
                score += 10

            # 4. Penalize chaos
            if atr > atr_avg * 1.6:
                score -= 10

            # 5. Reward respectful trend behaviour
            if personality == "TREND_DOMINANT":
                score += 5

            # 6. Penalize mean reversion markets
            if personality == "MEAN_REVERTING":
                score -= 5
            
            # 🎯 Reward clean range bounce
            if range_mode:
                score += 10

            if score > best_score:

                last_close = df.index[-1].to_pydatetime()
                minute = (last_close.minute // 5 + 1) * 5
                entry_time = last_close.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minute)
                expiry_time = entry_time + timedelta(minutes=5)

                best_score = score
                best_trade = {
                    "asset": asset,
                    "signal": signal,
                    "confidence": score,
                    "personality": personality,
                    "structure": structure,
                    "phase": phase,
                    "regime": regime,
                    "entry": entry_time.strftime("%H:%M"),
                    "expiry": expiry_time.strftime("%H:%M")
                }

    return best_trade

# ================= HEADER =================
st.markdown("""
<div class="block">
<h1>Malagna</h1>
<div class="metric">20-Rule Dominant Engine • All Markets • True M5</div>
</div>
""", unsafe_allow_html=True)

if st.button("Scan Market 🔍"):

    best = scan_all_markets()

    if best:

        signal_class = {
            "BUY": "signal-buy",
            "SELL": "signal-sell"
        }[best["signal"]]

        st.markdown(f"""
        <div class="block center">
            <div class="{signal_class}">{best['signal']}</div>
            <div class="metric">Best Opportunity: {best['asset']}</div>
            <div class="metric"><b>Confidence:</b> {best['confidence']}%</div>
        <div class="small">
            Structure: {best['structure']} • 
            Phase: {best['phase']} • 
            Regime: {best['regime']} • 
            Personality: {best['personality']}<br><br>
           🟢 Entry: {best['entry']}<br>
           🔴 Expiry: {best['expiry']}
        </div>
        """, unsafe_allow_html=True)

# ================= USER NOTE =================
st.markdown("""
<div class="block small">
⚠️ <b>Important Note</b><br>
Signals are generated using <b>M5 price action only</b>.<br>
Always confirm with your own analysis, trend context, and risk management.<br>
This tool supports decisions — it does not replace them.
</div>
""", unsafe_allow_html=True)

# ================= DATA =================

def forex_factory_red_news(currencies, window_minutes=30):
    """
    Returns True if high-impact (red) news is within ±window_minutes
    for the given currencies.
    """
    try:
        url = "https://www.forexfactory.com/calendar"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        now = datetime.utcnow().replace(tzinfo=pytz.UTC)

        for row in soup.select("tr.calendar__row"):
            impact = row.select_one(".impact span")
            currency = row.select_one(".currency")
            time_cell = row.select_one(".time")

            if not impact or not currency or not time_cell:
                continue

            # High-impact only
            if "high" not in impact.get("class", []):
                continue

            cur = currency.text.strip()
            if cur not in currencies:
                continue

            time_text = time_cell.text.strip()
            if time_text in ["All Day", "Tentative", ""]:
                continue

            event_time = datetime.strptime(time_text, "%H:%M")
            event_time = event_time.replace(
                year=now.year, month=now.month, day=now.day,
                tzinfo=pytz.UTC
            )

            diff = abs((event_time - now).total_seconds()) / 60
            if diff <= window_minutes:
                return True

    except Exception:
        pass

    return False

def send_telegram(message):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    try:
        requests.post(url, data=payload)
    except:
        pass

def time_until_entry(entry_time):

    now = datetime.now()
    entry = datetime.strptime(entry_time, "%H:%M")

    entry = entry.replace(
        year=now.year,
        month=now.month,
        day=now.day
    )

    return (entry - now).total_seconds()

def auto_bot():

    if not st.session_state.trade_active:

        best = scan_all_markets()

        if best:

            wait_seconds = time_until_entry(best["entry"]) - 120

            if wait_seconds > 0:
                time.sleep(wait_seconds)

            msg = f"""
📊 SIGNAL

Pair: {best['asset']}
Timeframe: M5
Signal: {best['signal']}

🟢 Entry: {best['entry']}
🔴 Expiry: {best['expiry']}
"""

            send_telegram(msg)

            st.session_state.trade_active = True
            st.session_state.result_checked = False
            st.session_state.last_signal = best

def evaluate_trade(pair, signal, entry_time, expiry_time):

    symbol = CURRENCIES.get(pair)

    df = fetch(symbol, "5m", "2d")

    if df is None or df.empty:
        return None

    # Convert entry/expiry into UTC timestamps
    today = datetime.utcnow().date()

    entry_dt = pd.Timestamp(f"{today} {entry_time}", tz="UTC")
    expiry_dt = pd.Timestamp(f"{today} {expiry_time}", tz="UTC")

    # Find nearest candle
    entry_idx = df.index.get_indexer([entry_dt], method="nearest")[0]
    expiry_idx = df.index.get_indexer([expiry_dt], method="nearest")[0]

    entry_price = float(df.iloc[entry_idx]["Close"])
    expiry_price = float(df.iloc[expiry_idx]["Close"])

    if signal == "BUY":
        return "WIN" if expiry_price > entry_price else "LOSS"

    if signal == "SELL":
        return "WIN" if expiry_price < entry_price else "LOSS"

    return None

def check_result():

    if st.session_state.trade_active and not st.session_state.result_checked:

        expiry_time = st.session_state.last_signal["expiry"]

        now = datetime.now()
        expiry = datetime.strptime(expiry_time, "%H:%M").replace(
            year=now.year,
            month=now.month,
            day=now.day
        )

        wait_seconds = (expiry - now).total_seconds()

        if wait_seconds > 0:
            time.sleep(wait_seconds)

        outcome = evaluate_trade(
            st.session_state.last_signal["asset"],
            st.session_state.last_signal["signal"],
            st.session_state.last_signal["entry"],
            st.session_state.last_signal["expiry"]
        )

        if outcome == "WIN":
            send_telegram("✅ WIN")

        else:
            send_telegram("⚠️ LOSS → M1")

            time.sleep(300)

            retry = evaluate_trade(
                st.session_state.last_signal["asset"],
                st.session_state.last_signal["signal"],
                st.session_state.last_signal["entry"],
                st.session_state.last_signal["expiry"]
            )

            if retry == "WIN":
                send_telegram("✅ M1 WIN")
            else:
                send_telegram("❌ M1 LOSS")

        # 🔒 Lock result
        st.session_state.result_checked = True
        st.session_state.trade_active = False
        
        st.session_state.pair_cooldown[
            st.session_state.last_signal["asset"]
        ] = datetime.now()

auto_bot()
check_result()
