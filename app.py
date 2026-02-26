import streamlit as st 
import yfinance as yf
import ta
import pandas as pd
import time
from datetime import datetime, timedelta 
import requests
from bs4 import BeautifulSoup
import pytz

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Work-Station", layout="wide")

if "pair_cooldown" not in st.session_state:
    st.session_state.pair_cooldown = {}

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

def range_quality(indicators):

    adx = indicators["adx"].iloc[-1]
    rsi = indicators["rsi"].iloc[-1]

    # Good range = weak trend + oscillation
    if adx < 18 and (40 < rsi < 60):
        return "IDEAL"

    if adx < 22:
        return "OK"

    return "POOR"

def detect_pullback_quality(df, indicators, structure):

    if indicators is None:
        return "NONE"

    close = indicators["close"]
    ema20 = indicators["ema20"]
    ema50 = indicators["ema50"]
    adx = indicators["adx"]

    price = close.iloc[-1]

    # Near value zone
    near_value = (
        abs(price - ema20.iloc[-1]) / price < 0.004 or
        abs(price - ema50.iloc[-1]) / price < 0.004
    )

    # Momentum cooling
    adx_now = adx.iloc[-1]
    adx_prev = adx.iloc[-4]

    momentum_cooling = adx_now <= adx_prev

    # Pullback should NOT be explosive
    recent = close.iloc[-8:]
    move = abs(recent.iloc[-1] - recent.iloc[0])
    noise = recent.diff().abs().sum()

    smooth = (move / noise) < 0.55 if noise != 0 else False

    if near_value and momentum_cooling and smooth:
        return "HEALTHY"

    if near_value and not momentum_cooling:
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

def classify_market_environment(df, indicators):

    if df is None or indicators is None or len(df) < 120:
        return "TRANSITION"

    close = indicators["close"]
    ema20 = indicators["ema20"]
    ema50 = indicators["ema50"]
    ema100 = indicators["ema100"]
    adx = indicators["adx"]
    atr = indicators["atr"]

    ema_bull = ema20.iloc[-1] > ema50.iloc[-1] > ema100.iloc[-1]
    ema_bear = ema20.iloc[-1] < ema50.iloc[-1] < ema100.iloc[-1]

    trend_strength = adx.iloc[-1]

    atr_now = atr.iloc[-1]
    atr_mean = atr.rolling(50).mean().iloc[-1]

    expanding = atr_now > atr_mean * 1.3
    contracting = atr_now < atr_mean * 0.85

    closes = close.iloc[-60:]

    move = abs(closes.iloc[-1] - closes.iloc[0])
    noise = closes.diff().abs().sum()

    smoothness = move / noise if noise != 0 else 0

    directional = smoothness > 0.55
    choppy = smoothness < 0.35

    if (ema_bull or ema_bear) and trend_strength > 25 and directional:
        return "TREND"

    if trend_strength > 20 and expanding:
        return "EXPANSION"

    if trend_strength < 20 and contracting and choppy:
        return "RANGE"

    return "TRANSITION"

def detect_direction(indicators):

    ema20 = indicators["ema20"].iloc[-1]
    ema50 = indicators["ema50"].iloc[-1]
    ema100 = indicators["ema100"].iloc[-1]

    if ema20 > ema50 > ema100:
        return "BULLISH"

    if ema20 < ema50 < ema100:
        return "BEARISH"

    return "NEUTRAL"

def detect_trend_pullback(indicators, direction):

    close = indicators["close"]
    ema20 = indicators["ema20"]
    ema50 = indicators["ema50"]
    adx = indicators["adx"]

    price = close.iloc[-1]

    # --- VALUE ZONE ---
    near_ema20 = abs(price - ema20.iloc[-1]) / price < 0.0035
    near_ema50 = abs(price - ema50.iloc[-1]) / price < 0.0045

    in_value = near_ema20 or near_ema50

    # --- MOMENTUM COOLING ---
    adx_now = adx.iloc[-1]
    adx_prev = adx.iloc[-4]

    cooling = adx_now <= adx_prev

    # --- CONTROLLED MOVE ---
    recent = close.iloc[-5:]
    move = abs(recent.iloc[-1] - recent.iloc[0])
    noise = recent.diff().abs().sum()

    smooth = (move / noise) < 0.55 if noise != 0 else False

    if direction == "BULLISH":
        if price <= ema20.iloc[-1] and in_value and cooling and smooth:
            return True

    if direction == "BEARISH":
        if price >= ema20.iloc[-1] and in_value and cooling and smooth:
            return True

    return False

# ================= NEW DECISION BRAIN =================

def movement_reality(indicators):
    close = indicators["close"]
    recent = close.iloc[-6:]

    move = abs(recent.iloc[-1] - recent.iloc[0])
    noise = recent.diff().abs().sum()

    if noise == 0:
        return "CHAOTIC"

    smoothness = move / noise

    if smoothness > 0.6:
        return "CLEAN"

    if smoothness > 0.4:
        return "MODERATE"

    return "CHAOTIC"

def structural_bias(df):

    highs = df["High"]
    lows  = df["Low"]

    # 🔒 Ensure Series
    if isinstance(highs, pd.DataFrame):
        highs = highs.iloc[:, 0]
    if isinstance(lows, pd.DataFrame):
        lows = lows.iloc[:, 0]

    highs = highs.astype(float)
    lows  = lows.astype(float)

    if len(highs) < 20:
        return "NEUTRAL"

    recent_high = float(highs.iloc[-5:].max())
    prior_high  = float(highs.iloc[-20:-10].max())

    recent_low  = float(lows.iloc[-5:].min())
    prior_low   = float(lows.iloc[-20:-10].min())

    if recent_high > prior_high and recent_low > prior_low:
        return "BULLISH"

    if recent_high < prior_high and recent_low < prior_low:
        return "BEARISH"

    return "NEUTRAL"

def environment_strength(indicators):

    adx = indicators["adx"].iloc[-1]
    atr = indicators["atr"].iloc[-1]
    atr_avg = indicators["atr"].rolling(20).mean().iloc[-1]

    if adx > 25 and atr > atr_avg:
        return "STRONG"

    if adx > 18:
        return "MODERATE"

    return "WEAK"


def phase_timing(indicators, bias):

    close = indicators["close"]
    last = close.iloc[-1]
    prev = close.iloc[-4]

    if bias == "BULLISH":
        return "CONTINUATION" if last >= prev else "PULLBACK"

    if bias == "BEARISH":
        return "CONTINUATION" if last <= prev else "PULLBACK"

    return "NONE"

def scan_all_markets():

    best_trade = None
    best_score = 0
    
    for asset, symbol in CURRENCIES.items():

        if pair_is_on_cooldown(asset):
            continue
    
        df = fetch(symbol, "5m", "7d")
        i = indicators(df)

        movement = movement_reality(i)
        bias = structural_bias(df)
        env = environment_strength(i)
        phase = phase_timing(i, bias)
        
        signal = "WAIT"
        confidence = 0
        reason = "No alignment"
        
        # ================= DECISION STACK =================
        
        if bias != "NEUTRAL":
        
            # Clean movement allows full trading
            if movement == "CLEAN":
        
                if env in ["STRONG", "MODERATE"]:
        
                    if phase == "CONTINUATION":
                        signal = "BUY" if bias == "BULLISH" else "SELL"
                        confidence = 85
                        reason = "Clean continuation"
        
                    elif phase == "PULLBACK" and env == "STRONG":
                        signal = "BUY" if bias == "BULLISH" else "SELL"
                        confidence = 75
                        reason = "Pullback in strong trend"
        
            # Moderate movement = safer trades
            elif movement == "MODERATE":
        
                if env == "STRONG" and phase == "CONTINUATION":
                    signal = "BUY" if bias == "BULLISH" else "SELL"
                    confidence = 75
                    reason = "Moderate but strong regime"
        
            # Chaotic = still allow strong trends
            elif movement == "CHAOTIC":
        
                if env == "STRONG" and phase == "CONTINUATION":
                    signal = "BUY" if bias == "BULLISH" else "SELL"
                    confidence = 65
                    reason = "Strong trend despite noise"

        # ================= SUPPORT / RESISTANCE =================
        sr_local = {"support": False, "resistance": False}
        recent_low  = i["close"].rolling(20).min().iloc[-1]
        recent_high = i["close"].rolling(20).max().iloc[-1]
        price = i["close"].iloc[-1]

        if abs(price - recent_low) / price < 0.002:
            sr_local["support"] = True
        if abs(price - recent_high) / price < 0.002:
            sr_local["resistance"] = True

        # ================= MARKET STATE =================
        personality = detect_market_personality(df, i, sr_local)
        movement = movement_quality(i)
        range_mode = is_range_market(i, sr_local)
        pullback = "NONE"

        # ================= SIGNAL DECISION =================
        
        if state == "TREND":

            pullback_ready = detect_trend_pullback(i, direction)
        
            if direction == "BULLISH" and pullback_ready:
                signal, reason, confidence = "BUY", "Trend pullback entry", 85
        
            elif direction == "BEARISH" and pullback_ready:
                signal, reason, confidence = "SELL", "Trend pullback entry", 85
        
            else:
                signal, reason, confidence = "WAIT", "Waiting for pullback", 0

        elif state == "RANGE":
        
            rsi = i["rsi"].iloc[-1]
        
            if sr_local["support"] and rsi < 40:
                signal, reason, confidence = "BUY", "Range bounce", 75
        
            elif sr_local["resistance"] and rsi > 60:
                signal, reason, confidence = "SELL", "Range rejection", 75
        
            else:
                signal, reason, confidence = "WAIT", "Range middle", 0
        
        elif state == "EXPANSION":
        
            if direction == "BULLISH":
                signal, reason, confidence = "BUY", "Breakout momentum", 85
        
            elif direction == "BEARISH":
                signal, reason, confidence = "SELL", "Breakout momentum", 85
        
            else:
                signal, reason, confidence = "WAIT", "Weak expansion", 0
    
        elif state == "TRANSITION":
        
            rsi = i["rsi"].iloc[-1]
        
            if direction == "BULLISH" and rsi < 45:
                signal, reason, confidence = "BUY", "Early reversal forming", 70
        
            elif direction == "BEARISH" and rsi > 55:
                signal, reason, confidence = "SELL", "Early reversal forming", 70
        
            else:
                signal, reason, confidence = "WAIT", "No structure shift", 0

        # ================= SCORING =================
        if signal in ["BUY", "SELL"]:

            score = confidence

            if pullback == "HEALTHY":
                score += 10
            
            if pullback == "WEAK":
                score -= 15

            if movement == "CLEAN":
                score += 10
        
            if movement == "CHAOTIC":
                score -= 10

            adx = i["adx"].iloc[-1]
            atr = i["atr"].iloc[-1]
            atr_avg = i["atr"].rolling(20).mean().iloc[-1]

            if 22 <= adx <= 35:
                score += 15

            if adx > 40:
                score -= 10

            if atr < atr_avg * 1.2:
                score += 10

            if atr > atr_avg * 1.6:
                score -= 10

            if personality == "TREND_DOMINANT":
                score += 5

            if range_mode:
                score += 10

            # Range timing boost
            if personality in ["RANGE_BOUND", "MEAN_REVERTING"]:
                rsi = i["rsi"].iloc[-1]
                if signal == "BUY" and rsi < 35:
                    score += 10
                if signal == "SELL" and rsi > 65:
                    score += 10

            # ================= BEST TRADE =================
            if score > best_score:

                last_close = df.index[-1].to_pydatetime()
                minute = (last_close.minute // 5 + 1) * 5
                entry_time = last_close.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minute)
                expiry_time = entry_time + timedelta(minutes=5)

                best_score = score
                best_trade = {
                    "state": state,
                    "direction": direction,
                    "asset": asset,
                    "signal": signal,
                    "confidence": score,
                    "personality": personality,
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
    
        st.session_state.pair_cooldown[best["asset"]] = datetime.now()
    
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
                State: {best['state']} • 
                Direction: {best['direction']} • 
                Personality: {best['personality']}
                🟢 Entry: {best['entry']}<br>
                🔴 Expiry: {best['expiry']}
            </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("No valid trade found right now. Market may be in cooldown or low quality.")

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

