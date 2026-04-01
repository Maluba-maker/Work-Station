import streamlit as st 
import ta
import yfinance as yf
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
    "USD/JPY": "USDJPY=X",
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
    "USD/CAD": "USDCAD=X",
    "GBP/CAD": "GBPCAD=X",
    "USD/CHF": "USDCHF=X",
    "CAD/JPY": "CADJPY=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/NZD": "EURNZD=X",
    "GBP/NZD": "GBPNZD=X",
    "AUD/NZD": "AUDNZD=X",
    "EUR/SGD": "EURSGD=X",
    "GBP/SGD": "GBPSGD=X",
    "USD/SGD": "USDSGD=X",
    "USD/NOK": "USDNOK=X",
    "USD/SEK": "USDSEK=X",
    "EUR/NOK": "EURNOK=X",
    "EUR/SEK": "EURSEK=X"
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

    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False)

        if df is None or df.empty:
            return None

        df = df.dropna()

        return df

    except:
        return None

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

def detect_trend_pullback(indicators, direction):

    close = indicators["close"]
    ema20 = indicators["ema20"]
    rsi = indicators["rsi"]

    price = close.iloc[-1]
    ema = ema20.iloc[-1]
    rsi_now = rsi.iloc[-1]

    # Distance to EMA
    near_ema = abs(price - ema) / price < 0.005  # 0.5% tolerance

    if direction == "BULLISH":
        # Pullback = price near EMA and RSI cooled
        if price <= ema and rsi_now < 55:
            return True

    if direction == "BEARISH":
        if price >= ema and rsi_now > 45:
            return True

    return False

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
    cooldown = timedelta(minutes=5)

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
    near_ema20 = abs(price - ema20.iloc[-1]) / price < 0.0006
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

def detect_breakout(df):

    highs = df["High"]
    lows = df["Low"]
    close = df["Close"]

    if isinstance(highs, pd.DataFrame):
        highs = highs.iloc[:,0]
    if isinstance(lows, pd.DataFrame):
        lows = lows.iloc[:,0]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:,0]

    highs = highs.astype(float)
    lows = lows.astype(float)
    close = close.astype(float)

    resistance = float(highs.iloc[-20:-3].max())
    support = float(lows.iloc[-20:-3].min())

    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])

    # breakout must CLOSE above level
    if prev <= resistance and last > resistance:
        return "BREAKOUT_UP"

    if prev >= support and last < support:
        return "BREAKOUT_DOWN"

    return None

# ===== ADD THIS FUNCTION HERE =====

def detect_market_cycle(df, indicators):

    if df is None or indicators is None or len(df) < 60:
        return "UNKNOWN"

    close = indicators["close"]
    adx = indicators["adx"]
    atr = indicators["atr"]

    adx_now = adx.iloc[-1]
    adx_prev = adx.iloc[-5]

    atr_now = atr.iloc[-1]
    atr_avg = atr.rolling(30).mean().iloc[-1]

    highs = df["High"].iloc[-20:]
    lows = df["Low"].iloc[-20:]

    if isinstance(highs, pd.DataFrame):
        highs = highs.iloc[:,0]
    if isinstance(lows, pd.DataFrame):
        lows = lows.iloc[:,0]

    highs = highs.astype(float)
    lows = lows.astype(float)

    range_size = highs.max() - lows.min()
    price = close.iloc[-1]

    tight_range = (range_size / price) < 0.003

    atr_contracting = atr_now < atr_avg * 0.85
    atr_expanding = atr_now > atr_avg * 1.15

    if adx_now > 25 and not tight_range:
        return "TREND"

    if adx_now < 20 and atr_contracting and tight_range:
        return "CONSOLIDATION"

    if adx_now > adx_prev and adx_now < 25 and tight_range:
        return "PRE_BREAKOUT"

    if atr_expanding and adx_now > 20 and not tight_range:
        return "EXPANSION"

    return "TRANSITION"
    
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

def detect_momentum_and_breakout(df_m1, df_m2, i_m1, i_m2):
    
    signal = None
    confidence = 0
    reason = ""

    # ===== MOMENTUM BURST (PRIMARY) =====
    closes = i_m1["close"].iloc[-4:]
    
    diffs = closes.diff().dropna()
    
    bullish = all(diffs > 0)
    bearish = all(diffs < 0)
    
    body_strength = abs(closes.iloc[-1] - closes.iloc[0])
    
    if bullish and body_strength > closes.std() * 0.5:
        signal = "BUY"
        confidence = 85
        reason = "Momentum burst (M1)"
    
    elif bearish and body_strength > closes.std() * 0.5:
        signal = "SELL"
        confidence = 85
        reason = "Momentum burst (M1)"

    # ===== BREAKOUT SCALP (STRONGER PRIORITY) =====
    highs = df_m2["High"].iloc[-15:]
    lows = df_m2["Low"].iloc[-15:]
    
    resistance = highs.max()
    support = lows.min()
    
    price = df_m2["Close"].iloc[-1]

    if price > resistance:
        signal = "BUY"
        confidence = 90
        reason = "Breakout (M2)"
    
    elif price < support:
        signal = "SELL"
        confidence = 90
        reason = "Breakout (M2)"

    return signal, confidence, reason

def scan_all_markets():

    best_trade = None
    best_score = 0

    for asset, symbol in CURRENCIES.items():

        if pair_is_on_cooldown(asset):
            continue

        # ===== NEWS FILTER =====
        base, quote = asset.split("/")
        if forex_factory_red_news([base, quote]):
            continue

        # ===== HIGHER TIMEFRAME (H1) =====
        df_h1 = fetch(symbol, "1h", "30d")
        i_h1 = indicators(df_h1)

        if df_h1 is None or df_h1.empty or i_h1 is None:
            continue

        htf_direction = detect_direction(i_h1)

        if htf_direction == "NEUTRAL":
            continue

        # ===== M5 =====
        df = fetch(symbol, "5m", "3d")
        i = indicators(df)

        if df is None or df.empty or i is None:
            continue

        # ===== M1 (ENTRY CONFIRMATION) =====
        df_m1 = fetch(symbol, "1m", "1d")

        if df_m1 is None or df_m1.empty:
            continue
        i_m1 = indicators(df_m1)
        # 🔥 FIX: Flatten columns (yfinance issue)
        if isinstance(df_m1.columns, pd.MultiIndex):
            df_m1.columns = df_m1.columns.get_level_values(0)
        
        # 🔥 Ensure correct column names
        df_m1 = df_m1.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })
        
        # 🔥 Keep only required columns
        required_cols = ["Open", "High", "Low", "Close"]
        
        for col in required_cols:
            if col not in df_m1.columns:
                continue
        
        # Volume is optional
        if "Volume" not in df_m1.columns:
            df_m1["Volume"] = 0

        df_m1.index = pd.to_datetime(df_m1.index)
        
        # 🔥 Create M2 candles manually
        df_m2 = df_m1.resample("2min").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()
        
        i_m2 = indicators(df_m2)
        
        m2_direction = detect_direction(i_m2)
        m2_movement = movement_reality(i_m2)
        
        m5_direction = detect_direction(i)

        # Only skip if completely unclear
        adx = i["adx"].iloc[-1]
        if htf_direction == "NEUTRAL" and adx < 15:
            continue
            
        breakout = detect_breakout(df)
        movement = movement_reality(i)

        if movement == "CHAOTIC" and adx < 20:
            continue

        signal = None
        confidence = 0
        reason = ""

        # ===== NEW 2-MIN ENTRY ENGINE =====

        signal, confidence, reason = detect_momentum_and_breakout(df_m1, df_m2, i_m1, i_m2)
        
        # ===== OPTIONAL HTF FILTER (LIGHT FILTER ONLY) =====
        
        if signal:
            
            # Only reduce bad trades, don’t block everything
            if signal == "BUY" and htf_direction == "BEARISH":
                confidence -= 10
            
            elif signal == "SELL" and htf_direction == "BULLISH":
                confidence -= 10

       # ===== HTF CONTEXT ADJUSTMENT =====

        if signal:
        
            if signal == "BUY" and htf_direction == "BULLISH":
                confidence += 10
        
            elif signal == "SELL" and htf_direction == "BEARISH":
                confidence += 10
        
            else:
                confidence -= 10

        # ===== FINAL ENTRY + CONFIRMATION =====

        if signal:
            # ===== LIGHT CONFIRMATION (DON’T KILL SIGNALS) =====

            if m2_movement == "CHAOTIC":
                confidence -= 10
            
            # ===== ENTRY TIMING (CORRECT NEXT 2-MIN CANDLE) =====

            now = datetime.now().replace(second=0, microsecond=0)
            
            minute = now.minute
            
            # Find next 2-minute candle
            if minute % 2 == 0:
                entry_time = now + timedelta(minutes=2)
            else:
                entry_time = now + timedelta(minutes=1)
            
            # Expiry = 2 minutes after entry
            expiry_time = entry_time + timedelta(minutes=2)
        
            # ===== BEST TRADE SELECTION =====
            if confidence > best_score:
                best_score = confidence
                best_trade = {
                    "state": "MOMENTUM",
                    "direction": m5_direction,
                    "asset": asset,
                    "signal": signal,
                    "confidence": confidence,
                    "personality": reason,
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
