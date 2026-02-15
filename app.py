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

def scan_all_markets():

    best_trade = None
    best_score = 0

    for asset, symbol in CURRENCIES.items():

        df = fetch(symbol, "5m", "2d")
        i = indicators(df)

        if df is None or i is None:
            continue

        structure = detect_structure_from_price(df, i)
        phase = detect_phase_from_price(df, structure)
        regime = detect_regime(df, i)

        sr_local = {
            "support": False,
            "resistance": False
        }

        recent_low  = i["close"].rolling(20).min().iloc[-1]
        recent_high = i["close"].rolling(20).max().iloc[-1]
        price = i["close"].iloc[-1]

        if abs(price - recent_low) / price < 0.002:
            sr_local["support"] = True
        if abs(price - recent_high) / price < 0.002:
            sr_local["resistance"] = True

        personality = detect_market_personality(df, i, sr_local)

        # Generate signal
        if personality == "TREND_DOMINANT":
            signal, reason, confidence = classify_market_state(structure, phase)

        elif personality == "MEAN_REVERTING":
            if sr_local["support"]:
                signal, reason, confidence = "BUY", "Mean reversion bounce", 70
            elif sr_local["resistance"]:
                signal, reason, confidence = "SELL", "Mean reversion rejection", 70
            else:
                signal, reason, confidence = "WAIT", "", 0
        else:
            signal, reason, confidence = classify_market_state(structure, phase)

        if signal in ["BUY", "SELL"]:
            score = confidence

            # Boost strong trend setups
            if personality == "TREND_DOMINANT" and regime == "STRONG_TREND":
                score += 10

            if score > best_score:
                best_score = score
                best_trade = {
                    "asset": asset,
                    "signal": signal,
                    "confidence": score,
                    "personality": personality,
                    "structure": structure,
                    "phase": phase
                }

    return best_trade
    
market = st.radio("Market", ["Currencies","Crypto","Commodities","Stocks"], horizontal=True)

if market == "Currencies":
    asset = st.selectbox(
    "Pair",
    list(CURRENCIES.keys()),
    key="currency_pair_select"
)
    symbol = CURRENCIES[asset]

elif market == "Crypto":
    asset = st.selectbox("Crypto", list(CRYPTO.keys()))
    symbol = CRYPTO[asset]

elif market == "Commodities":
    asset = st.selectbox("Commodity", list(COMMODITIES.keys()))
    symbol = COMMODITIES[asset]

else:
    asset = st.text_input("Stock ticker (e.g. AAPL, TSLA, MSFT)").upper()
    symbol = asset
# ================= HEADER =================
st.markdown("""
<div class="block">
<h1>Malagna</h1>
<div class="metric">20-Rule Dominant Engine • All Markets • True M5</div>
</div>
""", unsafe_allow_html=True)

if st.button("Scan Best Trade 🔍"):
    best = scan_all_markets()

    if best:
        st.success(f"""
BEST TRADE FOUND 🚀

Pair: {best['asset']}
Signal: {best['signal']}
Confidence: {best['confidence']}%
Personality: {best['personality']}
Structure: {best['structure']}
Phase: {best['phase']}
""")
    else:
        st.warning("No strong setup found.")

# ================= USER NOTE =================
st.markdown("""
<div class="block small">
⚠️ <b>Important Note</b><br>
Signals are generated using <b>M5 price action only</b>.<br>
Always confirm with your own analysis, trend context, and risk management.<br>
This tool supports decisions — it does not replace them.
</div>
""", unsafe_allow_html=True)

# ================= TRADINGVIEW SYMBOL =================
TV_SYMBOLS = {}

# FX
for k in CURRENCIES.keys():
    TV_SYMBOLS[k] = f"FX:{k.replace('/','')}"

# Crypto
for k in CRYPTO.keys():
    TV_SYMBOLS[k] = f"BINANCE:{k.replace('/','').replace('USD','USDT')}"

# Commodities
TV_SYMBOLS.update({
    "Gold": "COMEX:GC1!",
    "Silver": "COMEX:SI1!",
    "Crude Oil": "NYMEX:CL1!",
    "Brent Oil": "ICEEUR:BRN1!",
    "Natural Gas": "NYMEX:NG1!",
    "Copper": "COMEX:HG1!",
    "Corn": "CBOT:ZC1!",
    "Wheat": "CBOT:ZW1!"
})

if market == "Stocks" and asset:
    tv_symbol = f"NASDAQ:{asset}"
else:
    tv_symbol = TV_SYMBOLS.get(asset)

# ================= TRADINGVIEW CHART =================
if tv_symbol:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.components.v1.html(
        f"""
        <iframe
            src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval=5&theme=dark&style=1&locale=en"
            width="100%"
            height="420"
            frameborder="0"
            allowtransparency="true"
            scrolling="no">
        </iframe>
        """,
        height=430,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ================= DATA =================
@st.cache_data(ttl=60)
def fetch(symbol, interval, period):
    return yf.download(symbol, interval=interval, period=period, progress=False)

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

def extract_currencies(asset):
    if "/" in asset:
        return asset.split("/")
    return []

data_5m  = fetch(symbol, "5m", "5d")

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

i5 = indicators(data_5m)

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

def classify_market_state(structure, phase):

    if structure == "BULLISH" and phase == "CONTINUATION":
        return "BUY", "Uptrend continuation", 80

    if structure == "BEARISH" and phase == "CONTINUATION":
        return "SELL", "Downtrend continuation", 80

    # Pullbacks are ignored for beginners
    if phase == "PULLBACK":
        return "WAIT", "Pullback ignored for safety", 0

    return "WAIT", "No clear structure", 0

# ================= SHORT-TERM MOMENTUM (EMA20 SLOPE) =================
ema20_slope = 0

if i5 is not None:
    ema20 = i5.get("ema20")
    if ema20 is not None and len(ema20.dropna()) >= 3:
        ema20_slope = ema20.iloc[-1] - ema20.iloc[-3]

# ================= MARKET ACTIVITY =================
market_active = True
activity_note = ""

if i5:
    recent_move = abs(i5["close"].iloc[-1] - i5["close"].iloc[-6])
    avg_move = i5["close"].diff().abs().rolling(10).mean().iloc[-1]

    if avg_move > 0 and recent_move < avg_move * 0.6:
        market_active = False
        activity_note = "Low momentum"

# ================= SUPPORT / RESISTANCE (SIMPLE & SAFE) =================
sr = {
    "support": False,
    "resistance": False
}
if i5:
    recent_low  = i5["close"].rolling(20).min().iloc[-1]
    recent_high = i5["close"].rolling(20).max().iloc[-1]
    price = i5["close"].iloc[-1]

    if abs(price - recent_low) / price < 0.002:
        sr["support"] = True
    if abs(price - recent_high) / price < 0.002:
        sr["resistance"] = True

# ================= CANDLE TYPE (M5) =================
def candle_type(df):
    if df is None or df.empty or len(df) < 2:
        return "NEUTRAL"
    try:
        o = float(df["Open"].iloc[-1])
        c = float(df["Close"].iloc[-1])
        h = float(df["High"].iloc[-1])
        l = float(df["Low"].iloc[-1])
    except Exception:
        return "NEUTRAL"

    body = abs(c - o)
    full = h - l

    if full == 0:
        return "NEUTRAL"

    ratio = body / full
    if ratio >= 0.6:
        return "IMPULSE"
    elif ratio <= 0.3:
        return "NEUTRAL"
    else:
        return "REJECTION"

candle = candle_type(data_5m.iloc[:-1])

# ================= SIGNAL EVALUATION =================

# ================= SIGNAL EVALUATION =================

structure = detect_structure_from_price(data_5m, i5)
phase = detect_phase_from_price(data_5m, structure)
regime = detect_regime(data_5m, i5)
personality = detect_market_personality(data_5m, i5, sr)

if personality == "TREND_DOMINANT":
    signal, reason, confidence = classify_market_state(structure, phase)

elif personality == "MEAN_REVERTING":
    if sr["support"]:
        signal, reason, confidence = "BUY", "Mean reversion bounce", 70
    elif sr["resistance"]:
        signal, reason, confidence = "SELL", "Mean reversion rejection", 70
    else:
        signal, reason, confidence = "WAIT", "No reversal edge", 0

elif personality == "RANGE_BOUND":
    if sr["support"]:
        signal, reason, confidence = "BUY", "Range support", 65
    elif sr["resistance"]:
        signal, reason, confidence = "SELL", "Range resistance", 65
    else:
        signal, reason, confidence = "WAIT", "No range edge", 0

else:
    signal, reason, confidence = classify_market_state(structure, phase)

# ================= SIGNAL MEMORY =================
if "last_signal" not in st.session_state:
    st.session_state.last_signal = None

if signal == st.session_state.last_signal and signal != "WAIT":
    signal = "WAIT"
    reason = "Awaiting confirmation"
    confidence = max(60, confidence - 10)

st.session_state.last_signal = signal

# ================= NEWS FILTER (FOREX FACTORY) =================
news_note = ""
currencies = extract_currencies(asset)

if market == "Currencies" and currencies:
    if forex_factory_red_news(currencies):
        confidence = max(60, confidence - 20)
        news_note = "⚠️ High-impact news nearby"

        if confidence < 65:
            signal = "WAIT"
if news_note:
    reason = f"{reason} • {news_note}"

if signal == "WAIT" and not market_active and activity_note:
    reason = f"{reason} • {activity_note}"

# ================= ENTRY & EXPIRY (✅ ADDED) =================
entry_time = None
expiry_time = None

if signal in ["BUY","SELL"] and not data_5m.empty:
    last_close = data_5m.index[-1].to_pydatetime()
    minute = (last_close.minute // 5 + 1) * 5
    entry_time = last_close.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minute)
    expiry_time = entry_time + timedelta(minutes=5)

# ================= DISPLAY =================
signal_class = {
    "BUY": "signal-buy",
    "SELL": "signal-sell",
    "WAIT": "signal-wait"
}[signal]

signal_class = {
    "BUY": "signal-buy",
    "SELL": "signal-sell",
    "WAIT": "signal-wait"
}[signal]

st.markdown(f"""
<div class="block center">
  <div class="{signal_class}">{signal}</div>
  <div class="metric">{asset} · {market}</div>

  {"<div class='metric'><b>Confidence:</b> " + str(confidence) + "%</div>" if signal != "WAIT" else ""}
  {"<div class='metric'><b>Entry:</b> " + entry_time.strftime('%H:%M') + "</div>" if entry_time else ""}
  {"<div class='metric'><b>Expiry:</b> " + expiry_time.strftime('%H:%M') + "</div>" if expiry_time else ""}

  <div class="small">{reason}</div>
  <div class="small">
    Structure (M5): {structure} • Phase: {phase} • Regime: {regime} • Candle: {candle}
  </div>
""", unsafe_allow_html=True)

