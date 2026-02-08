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

# ================= HEADER =================
st.markdown("""
<div class="block">
<h1>Malagna</h1>
<div class="metric">20-Rule Dominant Engine • All Markets • True M5</div>
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
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)

    return {
        "close": close,
        "ema20": ta.trend.ema_indicator(close, 20),
        "ema50": ta.trend.ema_indicator(close, 50),
        "ema200": ta.trend.ema_indicator(close, 200),
        "rsi": ta.momentum.rsi(close, 14),
        "macd": ta.trend.macd_diff(close)
    }

i5 = indicators(data_5m)

# ================= PRICE STRUCTURE (LIVE DATA) =================

def detect_structure_from_price(df):
    """
    Uses swing logic on CLOSE prices.
    FIXED: ensures scalar values (no pandas ambiguity)
    """
    if df is None or len(df) < 20:
        return "RANGE"

    closes = df["Close"]

    # 🔒 FORCE Series (not DataFrame)
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]

    closes = closes.astype(float)

    recent_low  = float(closes.iloc[-10:].min())
    prior_low   = float(closes.iloc[-20:-10].min())

    recent_high = float(closes.iloc[-10:].max())
    prior_high  = float(closes.iloc[-20:-10].max())

    if recent_low > prior_low:
        return "BULLISH"

    if recent_high < prior_high:
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

def classify_market_state(structure, phase):

    if structure == "BULLISH" and phase == "CONTINUATION":
        return "BUY", "Uptrend continuation", 80

    if structure == "BEARISH" and phase == "CONTINUATION":
        return "SELL", "Downtrend continuation", 80

    if structure == "BULLISH" and phase == "PULLBACK":
        return "SELL", "Pullback in uptrend", 70

    if structure == "BEARISH" and phase == "PULLBACK":
        return "BUY", "Pullback in downtrend", 70

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

def get_swings(df, lookback=3):
    highs, lows = [], []

    if df is None or df.empty:
        return highs, lows

    for i in range(lookback, len(df) - lookback):
        window_highs = df["High"].iloc[i-lookback:i+lookback+1]
        window_lows  = df["Low"].iloc[i-lookback:i+lookback+1]

        cur_high = float(df["High"].iloc[i])
        cur_low  = float(df["Low"].iloc[i])

        if cur_high == float(window_highs.max()):
            highs.append((i, cur_high))

        if cur_low == float(window_lows.min()):
            lows.append((i, cur_low))

    return highs, lows

def detect_trend(highs, lows):
    if len(highs) < 2 or len(lows) < 2:
        return "RANGE"

    h1, h2 = highs[-2][1], highs[-1][1]
    l1, l2 = lows[-2][1], lows[-1][1]

    if h2 > h1 and l2 > l1:
        return "UPTREND"

    if h2 < h1 and l2 < l1:
        return "DOWNTREND"

    return "RANGE"

def break_of_structure(df, trend, highs, lows):
    if not highs or not lows:
        return False

    close = float(df["Close"].iloc[-1])

    if trend == "UPTREND":
        return close > highs[-1][1]

    if trend == "DOWNTREND":
        return close < lows[-1][1]

    return False

def pullback_zone(price, trend, highs, lows, tolerance=0.003):
    try:
        price = float(price)

        if trend == "UPTREND" and isinstance(lows, list) and len(lows) > 0:
            last_low = float(lows[-1][1])
            return abs(price - last_low) / price < tolerance

        if trend == "DOWNTREND" and isinstance(highs, list) and len(highs) > 0:
            last_high = float(highs[-1][1])
            return abs(price - last_high) / price < tolerance

    except Exception:
        pass

    return False

def loss_of_momentum(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    body_last = abs(last["Close"] - last["Open"])
    body_prev = abs(prev["Close"] - prev["Open"])

    return body_last < body_prev * 0.6

def candle_confirmation(df):
    if df is None or len(df) < 3:
        return None

    try:
        c1 = df.iloc[-3]
        c2 = df.iloc[-2]
        c3 = df.iloc[-1]

        o1, c1_ = float(c1["Open"]), float(c1["Close"])
        o2, c2_ = float(c2["Open"]), float(c2["Close"])
        o3, c3_ = float(c3["Open"]), float(c3["Close"])

    except Exception:
        return None

    # ================= ENGULFING =================
    bullish_engulf = (
        c2_ < o2 and
        c3_ > o3 and
        c3_ > o2 and
        o3 < c2_
    )

    bearish_engulf = (
        c2_ > o2 and
        c3_ < o3 and
        o3 > c2_ and
        c3_ < o2
    )

    # ================= MORNING / EVENING STAR =================
    body1 = abs(c1_ - o1)
    body2 = abs(c2_ - o2)
    body3 = abs(c3_ - o3)

    morning_star = (
        c1_ < o1 and
        body2 < body1 * 0.5 and
        c3_ > o3
    )

    evening_star = (
        c1_ > o1 and
        body2 < body1 * 0.5 and
        c3_ < o3
    )

    if bullish_engulf or morning_star:
        return "BULLISH"

    if bearish_engulf or evening_star:
        return "BEARISH"

    return None

# ================= SIGNAL EVALUATION =================
highs, lows = get_swings(data_5m)
trend = detect_trend(highs, lows)
bos = break_of_structure(data_5m, trend, highs, lows)

price = float(data_5m["Close"].iloc[-1])
pullback = pullback_zone(price, trend, highs, lows)
momentum_lost = loss_of_momentum(data_5m)
candle_pa = candle_confirmation(data_5m)

signal = "WAIT"
reason = "No valid price action setup"
confidence = 0

has_bullish_candle = candle_pa == "BULLISH"
has_bearish_candle = candle_pa == "BEARISH"

# ================= TREND CONTINUATION =================
if bool(bos) and bool(momentum_lost):
    if trend == "UPTREND" and has_bullish_candle:
        signal = "BUY"
        reason = "Uptrend continuation after BOS"
        confidence = 85

    if trend == "DOWNTREND" and has_bearish_candle:
        signal = "SELL"
        reason = "Downtrend continuation after BOS"
        confidence = 85

# ================= PULLBACK ENTRY =================
elif bool(pullback) and bool(momentum_lost):
    if trend == "UPTREND" and has_bullish_candle:
        signal = "BUY"
        reason = "Uptrend pullback entry"
        confidence = 80

    if trend == "DOWNTREND" and has_bearish_candle:
        signal = "SELL"
        reason = "Downtrend pullback entry"
        confidence = 80

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

st.markdown(f"""
<div class="block center">
  <div class="{signal_class}">{signal}</div>
  <div class="metric">{asset} · {market}</div>

  {"<div class='metric'><b>Confidence:</b> " + str(confidence) + "%</div>" if signal != "WAIT" else ""}
  {"<div class='metric'><b>Entry:</b> " + entry_time.strftime('%H:%M') + "</div>" if entry_time else ""}
  {"<div class='metric'><b>Expiry:</b> " + expiry_time.strftime('%H:%M') + "</div>" if expiry_time else ""}

  <div class="small">{reason}</div>
  <div class="small">
    Trend (M5): {trend} • BOS: {bos} • Candle: {candle_pa}
  </div>
""", unsafe_allow_html=True)
