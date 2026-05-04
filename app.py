import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta

# ================= PASSWORD =================
APP_PASSWORD = "2301"

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("## 🔐 Secure Access")
        pwd = st.text_input("Enter Password", type="password")

        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        elif pwd:
            st.error("Incorrect password")

        st.stop()

check_password()

st.set_page_config(page_title="EURUSD Engine", layout="wide")

PAIRS = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X"
]

# ================= DATA =================
@st.cache_data(ttl=60)
def fetch_data(pair, interval, period):
    df = yf.download(pair, interval=interval, period=period, progress=False)

    if df is None or df.empty:
        return None

    # 🔥 FIX: flatten multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })

    df = df.dropna()

    return df

# ================= INDICATORS =================
def indicators(df):

    if df is None or df.empty:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

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
        "adx": ta.trend.adx(high, low, close, 14)
    }

# ================= STRATEGY =================
def get_signal(df_h1, df_m5, df_m1):

    # VALIDATION
    if df_h1 is None or df_m5 is None or df_m1 is None:
        return None, "DATA ERROR"

    if len(df_h1) < 100 or len(df_m5) < 100 or len(df_m1) < 50:
        return None, "INSUFFICIENT DATA"

    i_h1 = indicators(df_h1)
    i_m5 = indicators(df_m5)

    if i_h1 is None or i_m5 is None:
        return None, "INDICATOR ERROR"

    ema20_m5 = i_m5["ema20"].iloc[-1]
    ema50_m5 = i_m5["ema50"].iloc[-1]
    price_m5 = df_m5["Close"].iloc[-1]
    
    if ema20_m5 > ema50_m5 and price_m5 > ema20_m5:
        trend = "BUY"
    elif ema20_m5 < ema50_m5 and price_m5 < ema20_m5:
        trend = "SELL"
    else:
        return None, "NO CLEAR TREND"

    # ADX FILTER (RELAXED)
    adx = i_m5["adx"].iloc[-1]
    if adx < 15:
        return None, "WEAK TREND"

    # PULLBACK
    price = float(df_m5["Close"].iloc[-1])
    ema20 = float(i_m5["ema20"].iloc[-1])
    rsi = float(i_m5["rsi"].iloc[-1])

    distance = abs(price - ema20) / ema20

    pullback_valid = distance < 0.006
    momentum_move = distance >= 0.006  # NEW
    
    # Allow either pullback OR strong momentum
    if not (pullback_valid or momentum_move):
        return None, "NO SETUP"

    # SOFT TREND CHECK
    m5_trend = "BUY" if i_m5["ema20"].iloc[-1] > i_m5["ema50"].iloc[-1] else "SELL"

    if m5_trend != trend:
        st.write("⚠️ M5 counter-trend")

    i_m1 = indicators(df_m1)
    if i_m1 is None:
        return None, "M1 INDICATOR ERROR"
    
    ema20_m1 = float(i_m1["ema20"].iloc[-1])
    
    # ENTRY (EMA REACTION)
    last = df_m1.iloc[-1]
    prev = df_m1.iloc[-2]
    
    if trend == "BUY":
        if prev["Low"] <= ema20_m1 and last["Close"] > prev["Close"]:
            return "BUY", "EMA REJECTION"
        
        body = abs(last["Close"] - last["Open"])
        prev_body = abs(prev["Close"] - prev["Open"])
    
        if last["Close"] > last["Open"] and body > prev_body:
            return "BUY", "MOMENTUM"
    
    if trend == "SELL":
        if prev["High"] >= ema20_m1 and last["Close"] < prev["Close"]:
            return "SELL", "EMA REJECTION"
    
        # 2. NEW: Momentum continuation
        body = abs(last["Close"] - last["Open"])
        prev_body = abs(prev["Close"] - prev["Open"])
    
        if last["Close"] < last["Open"] and body > prev_body:
            return "SELL", "MOMENTUM"
    
    return None, "WAIT ENTRY"

# ================= LOGGER =================
if "trades" not in st.session_state:
    st.session_state.trades = []

# ================= UI =================
st.title("EUR/USD Precision Engine")

if st.button("Scan Market"):

    for pair in PAIRS:

        df_h1 = fetch_data(pair, "1h", "30d")
        df_m5 = fetch_data(pair, "5m", "5d")
        df_m1 = fetch_data(pair, "1m", "1d")

        signal, status = get_signal(df_h1, df_m5, df_m1)

        st.subheader(f"📊 {pair}")

        if signal:
            now = datetime.now().replace(second=0, microsecond=0)
        
            if now.minute % 2 == 0:
                entry = now + timedelta(minutes=2)
            else:
                entry = now + timedelta(minutes=1)
        
            expiry = entry + timedelta(minutes=2)
        
            st.success(f"Signal: {signal} ({status})")
            st.write(f"Entry: {entry.strftime('%H:%M')}")
            st.write(f"Expiry: {expiry.strftime('%H:%M')}")
        else:
            st.warning(status)

        st.markdown("---")
            
        if st.button(f"Log Trade {pair}"):
            st.session_state.trades.append({
                "time": entry.strftime('%H:%M'),
                "signal": signal,
                "pair": pair,
                "result": None
            })

        else:
            # OPTIONAL: comment this out if too noisy
            # st.warning(f"{pair}: {status}")
            pass

    now = datetime.now().replace(second=0, microsecond=0)

    if now.minute % 2 == 0:
        next_entry = now + timedelta(minutes=2)
    else:
        next_entry = now + timedelta(minutes=1)

    st.write("Next Possible Entry Time:", next_entry.strftime("%H:%M"))

# ================= TRADE LOG =================
st.subheader("Trade Journal")

for i, trade in enumerate(st.session_state.trades):
    col1, col2, col3 = st.columns(3)

    col1.write(trade["time"])
    col2.write(trade["signal"])

    result = col3.selectbox(
        "Result",
        ["Pending", "Win", "Loss"],
        key=f"result_{i}"
    )

    st.session_state.trades[i]["result"] = result

# ================= STATS =================
wins = sum(1 for t in st.session_state.trades if t["result"] == "Win")
losses = sum(1 for t in st.session_state.trades if t["result"] == "Loss")
total = wins + losses

win_rate = (wins / total * 100) if total > 0 else 0

st.subheader("Performance")
st.write(f"Trades: {total}")
st.write(f"Wins: {wins}")
st.write(f"Losses: {losses}")
st.write(f"Win Rate: {win_rate:.2f}%")
