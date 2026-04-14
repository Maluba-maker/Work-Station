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

PAIR = "EURUSD=X"

# ================= DATA =================
@st.cache_data(ttl=60)
def fetch_data(interval, period):
    df = yf.download(PAIR, interval=interval, period=period, progress=False)

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

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # 🔥 FORCE SERIES (CRITICAL FIX)
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
    i_h1 = indicators(df_h1)
    i_m5 = indicators(df_m5)

    # TREND
    if i_h1["ema20"].iloc[-1] > i_h1["ema50"].iloc[-1] > i_h1["ema100"].iloc[-1]:
        trend = "BUY"
    elif i_h1["ema20"].iloc[-1] < i_h1["ema50"].iloc[-1] < i_h1["ema100"].iloc[-1]:
        trend = "SELL"
    else:
        return None, "NO TREND"

    # FILTER
    if i_m5["adx"].iloc[-1] < 20:
        return None, "WEAK MARKET"

    # ===== PULLBACK =====
    price = float(df_m5["Close"].iloc[-1])
    ema20 = float(i_m5["ema20"].iloc[-1])
    ema50 = float(i_m5["ema50"].iloc[-1])
    rsi = float(i_m5["rsi"].iloc[-1])
    
    in_zone = (
        abs(price - ema20)/price < 0.002
        or
        abs(price - ema50)/price < 0.002
    )

    in_zone = abs(price - ema20)/price < 0.003 or abs(price - ema50)/price < 0.003
    
    st.write("Trend:", trend)
    st.write("ADX:", i_m5["adx"].iloc[-1])
    st.write("RSI:", i_m5["rsi"].iloc[-1])
    st.write("Price:", price)
    st.write("EMA20:", ema20)
    st.write("EMA50:", ema50)
    st.write("In Pullback Zone:", in_zone)
    
    if trend == "BUY" and not (in_zone and rsi < 55):
        return None, "NO PULLBACK"

    if trend == "SELL" and not (in_zone and rsi > 45):
        return None, "NO PULLBACK"

    # ENTRY
    last = df_m1.iloc[-1]
    prev = df_m1.iloc[-2]

    if trend == "BUY":
        if last["Close"] > last["Open"]:
            return "BUY", "VALID"

    if trend == "SELL":
        if last["Close"] < last["Open"] and last["Close"] < prev["Low"]:
            return "SELL", "VALID"

    return None, "WAIT ENTRY"

# ================= LOGGER =================
if "trades" not in st.session_state:
    st.session_state.trades = []

# ================= UI =================
st.title("EUR/USD Precision Engine")

if st.button("Scan Market"):
    df_h1 = fetch_data("1h", "30d")
    df_m5 = fetch_data("5m", "5d")
    df_m1 = fetch_data("1m", "1d")

    signal, status = get_signal(df_h1, df_m5, df_m1)

    if signal:
        now = datetime.now().replace(second=0, microsecond=0)

        if now.minute % 2 == 0:
            entry = now + timedelta(minutes=2)
        else:
            entry = now + timedelta(minutes=1)

        expiry = entry + timedelta(minutes=2)

        st.success(f"Signal: {signal}")
        st.write(f"Entry: {entry.strftime('%H:%M')}")
        st.write(f"Expiry: {expiry.strftime('%H:%M')}")

        if st.button("Log Trade"):
            st.session_state.trades.append({
                "time": entry.strftime('%H:%M'),
                "signal": signal,
                "result": None
            })

    else:
        st.warning(status)

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
