"""
dashboard.py

Streamlit dashboard for the simulated exchange. Shows live order book
depth, recent trades, and fraud alerts. Also lets you place test orders
to trigger matching and (if you spam orders) fraud detection.

Run with:
    streamlit run dashboard.py

Expects the FastAPI backend running locally at http://localhost:8000
(see backend/main.py).
"""

import time

import pandas as pd
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Simulated Crypto Exchange", layout="wide")
st.title("🪙 Simulated Crypto Exchange")
st.caption("Portfolio project — order matching engine + live fraud detection layer. No real funds involved.")

symbol = st.sidebar.selectbox("Symbol", ["BTC", "ETH", "USDT"])
user_id = st.sidebar.text_input("User ID (for placing test orders)", value="demo_user_1")

st.sidebar.markdown("---")
st.sidebar.subheader("Place Order")
side = st.sidebar.radio("Side", ["buy", "sell"])
order_type = st.sidebar.radio("Type", ["limit", "market"])
quantity = st.sidebar.number_input("Quantity", min_value=0.0001, value=0.01, step=0.01)
price = None
if order_type == "limit":
    price = st.sidebar.number_input("Price", min_value=0.01, value=60000.0, step=100.0)

if st.sidebar.button("Submit Order"):
    payload = {
        "user_id": user_id,
        "symbol": symbol,
        "side": side,
        "order_type": order_type,
        "quantity": quantity,
        "price": price,
    }
    try:
        resp = requests.post(f"{API_BASE}/order", json=payload, timeout=5)
        resp.raise_for_status()
        st.sidebar.success(f"Order status: {resp.json()['status']}")
    except Exception as e:
        st.sidebar.error(f"Error placing order: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh"):
    st.rerun()

# ---------- main layout ----------

col1, col2, col3 = st.columns(3)

try:
    live_prices = requests.get(f"{API_BASE}/prices", timeout=5).json()
except Exception:
    live_prices = {}

with col1:
    st.metric("Live BTC (reference)", f"${live_prices.get('BTC', 0):,.2f}")
with col2:
    st.metric("Live ETH (reference)", f"${live_prices.get('ETH', 0):,.2f}")
with col3:
    try:
        ob = requests.get(f"{API_BASE}/orderbook/{symbol}", timeout=5).json()
        last = ob.get("last_price")
        st.metric(f"Last Traded {symbol}", f"${last:,.2f}" if last else "No trades yet")
    except Exception:
        st.metric(f"Last Traded {symbol}", "N/A")

st.markdown("---")

col_book, col_trades = st.columns(2)

with col_book:
    st.subheader(f"📖 Order Book — {symbol}")
    try:
        depth = requests.get(f"{API_BASE}/orderbook/{symbol}", timeout=5).json()["depth"]
        asks_df = pd.DataFrame(depth["asks"]).iloc[::-1] if depth["asks"] else pd.DataFrame(columns=["price", "quantity"])
        bids_df = pd.DataFrame(depth["bids"]) if depth["bids"] else pd.DataFrame(columns=["price", "quantity"])

        st.markdown("**Asks (sell orders)**")
        st.dataframe(asks_df, use_container_width=True, hide_index=True)
        st.markdown("**Bids (buy orders)**")
        st.dataframe(bids_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not load order book: {e}")

with col_trades:
    st.subheader(f"📈 Recent Trades — {symbol}")
    try:
        trades = requests.get(f"{API_BASE}/trades/{symbol}", timeout=5).json()
        if trades:
            df = pd.DataFrame(trades)[["price", "quantity", "buy_user_id", "sell_user_id", "timestamp"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades yet — place matching buy/sell orders to see fills.")
    except Exception as e:
        st.warning(f"Could not load trades: {e}")

st.markdown("---")

st.subheader("🚨 Fraud / Anomaly Alerts")
st.caption("Powered by the same Isolation Forest approach as AMLGuard AI — flags wash trading, spoofing, and outlier order sizes.")
try:
    alerts = requests.get(f"{API_BASE}/fraud-alerts", timeout=5).json()
    if alerts:
        df = pd.DataFrame(alerts)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.success("No fraud alerts triggered yet. Try placing several rapid orders or repeated trades with the same counterparty to trigger detection.")
except Exception as e:
    st.warning(f"Could not load fraud alerts: {e}")

st.markdown("---")
st.subheader("💰 Wallet")
try:
    wallet = requests.get(f"{API_BASE}/wallet/{user_id}", timeout=5).json()
    st.json(wallet)
except Exception as e:
    st.warning(f"Could not load wallet: {e}")