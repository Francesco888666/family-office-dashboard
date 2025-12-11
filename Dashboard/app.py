# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import time
import io
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(page_title="Family Office Dashboard", layout="wide")

# -------------------------
# Helper: robust download
# -------------------------
def safe_download(ticker, period="2y", retries=3, delay=1):
    """Scarica 'Adj Close' da yfinance con retry e threads=False."""
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, threads=False, progress=False)
            if df is not None and not df.empty and "Adj Close" in df.columns:
                return df["Adj Close"]
        except Exception:
            pass
        time.sleep(delay)
    return None

# -------------------------
# Finance helpers
# -------------------------
def annualized_return_from_series(price_series):
    n_years = (price_series.index[-1] - price_series.index[0]).days / 365.25
    return (price_series[-1] / price_series[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan

def max_drawdown(price_series):
    cum_max = price_series.cummax()
    drawdown = (price_series - cum_max) / cum_max
    return drawdown.min(), drawdown

def ulcer_index(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    squared = drawdown ** 2
    ui = np.sqrt(squared.mean())
    return ui

def sharpe_ratio(returns, rf=0.01):
    ann_excess = (returns.mean() * 252) - rf
    ann_vol = returns.std() * np.sqrt(252)
    return ann_excess / ann_vol if ann_vol > 0 else np.nan

def sortino_ratio(returns, rf=0.01):
    neg = returns[returns < 0]
    downside = neg.std() * np.sqrt(252) if len(neg) > 0 else np.nan
    ann_excess = (returns.mean() * 252) - rf
    return ann_excess / downside if downside and downside > 0 else np.nan

# -------------------------
# SQLite (local storage)
# -------------------------
DB_PATH = "portfolios.db"
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client TEXT,
            ticker TEXT,
            quantity REAL,
            price REAL,
            sector TEXT,
            country TEXT,
            assetclass TEXT
        );
    """)
    conn.commit()
    conn.close()

def save_df_to_db(df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("portfolios", conn, if_exists="append", index=False)
    conn.close()

def load_clients_from_db():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM portfolios", conn)
    conn.close()
    return df

init_db()

# -------------------------
# Sidebar: Upload CSV or Load DB
# -------------------------
st.sidebar.header("📁 Input dati")
mode = st.sidebar.radio("Sorgente dati", ["Carica CSV", "Usa DB (salvati)"])

df = None

if mode == "Carica CSV":
    uploaded = st.sidebar.file_uploader("Carica file CSV (Client,Ticker,Quantity,Price,...)", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            df.columns = [c.strip() for c in df.columns]
        except Exception as e:
            st.error(f"Errore lettura CSV: {e}")
elif mode == "Usa DB (salvati)":
    df = load_clients_from_db()

# -------------------------
# Check df valido
# -------------------------
if df is None or df.empty:
    st.warning("Nessun dato disponibile. Carica un CSV valido o salva dati nel DB.")
    st.stop()

# -------------------------
# Aggiungi colonne mancanti
# -------------------------
for col in ["Price", "Sector", "Country", "AssetClass"]:
    if col not in df.columns:
        df[col] = np.nan

# -------------------------
# Gestione Price mancante
# -------------------------
if df["Price"].isna().all():
    st.warning("Colonna 'Price' vuota o mancante: verrà calcolata dai prezzi storici")
df["Value"] = df["Quantity"] * df["Price"]
total_val = df["Value"].sum()
df["Weight"] = df["Value"] / total_val if total_val > 0 else 1 / len(df)

# -------------------------
# Client selection
# -------------------------
clients = df["Client"].unique().tolist()
selected_client = st.sidebar.selectbox("Seleziona cliente", clients)
df_client = df[df["Client"] == selected_client].copy()

st.title(f"Family Office Dashboard — {selected_client}")

# -------------------------
# Holdings table
# -------------------------
st.subheader("Holdings")
st.dataframe(df_client[["Ticker","Quantity","Price","Value","Sector","Country","AssetClass","Weight"]])

# -------------------------
# Allocation charts
# -------------------------
st.subheader("Allocazione")
fig_class = px.pie(df_client.groupby("AssetClass")["Value"].sum().reset_index(),
                   names="AssetClass", values="Value", hole=0.4, title="Asset Class Allocation")
fig_sector = px.bar(df_client.groupby("Sector")["Value"].sum().reset_index(),
                    x="Sector", y="Value", title="Sector Allocation")
col1, col2 = st.columns(2)
col1.plotly_chart(fig_class, use_container_width=True)
col2.plotly_chart(fig_sector, use_container_width=True)

# -------------------------
# Historical prices & performance
# -------------------------
st.subheader("Performance Storica")
tickers = df_client["Ticker"].unique().tolist()
prices = pd.DataFrame()
failed = []
for t in tickers:
    s = safe_download(t, period="2y")
    if s is None:
        failed.append(t)
    else:
        prices[t] = s

if prices.empty:
    st.error("Nessun dato storico disponibile per i ticker del cliente.")
    st.stop()
if failed:
    st.warning(f"I seguenti ticker non hanno dati storici: {failed}")

prices = prices.dropna(axis=1, how="all").fillna(method="ffill").dropna(axis=0, how="any")

cum = prices / prices.iloc[0]
fig_perf = go.Figure()
for t in cum.columns:
    fig_perf.add_trace(go.Scatter(x=cum.index, y=cum[t], mode="lines", name=t))
fig_perf.update_layout(title="Performance Cumulativa", xaxis_title="Data", yaxis_title="Cumulativo")
st.plotly_chart(fig_perf, use_container_width=True)

# -------------------------
# Returns & metrics
# -------------------------
rets = prices.pct_change().dropna()
weights = df_client.set_index("Ticker")["Weight"].reindex(prices.columns).fillna(0).values
port_daily = rets.dot(weights)
port_ann = (1 + port_daily).prod() ** (252/len(port_daily)) - 1 if len(port_daily)>0 else np.nan
port_vol = port_daily.std() * np.sqrt(252)
port_sharpe = sharpe_ratio(port_daily)
port_sortino = sortino_ratio(port_daily)
mdd_val, mdd_series = max_drawdown((1+port_daily).cumprod())
ui = ulcer_index(port_daily)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rendimento annuo stimato", f"{port_ann:.2%}")
col2.metric("Volatilità annua", f"{port_vol:.2%}")
col3.metric("Sharpe", f"{port_sharpe:.2f}")
col4.metric("Max Drawdown", f"{mdd_val:.2%}")
st.metric("Ulcer Index", f"{ui:.4f}")

# -------------------------
# Drawdown chart
# -------------------------
eq = (1 + port_daily).cumprod()
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=eq.index, y=eq, mode="lines", name="Equity"))
fig_dd.add_trace(go.Scatter(x=mdd_series.index, y=mdd_series, mode="lines", name="Drawdown"))
fig_dd.update_layout(title="Equity Curve & Drawdown", xaxis_title="Data")
st.plotly_chart(fig_dd, use_container_width=True)

# -------------------------
# Correlation
# -------------------------
st.subheader("Correlazioni")
fig_corr = px.imshow(rets.corr(), text_auto=True, aspect="auto", title="Correlation matrix")
st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------
# CSV download
# -------------------------
if st.button("Scarica CSV Report"):
    csv_bytes = df_client.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes,
                       file_name=f"report_{selected_client}.csv", mime="text/csv")

st.info("Dashboard pronta — Carica CSV o DB e seleziona cliente per visualizzare dati.")
