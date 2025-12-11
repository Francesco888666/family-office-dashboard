# app.py aggiornato per usare Stooq invece di yfinance
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import time
import io
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import pandas_datareader.data as web

st.set_page_config(page_title="Family Office Dashboard", layout="wide")

# -------------------------
# Finance helpers
# -------------------------
def annualized_return_from_series(price_series):
    n_years = (price_series.index[-1] - price_series.index[0]).days / 365.25
    return (price_series[-1] / price_series[0]) ** (1 / n_years) - 1 if n_years>0 else np.nan

def max_drawdown(price_series):
    cum_max = price_series.cummax()
    drawdown = (price_series - cum_max)/cum_max
    return drawdown.min(), drawdown

def sharpe_ratio(returns, rf=0.01):
    ann_excess = (returns.mean()*252) - rf
    ann_vol = returns.std()*np.sqrt(252)
    return ann_excess/ann_vol if ann_vol>0 else np.nan

def sortino_ratio(returns, rf=0.01):
    neg = returns[returns<0]
    downside = neg.std()*np.sqrt(252) if len(neg)>0 else np.nan
    ann_excess = (returns.mean()*252) - rf
    return ann_excess/downside if downside and downside>0 else np.nan

# -------------------------
# SQLite DB
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
# Sidebar
# -------------------------
st.sidebar.header("📁 Input dati")
mode = st.sidebar.radio("Sorgente dati", ["Carica CSV", "Usa DB"])

df = None
if mode=="Carica CSV":
    uploaded = st.sidebar.file_uploader("Carica file CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded, sep=",", encoding="utf-8-sig")
            df.columns = [c.strip() for c in df.columns]
            # rinomina colonne italiane se necessario
            if 'Cliente' in df.columns:
                df = df.rename(columns={'Cliente':'Client'})
            if 'Quantita' in df.columns:
                df = df.rename(columns={'Quantita':'Quantity'})
            if 'Prezzo' in df.columns:
                df = df.rename(columns={'Prezzo':'Price'})
            # controllo colonna obbligatoria
            if "Client" not in df.columns:
                st.error("Il CSV non contiene la colonna obbligatoria 'Client'. Controlla il file.")
                st.stop()
        except Exception as e:
            st.error(f"Errore lettura CSV: {e}")
elif mode=="Usa DB":
    df = load_clients_from_db()

# -------------------------
# Check df
# -------------------------
if df is None or df.empty:
    st.warning("Nessun dato disponibile. Carica un CSV valido o salva dati nel DB.")
    st.stop()

# -------------------------
# Check colonne obbligatorie
# -------------------------
required_cols = ["Client","Ticker","Quantity"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Colonna obbligatoria mancante: {col}")
        st.stop()

# -------------------------
# Aggiungi colonne mancanti
# -------------------------
for col in ["Price","Sector","Country","AssetClass"]:
    if col not in df.columns:
        df[col] = np.nan

# -------------------------
# Gestione Price
# -------------------------
df["Value"] = df["Quantity"]*df["Price"]
total_val = df["Value"].sum()
df["Weight"] = df["Value"]/total_val if total_val>0 else 1/len(df)

# -------------------------
# Client selection
# -------------------------
clients = df["Client"].unique().tolist()
selected_client = st.sidebar.selectbox("Seleziona cliente", clients)
df_client = df[df["Client"]==selected_client].copy()

st.title(f"Family Office Dashboard — {selected_client}")

# -------------------------
# Holdings
# -------------------------
st.subheader("Holdings")
st.dataframe(df_client[["Ticker","Quantity","Price","Value","Sector","Country","AssetClass","Weight"]])

# -------------------------
# Allocazione charts
# -------------------------
agg_assetclass = df_client.groupby("AssetClass")["Value"].sum().reset_index()
agg_sector = df_client.groupby("Sector")["Value"].sum().reset_index()
col1,col2,col3 = st.columns(3)
col1.dataframe(agg_assetclass)
col2.dataframe(agg_sector)

fig_class = px.pie(agg_assetclass,names="AssetClass",values="Value",hole=0.4,title="Asset Class Allocation")
fig_sector = px.bar(agg_sector,x="Sector",y="Value",title="Sector Allocation")
col1,col2 = st.columns(2)
col1.plotly_chart(fig_class,use_container_width=True)
col2.plotly_chart(fig_sector,use_container_width=True)

# -------------------------
# Historical prices via Stooq
# -------------------------
st.subheader("Performance Storica")
tickers = df_client["Ticker"].unique().tolist()
prices = pd.DataFrame()
failed = []
for t in tickers:
    try:
        s = web.DataReader(t, 'stooq')['Close'].sort_index()  # Stooq restituisce serie storica decrescente
        s = s[-504:]  # circa 2 anni di trading (252 giorni/anno * 2)
        prices[t] = s
    except Exception as e:
        failed.append(t)

if prices.empty:
    st.error("Nessun dato storico disponibile per i ticker.")
    st.stop()
if failed:
    st.warning(f"I seguenti ticker non hanno dati storici: {failed}")

prices = prices.dropna(axis=1,how="all").fillna(method="ffill").dropna(axis=0,how="any")

# Cumulative performance
cum = prices/prices.iloc[0]
fig_perf = go.Figure()
for t in cum.columns:
    fig_perf.add_trace(go.Scatter(x=cum.index,y=cum[t],mode="lines",name=t))
fig_perf.update_layout(title="Performance Cumulativa",xaxis_title="Data",yaxis_title="Cumulativo")
st.plotly_chart(fig_perf,use_container_width=True)

# -------------------------
# Returns & metrics
# -------------------------
rets = prices.pct_change().dropna()
weights = df_client.set_index("Ticker")["Weight"].reindex(prices.columns).fillna(0).values
port_daily = rets.dot(weights)
port_ann = (1+port_daily).prod()**(252/len(port_daily))-1 if len(port_daily)>0 else np.nan
port_vol = port_daily.std()*np.sqrt(252)
port_sharpe = sharpe_ratio(port_daily)
port_sortino = sortino_ratio(port_daily)
mdd_val,_ = max_drawdown((1+port_daily).cumprod())

col1,col2,col3,col4 = st.columns(4)
col1.metric("Rendimento annuo stimato",f"{port_ann:.2%}")
col2.metric("Volatilità annua",f"{port_vol:.2%}")
col3.metric("Sharpe",f"{port_sharpe:.2f}")
col4.metric("Max Drawdown",f"{mdd_val:.2%}")

# -------------------------
# Heatmap rendimenti
# -------------------------
st.subheader("Heatmap rendimenti")
fig_heat = px.imshow(rets.corr(),text_auto=True,aspect="auto",title="Correlazione tra asset")
st.plotly_chart(fig_heat,use_container_width=True)

# Monte Carlo simulation e PDF rimangono invariati rispetto alla versione precedente
