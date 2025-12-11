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
import matplotlib.pyplot as plt

st.set_page_config(page_title="Family Office Ultra Dashboard (Report+DB)", layout="wide")

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
    # Ulcer index calculation from daily returns as in literature (simplified)
    # Convert cumulative to drawdown percentage series
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
# UI: Sidebar - upload or load DB
# -------------------------
st.sidebar.header("📁 Input dati")
mode = st.sidebar.radio("Sorgente dati", ["Carica CSV", "Usa DB (salvati)"])

if mode == "Carica CSV":
    uploaded = st.sidebar.file_uploader("Carica file CSV (Client,Ticker,Quantity,Price,...)",
                                        type=["csv"], accept_multiple_files=False)
    if uploaded:
        df = pd.read_csv(uploaded)
        # Normalize column names
        df.columns = [c.strip() for c in df.columns]
        # If Price missing compute later; ensure columns exist
        required = {"Client", "Ticker", "Quantity"}
        if not required.issubset(set(df.columns)):
            st.error(f"CSV mancante colonne richieste: {required}. Il CSV caricato ha: {list(df.columns)}")
            st.stop()
        # Fill optional columns
        for col in ["Price", "Sector", "Country", "AssetClass"]:
            if col not in df.columns:
                df[col] = np.nan
        # Option to save to DB
        if st.sidebar.button("Salva questo portafoglio nel DB"):
            save_df_to_db(df[["Client","Ticker","Quantity","Price","Sector","Country","AssetClass"]])
            st.sidebar.success("Salvato nel DB")
else:
    df = load_clients_from_db()
    if df.empty:
        st.warning("Nessun portafoglio nel DB. Carica un CSV o salva uno in DB.")
        st.stop()

# -------------------------
# Preprocessing: compute Value and Weight
# -------------------------
# Try to compute price if missing using latest yfinance quote (best effort)
if "Price" not in df.columns or df["Price"].isna().all():
    tickers_unique = df["Ticker"].unique()
    latest_prices = {}
    for t in tickers_unique:
        s = safe_download(t, period="5d", retries=2, delay=1)
        if s is not None and len(s) > 0:
            latest_prices[t] = s[-1]
    df["Price"] = df.apply(lambda row: latest_prices.get(row["Ticker"], np.nan) if pd.isna(row["Price"]) else row["Price"], axis=1)

df["Value"] = df["Quantity"] * df["Price"]
# If there is zero or NaN price produce a warning
if df["Price"].isna().any():
    st.warning("Alcuni asset non hanno prezzo: verranno esclusi dai calcoli che richiedono prezzi storici.")

# -------------------------
# Client selection
# -------------------------
clients = df["Client"].unique().tolist()
selected_client = st.sidebar.selectbox("Seleziona cliente", clients)
df_client = df[df["Client"] == selected_client].copy()

st.title(f"Family Office Dashboard — {selected_client}")

# Compute weights if not present (based on value)
if "Weight" not in df_client.columns or df_client["Value"].sum() == 0:
    total_val = df_client["Value"].sum()
    if total_val > 0:
        df_client["Weight"] = df_client["Value"] / total_val
    else:
        df_client["Weight"] = 1.0 / len(df_client)

# Show holdings
st.subheader("Holdings")
st.dataframe(df_client[["Ticker","Quantity","Price","Value","Sector","Country","AssetClass","Weight"]])

# -------------------------
# Allocation charts
# -------------------------
st.subheader("Allocazione")
col1, col2 = st.columns(2)
with col1:
    by_class = df_client.groupby("AssetClass", dropna=False)["Value"].sum().reset_index()
    fig_class = px.pie(by_class, names="AssetClass", values="Value", title="Asset Class Allocation", hole=0.4)
    st.plotly_chart(fig_class, use_container_width=True)
with col2:
    by_sector = df_client.groupby("Sector", dropna=False)["Value"].sum().reset_index()
    fig_sector = px.bar(by_sector.sort_values("Value", ascending=False), x="Sector", y="Value", title="Sector Allocation")
    st.plotly_chart(fig_sector, use_container_width=True)

# -------------------------
# Historical prices & performance
# -------------------------
st.subheader("Performance Storica e Metriche di Rendimento/Rischio")

# Download historical prices robustly
tickers = df_client["Ticker"].unique().tolist()
prices = pd.DataFrame()
failed = []
for t in tickers:
    s = safe_download(t, period="2y", retries=3, delay=1)
    if s is None:
        failed.append(t)
    else:
        prices[t] = s

if prices.empty:
    st.error("Nessun dato storico disponibile per i ticker del cliente.")
    st.stop()

if failed:
    st.warning(f"I seguenti ticker non hanno dati storici: {failed}")

# Align and drop columns with NaNs (if any)
prices = prices.dropna(axis=1, how="all")
prices = prices.fillna(method="ffill").dropna(axis=0, how="any")

# Cumulative performance
cum = prices / prices.iloc[0]
fig = go.Figure()
for t in cum.columns:
    fig.add_trace(go.Scatter(x=cum.index, y=cum[t], mode="lines", name=t))
fig.update_layout(title="Performance Cumulativa", xaxis_title="Data", yaxis_title="Cumulativo")
st.plotly_chart(fig, use_container_width=True)

# Returns
rets = prices.pct_change().dropna()

# Portfolio aggregated metrics
weights = df_client.set_index("Ticker")["Weight"].reindex(prices.columns).fillna(0).values
port_daily = rets.dot(weights)
port_ann = (1 + port_daily).prod() ** (252/len(port_daily)) - 1 if len(port_daily)>0 else np.nan
port_vol = port_daily.std() * np.sqrt(252)
port_sharpe = sharpe_ratio(port_daily)
port_sortino = sortino_ratio(port_daily)
mdd_val, mdd_series = max_drawdown((1+port_daily).cumprod())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rendimento annuo stimato", f"{port_ann:.2%}")
col2.metric("Volatilità annua", f"{port_vol:.2%}")
col3.metric("Sharpe", f"{port_sharpe:.2f}")
col4.metric("Max Drawdown", f"{mdd_val:.2%}")

# Ulcer Index
ui = ulcer_index(port_daily)
st.metric("Ulcer Index", f"{ui:.4f}")

# Drawdown chart
st.subheader("Drawdown & Equity Curve")
eq = (1 + port_daily).cumprod()
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=eq.index, y=eq, mode="lines", name="Equity"))
fig_dd.add_trace(go.Scatter(x=mdd_series.index, y=mdd_series, mode="lines", name="Drawdown"))
fig_dd.update_layout(title="Equity Curve & Drawdown", xaxis_title="Data")
st.plotly_chart(fig_dd, use_container_width=True)

# Correlation heatmap
st.subheader("Correlazioni tra asset")
fig_corr = px.imshow(rets.corr(), text_auto=True, aspect="auto", title="Correlation matrix")
st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------
# Risk metrics per asset
# -------------------------
st.subheader("Metriche rischio per asset")
risk_rows = []
for t in rets.columns:
    r = rets[t]
    rr = {
        "Ticker": t,
        "CAGR": annualized_return_from_series(prices[t]),
        "Volatility": r.std() * np.sqrt(252),
        "Sharpe": sharpe_ratio(r),
        "Sortino": sortino_ratio(r),
        "MaxDrawdown": max_drawdown(prices[t])[0]
    }
    risk_rows.append(rr)
df_risk = pd.DataFrame(risk_rows).set_index("Ticker")
st.dataframe(df_risk)

# Radar chart for risk (normalize for plot)
st.subheader("Radar profilo rischio (Vol, Sharpe, Sortino, |MDD|)")
radar_df = df_risk.copy().reset_index()
# Normalize columns for radar scale (simple minmax)
cols = ["Volatility","Sharpe","Sortino","MaxDrawdown"]
norm = {}
for c in cols:
    v = radar_df[c].abs() if c=="MaxDrawdown" else radar_df[c]
    minv, maxv = v.min(), v.max()
    if maxv - minv == 0:
        radar_df[c+"_norm"] = 0.5
    else:
        radar_df[c+"_norm"] = (v - minv) / (maxv - minv)
fig_radar = go.Figure()
for _, row in radar_df.iterrows():
    fig_radar.add_trace(go.Scatterpolar(
        r=[row[c+"_norm"] for c in cols],
        theta=cols,
        fill='toself',
        name=row["Ticker"]
    ))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)))
st.plotly_chart(fig_radar, use_container_width=True)

# -------------------------
# Monte Carlo portfolio simulation
# -------------------------
st.subheader("Simulazione Monte Carlo portafoglio")
n_sim = st.slider("Numero simulazioni", min_value=500, max_value=20000, value=5000, step=500)
horizon = st.slider("Giorni orizzonte (trading)", min_value=30, max_value=252*3, value=252)
mu = rets.mean().values
cov = rets.cov().values

sim_results = []
rng = np.random.default_rng()
for i in range(n_sim):
    sim_daily = rng.multivariate_normal(mu, cov, horizon)
    sim_port = np.cumprod(1 + sim_daily.dot(weights))[-1]
    sim_results.append(sim_port)

fig_mc = px.histogram(sim_results, nbins=100, title="Distribuzione Monte Carlo (valore relativo finale)")
st.plotly_chart(fig_mc, use_container_width=True)

# -------------------------
# PDF report generation (with images)
# -------------------------
st.subheader("Genera Report PDF (con grafici)")

def fig_to_image_bytes(fig, width=900, height=600):
    """Salva un Plotly figure in memoria come PNG (usa kaleido)."""
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
    return img_bytes

def create_pdf_report(client_name, df_client, figures_dict, output_path="report.pdf"):
    """
    figures_dict = {"title": plotly_fig, ...}
    Salva PDF con i grafici immessi.
    """
    c = canvas.Canvas(output_path, pagesize=A4)
    w, h = A4
    margin = 40
    y = h - margin
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, f"Report Portafoglio — {client_name}")
    y -= 30
    c.setFont("Helvetica", 10)
    # summary table (top)
    total_val = df_client["Value"].sum()
    c.drawString(margin, y, f"Valore Totale: {total_val:,.2f} | Asset: {len(df_client)}")
    y -= 20

    for title, fig in figures_dict.items():
        # convert fig to image bytes
        try:
            img_bytes = fig_to_image_bytes(fig, width=700, height=400)
            img = ImageReader(io.BytesIO(img_bytes))
            if y < 300:
                c.showPage()
                y = h - margin
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y, title)
            y -= 16
            # draw image
            img_w = 500
            img_h = 300
            c.drawImage(img, margin, y - img_h, width=img_w, height=img_h)
            y -= img_h + 20
        except Exception as e:
            c.setFont("Helvetica", 9)
            c.drawString(margin, y, f"Impossibile inserire grafico: {title} ({e})")
            y -= 12

    c.save()

# Prepare a set of figures to embed
figures = {
    "Allocazione AssetClass": fig_class,
    "Allocazione Settore": fig_sector,
    "Performance Cumulativa": fig,
    "Correlazioni": fig_corr,
    "Equity & Drawdown": fig_dd,
    "Monte Carlo": fig_mc
}

if st.button("Genera PDF completo"):
    tmp_pdf = f"report_{selected_client}.pdf"
    try:
        create_pdf_report(selected_client, df_client, figures, output_path=tmp_pdf)
        with open(tmp_pdf, "rb") as f:
            st.download_button("Scarica PDF Report", f.read(), file_name=tmp_pdf, mime="application/pdf")
        st.success("Report generato con successo.")
        # cleanup
        try:
            os.remove(tmp_pdf)
        except:
            pass
    except Exception as e:
        st.error(f"Errore nella generazione PDF: {e}")

# CSV report download
if st.button("Scarica CSV Report (filtered)"):
    csv_bytes = df_client.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name=f"report_{selected_client}.csv", mime="text/csv")

# -------------------------
# Simple rule-based "AI chat" (no external API)
# -------------------------
st.subheader("Chat di supporto (AI locale) — chiedi suggerimenti sul portafoglio")
user_q = st.text_input("Fai una domanda (es. 'Qual è il rischio?')")

def local_ai_reply(question, metrics):
    q = question.lower()
    ans = []
    if "rischio" in q or "volatil" in q:
        ans.append(f"Volatilità annua stimata: {metrics['volatility']:.2%}.")
        if metrics['volatility'] > 0.20:
            ans.append("Il portafoglio ha una volatilità elevata (>20%). Considera diversificazione o riduzione delle posizioni più volatili.")
        else:
            ans.append("La volatilità è in range moderato.")
    if "drawdown" in q or "perdita" in q:
        ans.append(f"Max drawdown stimato: {metrics['mdd']:.2%}.")
        if metrics['mdd'] < -0.15:
            ans.append("Il drawdown è significativo; verifica le posizioni con più contribuzione negativa.")
    if "sharpe" in q:
        ans.append(f"Sharpe ratio stimato: {metrics['sharpe']:.2f}.")
        if metrics['sharpe'] < 0.5:
            ans.append("Sharpe basso — rendimento corretto per rischio non favorevole.")
    if "consigli" in q or "migliorare" in q or "dove" in q:
        ans.append("Controlla esposizione settoriale e concentrazione: riduci peso dove c'è sovraesposizione rispetto agli obiettivi.")
        ans.append("Valuta coperture, re-balance periodico e controllo costi/commissioni.")
    if not ans:
        ans = ["Mi dispiace, posso rispondere a domande su rischio, drawdown, Sharpe e suggerimenti di diversificazione."]
    return " ".join(ans)

# pass metrics to ai
metrics = {"volatility": port_vol, "mdd": mdd_val, "sharpe": port_sharpe}
if user_q:
    reply = local_ai_reply(user_q, metrics)
    st.info(reply)

st.info("Fine dashboard — puoi generare report PDF, scaricare CSV o salvare portafogli in DB.")
