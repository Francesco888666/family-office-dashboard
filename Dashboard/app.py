# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
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
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, threads=False, progress=False)
            if df is not None and not df.empty and "Adj Close" in df.columns:
                return df["Adj Close"]
        except:
            pass
        time.sleep(delay)
    return None

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
# Sidebar
# -------------------------
st.sidebar.header("📁 Input Data")
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

df = None
if uploaded:
    try:
        df = pd.read_csv(uploaded, sep=",", encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        # Rename columns if they are in another language
        if 'Cliente' in df.columns:
            df = df.rename(columns={'Cliente':'Client'})
        if 'Quantita' in df.columns:
            df = df.rename(columns={'Quantita':'Quantity'})
        if 'Prezzo' in df.columns:
            df = df.rename(columns={'Prezzo':'Price'})
        # Check required column
        if "Client" not in df.columns:
            st.error("CSV file does not contain required column 'Client'.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
else:
    st.warning("Please upload a CSV file.")
    st.stop()

# -------------------------
# Check required columns
# -------------------------
required_cols = ["Client","Ticker","Quantity"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

# -------------------------
# Add missing optional columns
# -------------------------
for col in ["Price","Sector","Country","AssetClass"]:
    if col not in df.columns:
        df[col] = np.nan

# -------------------------
# Compute value and weights
# -------------------------
if df["Price"].isna().all():
    st.warning("Price column is empty: will be computed from historical prices")
df["Value"] = df["Quantity"]*df["Price"]
total_val = df["Value"].sum()
df["Weight"] = df["Value"]/total_val if total_val>0 else 1/len(df)

# -------------------------
# Client selection
# -------------------------
clients = df["Client"].unique().tolist()
selected_client = st.sidebar.selectbox("Select client", clients)
df_client = df[df["Client"]==selected_client].copy()

st.title(f"Family Office Dashboard — {selected_client}")

# -------------------------
# Holdings
# -------------------------
st.subheader("Holdings")
st.dataframe(df_client[["Ticker","Quantity","Price","Value","Sector","Country","AssetClass","Weight"]])

# -------------------------
# Portfolio aggregation
# -------------------------
st.subheader("Portfolio Aggregations")
agg_assetclass = df_client.groupby("AssetClass")["Value"].sum().reset_index()
agg_sector = df_client.groupby("Sector")["Value"].sum().reset_index()
agg_country = df_client.groupby("Country")["Value"].sum().reset_index()
col1,col2,col3 = st.columns(3)
col1.dataframe(agg_assetclass)
col2.dataframe(agg_sector)
col3.dataframe(agg_country)

# -------------------------
# Allocation charts
# -------------------------
fig_class = px.pie(agg_assetclass,names="AssetClass",values="Value",hole=0.4,title="Asset Class Allocation")
fig_sector = px.bar(agg_sector,x="Sector",y="Value",title="Sector Allocation")
col1,col2 = st.columns(2)
col1.plotly_chart(fig_class,use_container_width=True)
col2.plotly_chart(fig_sector,use_container_width=True)

# -------------------------
# Historical prices
# -------------------------
st.subheader("Historical Performance")
tickers = df_client["Ticker"].unique().tolist()
prices = pd.DataFrame()
failed = []

for t in tickers:
    try:
        s = safe_download(t, period="2y")
        if s is None or s.empty:
            failed.append(t)
            st.warning(f"No historical data for {t}, skipped.")
            continue
        prices[t] = s
    except Exception as e:
        failed.append(t)
        st.warning(f"Error downloading {t}: {e}, skipped.")

prices = prices.dropna(axis=1, how="all").fillna(method="ffill").dropna(axis=0, how="any")
if prices.empty:
    st.error("No historical data available for valid tickers.")
    st.stop()

# Cumulative performance
cum = prices/prices.iloc[0]
fig_perf = go.Figure()
for t in cum.columns:
    fig_perf.add_trace(go.Scatter(x=cum.index,y=cum[t],mode="lines",name=t))
fig_perf.update_layout(title="Cumulative Performance",xaxis_title="Date",yaxis_title="Cumulative")
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
col1.metric("Estimated Annual Return",f"{port_ann:.2%}")
col2.metric("Annual Volatility",f"{port_vol:.2%}")
col3.metric("Sharpe Ratio",f"{port_sharpe:.2f}")
col4.metric("Max Drawdown",f"{mdd_val:.2%}")

# -------------------------
# Heatmap of correlations
# -------------------------
st.subheader("Return Correlation Heatmap")
fig_heat = px.imshow(rets.corr(),text_auto=True,aspect="auto",title="Asset Correlation")
st.plotly_chart(fig_heat,use_container_width=True)

# -------------------------
# Monte Carlo simulation
# -------------------------
st.subheader("Monte Carlo Simulation")
n_sim = st.slider("Number of simulations", 500, 20000, 5000, 500)
horizon = st.slider("Simulation horizon (days)", 30, 252*3, 252)

mu = rets.mean().values
cov = rets.cov().values
sim_results = []
rng = np.random.default_rng()
for i in range(n_sim):
    sim_daily = rng.multivariate_normal(mu,cov,horizon)
    sim_port = np.cumprod(1+sim_daily.dot(weights))[-1]
    sim_results.append(sim_port)

fig_mc = px.histogram(sim_results,nbins=100,title="Monte Carlo Final Portfolio Distribution")
st.plotly_chart(fig_mc,use_container_width=True)

# -------------------------
# PDF report
# -------------------------
st.subheader("Generate PDF Report")
def fig_to_image_bytes(fig,width=900,height=600):
    return fig.to_image(format="png",width=width,height=height,scale=2)

def create_pdf_report(client_name,df_client,figures_dict,output_path="report.pdf"):
    c = canvas.Canvas(output_path,pagesize=A4)
    w,h = A4
    margin = 40
    y = h - margin
    c.setFont("Helvetica-Bold",16)
    c.drawString(margin,y,f"Portfolio Report — {client_name}")
    y -= 30
    c.setFont("Helvetica",10)
    total_val = df_client["Value"].sum()
    c.drawString(margin,y,f"Total Value: {total_val:,.2f} | Assets: {len(df_client)}")
    y -= 20
    for title,fig in figures_dict.items():
        try:
            img_bytes = fig_to_image_bytes(fig,width=700,height=400)
            img = ImageReader(io.BytesIO(img_bytes))
            if y<300:
                c.showPage()
                y = h - margin
            c.setFont("Helvetica-Bold",12)
            c.drawString(margin,y,title)
            y -= 16
            c.drawImage(img,margin,y-300,width=500,height=300)
            y -= 320
        except:
            c.setFont("Helvetica",9)
            c.drawString(margin,y,f"Cannot insert chart: {title}")
            y -= 12
    c.save()

figures = {
    "Asset Class Allocation": fig_class,
    "Sector Allocation": fig_sector,
    "Cumulative Performance": fig_perf,
    "Correlation Heatmap": fig_heat,
    "Monte Carlo": fig_mc
}

if st.button("Generate PDF Report"):
    tmp_pdf = f"report_{selected_client}.pdf"
    try:
        create_pdf_report(selected_client,df_client,figures,output_path=tmp_pdf)
        with open(tmp_pdf,"rb") as f:
            st.download_button("Download PDF",f.read(),file_name=tmp_pdf,mime="application/pdf")
        st.success("PDF Report generated successfully.")
        try: os.remove(tmp_pdf)
        except: pass
    except Exception as e:
        st.error(f"Error generating PDF: {e}")

# -------------------------
# Download CSV
# -------------------------
if st.button("Download CSV Report"):
    csv_bytes = df_client.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV",data=csv_bytes,
                       file_name=f"report_{selected_client}.csv",mime="text/csv")

st.info("Dashboard complete — holdings, metrics, correlation heatmap, Monte Carlo simulation, and PDF report ready.")
