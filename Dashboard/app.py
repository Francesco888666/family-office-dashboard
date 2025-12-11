import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide", page_title="Family Office Ultra Dashboard")
st.title("Family Office Ultra Dashboard - Portfolio Analytics")

# =======================
# Funzioni rischio e performance
# =======================
def sharpe_ratio(returns, risk_free_rate=0.01):
    return (returns.mean() - risk_free_rate/252) / returns.std() * np.sqrt(252)

def sortino_ratio(returns, risk_free_rate=0.01):
    neg_returns = returns[returns < 0]
    return (returns.mean() - risk_free_rate/252) / neg_returns.std() * np.sqrt(252)

def max_drawdown(series):
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min()

def cagr(series):
    n_years = (series.index[-1] - series.index[0]).days / 365.25
    return (series[-1] / series[0]) ** (1/n_years) - 1

def monte_carlo_portfolio(df_filtered, tickers, iterations=1000, days=252):
    prices = pd.DataFrame()
    for t in tickers:
        try:
            prices[t] = yf.download(t, period="1y")['Adj Close']
        except:
            pass
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    results = np.zeros((iterations, days))
    for i in range(iterations):
        daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, days)
        portfolio = (daily_returns @ df_filtered['Value'].values / df_filtered['Value'].sum() + 1).cumprod()
        results[i, :] = portfolio
    return results, prices.index[:days]

def radar_risk(df_risk):
    categories = ['Volatilità', 'Sharpe', 'Sortino', 'Max Drawdown']
    fig = go.Figure()
    for _, row in df_risk.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Volatilità'], row['Sharpe'], row['Sortino'], abs(row['Max Drawdown'])],
            theta=categories,
            fill='toself',
            name=row['Ticker']
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    return fig

# =======================
# Caricamento CSV
# =======================
st.sidebar.subheader("Caricamento portafogli clienti")
uploaded_files = st.sidebar.file_uploader(
    "Carica CSV dei clienti (opzionale, se non presenti nel repo)",
    type="csv",
    accept_multiple_files=True
)

# Controlla se ci sono CSV pre-caricati nella cartella "data" del repo
data_folder = "data"
df_all = pd.DataFrame()
if os.path.exists(data_folder):
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_folder, file_name))
            df_all = pd.concat([df_all, df], ignore_index=True)

# Se ci sono file caricati via uploader, aggiungili
if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)
        df_all = pd.concat([df_all, df], ignore_index=True)

if not df_all.empty:
    df_all['Value'] = df_all['Quantity'] * df_all['Price']

    # =======================
    # Selezione cliente
    # =======================
    clients = df_all['Cliente'].unique()
    selected_client = st.sidebar.selectbox("Seleziona Cliente", clients)
    df_client = df_all[df_all['Cliente'] == selected_client]

    # =======================
    # Filtri dinamici
    # =======================
    sectors = df_client['Sector'].unique()
    selected_sector = st.sidebar.multiselect("Filtra per Settore", sectors, default=list(sectors))

    asset_classes = df_client['AssetClass'].unique()
    selected_class = st.sidebar.multiselect("Filtra per AssetClass", asset_classes, default=list(asset_classes))

    countries = df_client['Country'].unique()
    selected_country = st.sidebar.multiselect("Filtra per Paese", countries, default=list(countries))

    df_filtered = df_client[
        (df_client['Sector'].isin(selected_sector)) &
        (df_client['AssetClass'].isin(selected_class)) &
        (df_client['Country'].isin(selected_country))
    ]

    # =======================
    # KPI principali
    # =======================
    total_value = df_filtered['Value'].sum()
    num_assets = len(df_filtered)
    num_sectors = df_filtered['Sector'].nunique()
    num_classes = df_filtered['AssetClass'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Valore Totale", f"${total_value:,.2f}")
    col2.metric("Numero Asset", num_assets)
    col3.metric("Numero Settori", num_sectors)
    col4.metric("Numero AssetClass", num_classes)

    # =======================
    # Grafico Allocazione AssetClass
    # =======================
    alloc_class = df_filtered.groupby('AssetClass')['Value'].sum().reset_index()
    alloc_class['Percentage'] = alloc_class['Value'] / alloc_class['Value'].sum() * 100
    fig_class = px.pie(alloc_class, names='AssetClass', values='Percentage', title="Allocazione Asset Class")
    st.plotly_chart(fig_class, use_container_width=True)

    # =======================
    # Grafico Allocazione Settoriale
    # =======================
    alloc_sector = df_filtered.groupby('Sector')['Value'].sum().reset_index()
    fig_sector = px.bar(alloc_sector, x='Sector', y='Value', text='Value', title="Allocazione Settoriale")
    st.plotly_chart(fig_sector, use_container_width=True)

    # =======================
    # Performance storica multi-asset
    # =======================
    st.subheader("Performance Storica Multi-Asset")
    tickers = df_filtered['Ticker'].unique()
    prices = pd.DataFrame()
    for t in tickers:
        try:
            prices[t] = yf.download(t, period="1y")['Adj Close']
        except:
            st.warning(f"Dati non disponibili per {t}")

    if not prices.empty:
        cum_returns = prices / prices.iloc[0]
        fig_perf = go.Figure()
        for t in cum_returns.columns:
            fig_perf.add_trace(go.Scatter(
                x=cum_returns.index, y=cum_returns[t], mode='lines', name=t
            ))
        fig_perf.update_layout(title="Performance Storica Cumulativa", xaxis_title="Data", yaxis_title="Cumulativo")
        st.plotly_chart(fig_perf, use_container_width=True)

        # =======================
        # Matrice correlazione
        # =======================
        corr_matrix = cum_returns.pct_change().corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, title="Matrice di Correlazione")
        st.plotly_chart(fig_corr, use_container_width=True)

        # =======================
        # Metriche rischio per ogni asset
        # =======================
        risk_metrics = []
        for t in cum_returns.columns:
            r = cum_returns[t].pct_change().dropna()
            metrics = {
                "Ticker": t,
                "CAGR": cagr(cum_returns[t]),
                "Volatilità": r.std() * np.sqrt(252),
                "Sharpe": sharpe_ratio(r),
                "Sortino": sortino_ratio(r),
                "Max Drawdown": max_drawdown(cum_returns[t])
            }
            risk_metrics.append(metrics)
        df_risk = pd.DataFrame(risk_metrics)
        st.subheader("Metriche di Rischio per Asset")
        st.dataframe(df_risk)

        # =======================
        # Radar rischio
        # =======================
        st.subheader("Grafico Radar Rischi")
        fig_radar = radar_risk(df_risk)
        st.plotly_chart(fig_radar, use_container_width=True)

        # =======================
        # Monte Carlo simulazioni
        # =======================
        st.subheader("Simulazione Monte Carlo Portafoglio")
        sim_results, sim_dates = monte_carlo_portfolio(df_filtered, tickers, iterations=500, days=252)
        fig_mc = go.Figure()
        for i in range(50):
            fig_mc.add_trace(go.Scatter(x=sim_dates, y=sim_results[i], line=dict(color='blue', width=0.5), opacity=0.5))
        fig_mc.update_layout(title="Simulazioni Monte Carlo Portafoglio", xaxis_title="Data", yaxis_title="Valore Simulato")
        st.plotly_chart(fig_mc, use_container_width=True)

        # =======================
        # Alert automatici
        # =======================
        threshold_drawdown = -0.2  # 20%
        for _, row in df_risk.iterrows():
            if row['Max Drawdown'] < threshold_drawdown:
                st.warning(f"ATTENZIONE: {row['Ticker']} ha un drawdown massimo di {row['Max Drawdown']:.2%}")

    # =======================
    # Esportazione report CSV
    # =======================
    st.subheader("Esporta Report Filtrato")
    csv_report = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Scarica CSV Report",
        data=csv_report,
        file_name=f"report_{selected_client}.csv",
        mime="text/csv"
    )

else:
    st.info("Carica i CSV dei portafogli dei clienti o aggiungili nella cartella 'data' per iniziare l'analisi")
