import requests
from bs4 import BeautifulSoup
import random
from collections import defaultdict
import csv

# --- Funzione per estrarre ticker Nasdaq da Stooq ---
def get_tickers_from_stooq():
    url = "https://stooq.com/t/n/?i=10"  # Nasdaq ticker list pagina Stooq
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    tickers = []
    for a in soup.select("table.fth1 a"):
        ticker = a.text.strip()
        if ticker.endswith(".US"):  # filtro per ticker USA (Nasdaq, NYSE)
            tickers.append(ticker)
    return list(set(tickers))  # rimuove duplicati

# --- Dati di supporto ---
SECTORS = [
    "Tech", "Finance", "Healthcare", "Consumer", "Energy", "Auto", "Food",
    "Entertainment", "Semiconductors", "Aerospace", "Utilities", "Real Estate"
]

COUNTRIES = [
    "USA", "UK", "Germany", "France", "Japan", "China", "Canada", "Italy",
    "Netherlands", "Taiwan", "Australia", "Brazil"
]

ASSET_CLASSES = ["Stock", "ETF", "Bond"]

# --- Parametri ---
NUM_CLIENTS = 50
MIN_POSITIONS_PER_CLIENT = 5
MAX_POSITIONS_PER_CLIENT = 8
MAX_TICKER_REPETITIONS = 10

# --- Estrazione ticker ---
try:
    TICKERS = get_tickers_from_stooq()
    if len(TICKERS) < 50:
        raise ValueError("Ticker insufficienti estratti da Stooq, uso lista di backup.")
except Exception as e:
    print(f"Errore estrazione ticker da Stooq: {e}")
    TICKERS = [
        "AAPL.US", "MSFT.US", "GOOGL.US", "AMZN.US", "TSLA.US", "NVDA.US", "JPM.US", "BAC.US", "V.US", "JNJ.US",
        "PFE.US", "CVX.US", "XOM.US", "MCD.US", "SBUX.US", "PEP.US", "KO.US", "CRM.US", "ADBE.US", "INTU.US"
    ]

random.shuffle(TICKERS)

# --- Generazione dati clienti ---
all_client_positions = []
ticker_counts = defaultdict(int)

for i in range(1, NUM_CLIENTS + 1):
    client_id = f"Client_{i:02d}"
    num_positions = random.randint(MIN_POSITIONS_PER_CLIENT, MAX_POSITIONS_PER_CLIENT)

    client_tickers_used = set()

    for _ in range(num_positions):
        available_tickers = [t for t in TICKERS if ticker_counts[t] < MAX_TICKER_REPETITIONS and t not in client_tickers_used]
        if not available_tickers:
            break
        selected_ticker = random.choice(available_tickers)
        ticker_counts[selected_ticker] += 1
        client_tickers_used.add(selected_ticker)

        quantity = random.randint(1, 20)
        price = round(random.uniform(10.0, 1500.0), 2)
        sector = random.choice(SECTORS)
        country = random.choice(COUNTRIES)
        asset_class = random.choice(ASSET_CLASSES)

        all_client_positions.append([
            client_id,
            selected_ticker,
            quantity,
            price,
            sector,
            country,
            asset_class
        ])

# --- Scrittura su file CSV ---
filename = "clienti_posizioni.csv"
header = ["Client", "Ticker", "Quantity", "Price", "Sector", "Country", "AssetClass"]

with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(all_client_positions)

print(f"File CSV '{filename}' generato con successo.")
