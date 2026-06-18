import os
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime
from features import ASSET_TICKERS

DB_PATH = "data/research_engine.db"


def fetch_and_load_data():
    """Ingest all market, commodity, and macro data into the research database.

    Data sources (all via yfinance — single dependency):
        - Stock/ETF prices: 17 assets across 4 sectors + 3 index/sector ETFs
        - Commodities: Crude oil (CL=F), Copper (HG=F), Aluminium (ALI=F)
        - Macro: 10Y Treasury yield (^TNX), USD Index (DX-Y.NYB),
                 10Y Breakeven Inflation Rate (^T10YIE)
    """
    start_date = "2021-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)

    # ==================================================
    # STEP 1: Macro and Commodity Data
    # ==================================================
    print("⚡ Step 1: Ingesting macro and commodity data via yfinance...")

    macro_tickers = {
        '^TNX': 'bond_yield_10y',
        'DX-Y.NYB': 'usd_index',
        'CL=F': 'crude_oil',
        'HG=F': 'copper_price',
        'ALI=F': 'aluminium_price',
    }

    try:
        macro_download = yf.download(
            list(macro_tickers.keys()),
            start=start_date, end=end_date,
            progress=False
        )

        macro_close = macro_download['Close'].copy()
        macro_close.index = macro_close.index.strftime('%Y-%m-%d')
        macro_close = macro_close.rename(columns=macro_tickers)

        # Separate into commodity and macro tables
        commodity_df = macro_close[['crude_oil', 'copper_price', 'aluminium_price']].dropna(how='all')
        macro_df = macro_close[['bond_yield_10y', 'usd_index']].copy().dropna(how='all')

        # Download T10YIE from FRED directly
        print("   -> Fetching T10YIE (10-Year Breakeven Inflation Rate) from FRED...")
        try:
            fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10YIE"
            fred_df = pd.read_csv(fred_url)
            fred_df['observation_date'] = pd.to_datetime(fred_df['observation_date']).dt.strftime('%Y-%m-%d')
            fred_df['T10YIE'] = pd.to_numeric(fred_df['T10YIE'], errors='coerce')
            fred_df = fred_df.rename(columns={'observation_date': 'date', 'T10YIE': 'cpi_inflation'})
            fred_df = fred_df.set_index('date')
            
            macro_df = macro_df.merge(fred_df, left_index=True, right_index=True, how='left')
            macro_df['cpi_inflation'] = macro_df['cpi_inflation'].ffill().bfill()
        except Exception as e:
            print(f"      ⚠️ Failed to fetch T10YIE from FRED ({e}), falling back to 3.2%")
            macro_df['cpi_inflation'] = 3.2

        commodity_df.to_sql("commodity_prices", conn, if_exists="replace", index_label="date")
        macro_df.to_sql("macro_indicators", conn, if_exists="replace", index_label="date")
        print(f"   ✅ Saved {len(commodity_df)} commodity rows, {len(macro_df)} macro rows.")

    except Exception as e:
        print(f"❌ Error fetching macro data: {e}")

    # ==================================================
    # STEP 2: Stock and ETF Price Data
    # ==================================================
    print("\n⚡ Step 2: Ingesting stock and ETF data via yfinance...")

    # Clear existing stock data to avoid duplicate key errors on re-runs
    conn.execute("DELETE FROM stock_prices")
    conn.commit()

    for ticker in ASSET_TICKERS:
        print(f"   -> Fetching: {ticker}")
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)

            if not hist.empty:
                stock_df = hist[['Close', 'Volume']].copy().dropna()
                stock_df['ticker'] = ticker
                stock_df.index = stock_df.index.strftime('%Y-%m-%d')
                stock_df = stock_df.rename(columns={'Close': 'close_price', 'Volume': 'volume'})

                stock_df.to_sql("stock_prices", conn, if_exists="append", index_label="date")
                print(f"      ✅ {len(stock_df)} rows saved")

        except Exception as e:
            print(f"      ❌ Error for {ticker}: {e}")

    conn.close()
    print("\n🔋 Pipeline complete. All tables populated.")


if __name__ == "__main__":
    fetch_and_load_data()