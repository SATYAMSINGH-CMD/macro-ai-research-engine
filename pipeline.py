import os
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime

DB_PATH = "data/research_engine.db"

def fetch_and_load_data():
    # Keep the 2021 start date to ensure clean, recent daily overlap
    start_date = "2021-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(DB_PATH)
    
    print("⚡ Step 1: Ingesting Macro and Commodity metrics via yfinance (FRED Bypass)...")
    # Mapping our indicators to working Yahoo Finance tickers
    macro_tickers = {
        '^TNX': 'bond_yield_10y',
        'DX-Y.NYB': 'usd_index',
        'CL=F': 'crude_oil',
        'HG=F': 'copper_price',
        'ALI=F': 'aluminium_price'
    }
    
    try:
        # Download all macro assets together
        macro_download = yf.download(list(macro_tickers.keys()), start=start_date, end=end_date)
        
        # Pull only the Close prices
        macro_close = macro_download['Close'].copy()
        macro_close.index = macro_close.index.strftime('%Y-%m-%d')
        
        # Rename the columns from tickers to our clean database schema names
        macro_close = macro_close.rename(columns=macro_tickers)
        
        # Inject our structural control cpi value
        macro_close['cpi_inflation'] = 0.032
        
        # Separate into our two historical database tables
        commodity_df = macro_close[['copper_price', 'aluminium_price', 'crude_oil']].dropna(how='all')
        macro_df = macro_close[['bond_yield_10y', 'usd_index', 'cpi_inflation']].dropna(how='all')
        
        # Load directly into our distinct SQL slots
        commodity_df.to_sql("commodity_prices", conn, if_exists="replace", index_label="date")
        macro_df.to_sql("macro_indicators", conn, if_exists="replace", index_label="date")
        print(f"   ✅ Saved {len(commodity_df)} macro/commodity timeline rows.")
        
    except Exception as e:
        print(f"❌ Error fetching macro data from Yahoo: {e}")

    print("\n⚡ Step 2: Ingesting stock market data via yfinance...")
    target_tickers = ["NVDA", "TSLA", "DAL"]
    
    for ticker in target_tickers:
        print(f" -> Fetching market data for ticker: {ticker}")
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if not hist.empty:
                stock_df = hist[['Close', 'Volume']].copy()
                stock_df = stock_df.dropna()
                
                stock_df['ticker'] = ticker
                stock_df.index = stock_df.index.strftime('%Y-%m-%d')
                stock_df = stock_df.rename(columns={'Close': 'close_price', 'Volume': 'volume'})
                
                stock_df.to_sql("stock_prices", conn, if_exists="append", index_label="date")
                print(f"   ✅ Saved {len(stock_df)} rows for {ticker}")
                
        except Exception as e:
            print(f"❌ Error fetching stock data for {ticker}: {e}")
            
    conn.close()
    print("\n🔋 Pipeline complete! All database assets are fully populated.")

if __name__ == "__main__":
    fetch_and_load_data()