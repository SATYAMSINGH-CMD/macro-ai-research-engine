import sqlite3
import os


def init_research_engine():
    """Initialize the SQLite database schema for the Macro Research Engine.

    Creates three tables with strict constraints:
        - stock_prices: Daily OHLCV data per ticker (composite PK: date + ticker)
        - commodity_prices: Daily commodity spot prices (PK: date)
        - macro_indicators: Daily macroeconomic indicators (PK: date)

    Safe to run multiple times — uses CREATE TABLE IF NOT EXISTS.
    """
    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect("data/research_engine.db", timeout=30.0)
    cursor = conn.cursor()

    # Enable WAL mode for safe concurrent reads during dashboard queries
    cursor.execute("PRAGMA journal_mode=WAL;")

    # Table 1: Stock / ETF price data
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_prices (
        date TEXT,
        ticker TEXT NOT NULL,
        close_price REAL NOT NULL,
        volume INTEGER NOT NULL,
        PRIMARY KEY (date, ticker)
    )
    """)

    # Table 2: Commodity prices (daily)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS commodity_prices (
        date TEXT PRIMARY KEY,
        crude_oil REAL NOT NULL,
        copper_price REAL NOT NULL,
        aluminium_price REAL NOT NULL
    )
    """)

    # Table 3: Macroeconomic indicators (daily)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS macro_indicators (
        date TEXT PRIMARY KEY,
        bond_yield_10y REAL NOT NULL,
        usd_index REAL NOT NULL,
        cpi_inflation REAL NOT NULL
    )
    """)

    # Indexes for join performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_date ON stock_prices(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_commodity_date ON commodity_prices(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_indicators(date)")

    conn.commit()
    conn.close()
    print("✅ Database schema initialized: stock_prices, commodity_prices, macro_indicators")


if __name__ == "__main__":
    init_research_engine()
