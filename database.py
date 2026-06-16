import sqlite3
import os
def init_research_engine():
    # Force python to create the data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # 🧱 CONNECTION BOILERPLATE: Setting up a safe, responsive connection string
    conn = sqlite3.connect("data/research_engine.db", timeout=30.0)
    cursor = conn.cursor()
    
    # 🧱 WAL MODE BOILERPLATE: Forces the database to handle heavy, concurrent actions safely
    cursor.execute("PRAGMA journal_mode=WAL;")
    
    # 🧠 CUSTOM CORE: Table 1 - Latest Month Volume Metrics
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_prices (
        date TEXT ,
        ticker TEXT NOT NULL,
        close_price REAL NOT NULL,
        volume INTEGER NOT NULL,
        PRIMARY KEY (date, ticker)
    )
    """)
    
    # 🧠 CUSTOM CORE: Table 2 - Historical Timeline (12 Months of Data)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS commodity_prices (
        date TEXT PRIMARY KEY,
        copper_price REAL NOT NULL,
        aluminium_price REAL NOT NULL, 
        crude_oil REAL NOT NULL
    )
    """)
    
    # 🧠 CUSTOM CORE: Table 3 - Financial Performance Metrics
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS macro_indicators (
        date TEXT PRIMARY KEY,
        bond_yield_10y REAL NOT NULL,
        usd_index REAL NOT NULL,
        cpi_inflation REAL NOT NULL
    )
    """)
    
    # 🧱 INDEXING BOILERPLATE: Speeds up data lookups when joining tables later
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_commodity ON commodity_prices(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_macro ON macro_indicators(date)")
    
    conn.commit()
    conn.close()
    print("🔋 Relational Research Engine Database Schema Initialized Successfully!")
if __name__ == "__main__":
    init_research_engine()
