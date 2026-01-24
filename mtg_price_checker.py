#! /usr/bin/env python3

# pylint: disable=missing-module-docstring, missing-class-docstring, missing-function-docstring


import time
import sqlite3
import datetime
import os
import requests

# --- Configuration ---
# List of cards to track: (Card Name, Set Code, Target Price USD, [Optional] Collector Number)
CARD_LIST = [
    ("Black Lotus", "LEA", 20000.00),
    ("Mox Pearl", "LEA", 4000.00),
    ("Underground Sea", "3ED", 600.00),
    ("Volcanic Island", "3ED", 600.00),
    ("Tropical Island", "3ED", 500.00),
    ("Force of Will", "ALL", 80.00),
    ("Tarmogoyf", "FUT", 100.00), # Just for fun, seeing how low it goes
    ("Maha, Its Feathers Night", "BLB", 20.00, "289"), # Specific Showcase printing
]

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prices.db")

# --- Database Functions ---

def setup_database():
    """Creates the price_history table if it doesn't exist and handles migrations."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_name TEXT NOT NULL,
            set_code TEXT NOT NULL,
            price_usd REAL,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Migration: Add collector_number column if it doesn't exist
    try:
        cursor.execute("ALTER TABLE price_history ADD COLUMN collector_number TEXT")
        print("[Info] Added 'collector_number' column to database.")
    except sqlite3.OperationalError:
        # Column likely already exists
        pass

    conn.commit()
    return conn

def save_price(conn, card_name, set_code, price_usd, collector_number=None):
    """
    Saves the fetched price to the database, ensuring only one entry per day per card.
    """
    cursor = conn.cursor()

    # Check if we already have an entry for today (UTC)
    if collector_number:
        query = '''
            SELECT id FROM price_history
            WHERE card_name = ?
              AND set_code = ?
              AND collector_number = ?
              AND date(fetched_at) = date('now')
        '''
        params = (card_name, set_code, collector_number)
    else:
        query = '''
            SELECT id FROM price_history
            WHERE card_name = ?
              AND set_code = ?
              AND collector_number IS NULL
              AND date(fetched_at) = date('now')
        '''
        params = (card_name, set_code)

    cursor.execute(query, params)

    if cursor.fetchone():
        print(f"  [Info] Price for {card_name} already saved today.")
        return

    cursor.execute('''
        INSERT INTO price_history (card_name, set_code, price_usd, collector_number)
        VALUES (?, ?, ?, ?)
    ''', (card_name, set_code, price_usd, collector_number))
    conn.commit()

def get_history(conn, card_name, set_code, collector_number=None, limit=5):
    """Retrieves the last N price entries for a specific card/set."""
    cursor = conn.cursor()

    if collector_number:
        cursor.execute('''
            SELECT price_usd, fetched_at FROM price_history
            WHERE card_name = ? AND set_code = ? AND collector_number = ?
            ORDER BY fetched_at DESC
            LIMIT ?
        ''', (card_name, set_code, collector_number, limit))
    else:
        cursor.execute('''
            SELECT price_usd, fetched_at FROM price_history
            WHERE card_name = ? AND set_code = ? AND collector_number IS NULL
            ORDER BY fetched_at DESC
            LIMIT ?
        ''', (card_name, set_code, limit))

    return cursor.fetchall()

# --- API Functions ---

def get_card_price(card_name, set_code, collector_number=None):
    """
    Fetches the current non-foil market price for a specific card printing from Scryfall.
    Returns: price (float) or None if not found/no price.
    """
    if collector_number:
        # Fetch by Set + Collector Number (Specific Printing)
        url = f"https://api.scryfall.com/cards/{set_code.lower()}/{collector_number}"
        params = {}
    else:
        # Fetch by Name + Set
        url = "https://api.scryfall.com/cards/named"
        params = {
            "exact": card_name,
            "set": set_code
        }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Raise error for 404, 500 etc
        data = response.json()

        # Validate name if we fetched by collector number to avoid "wrong card" issues
        fetched_name = data.get("name")
        if collector_number and fetched_name and card_name.lower() not in fetched_name.lower():
            print(f"  {Colors.RED}Error: Collector number {collector_number} returned '{fetched_name}', expected '{card_name}'{Colors.ENDC}")
            return None

        # Scryfall prices are in 'prices' object
        prices = data.get("prices", {})
        price_usd = prices.get("usd")

        if price_usd:
            return float(price_usd)
        else:
            print(f"  Warning: No USD price found for {card_name} ({set_code})")
            return None

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            cn_info = f" #{collector_number}" if collector_number else ""
            print(f"  Error: Card not found: {card_name} ({set_code}{cn_info})")
        else:
            print(f"  API Error for {card_name}: {e}")
        return None
    except Exception as e:
        print(f"  Unexpected error for {card_name}: {e}")
        return None

# --- Recommendation Logic ---

def generate_recommendation(current_price, target_price, history):
    """
    Generates a BUY/WAIT/NEUTRAL recommendation.

    Logic:
    - BUY: Current price <= Target Price
    - BUY: Current price < All-time low in known history (min 3 entries)
    - WAIT: Price > Target
    - NEUTRAL: No valid price or insufficient data
    """
    if current_price is None:
        return "UNKNOWN"

    if current_price <= target_price:
        return "BUY (Below Target)"

    # Check if it's an historical low
    valid_history_prices = [h[0] for h in history if h[0] is not None]

    if len(valid_history_prices) >= 3:
        min_hist = min(valid_history_prices)
        if current_price < min_hist:
            return "BUY (New Low)"

    return "WAIT"

# --- Visualization ---

def render_ascii_graph(history):
    """
    Renders a simple ASCII bar chart for the given history.
    History is expected to be a list of (price, date_str).
    """
    if not history:
        return

    # Sort by date ascending (Oldest first)
    chrono_history = sorted(history, key=lambda x: x[1])

    prices = [h[0] for h in chrono_history if h[0] is not None]
    if not prices:
        return

    min_p = min(prices)
    max_p = max(prices)
    distinct_range = max_p - min_p

    print("  Price History (Trend):")
    max_bar_width = 30

    for price, date in chrono_history:
        if price is None:
            continue

        date_short = date.split(' ')[0]

        if distinct_range == 0:
            bar_len = max_bar_width // 2
        else:
            pct = (price - min_p) / distinct_range
            bar_len = 1 + int(pct * (max_bar_width - 1))

        bar = '#' * bar_len
        print(f"    {date_short}: {bar:<30} ${price:.2f}")

# --- Colors ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# --- Main Execution ---

if __name__ == "__main__":
    print(f"{Colors.HEADER}--- MTG Price Fetcher ---{Colors.ENDC}")
    print(f"Timestamp: {datetime.datetime.now()}")

    conn = setup_database()

    for item in CARD_LIST:
        # Unpack tuple with optional collector number
        if len(item) == 4:
            card_name, set_code, target_price, collector_number = item
        else:
            card_name, set_code, target_price = item
            collector_number = None

        cn_str = f" #{collector_number}" if collector_number else ""
        print(f"\n{Colors.BOLD}Checking: {card_name} [{set_code}{cn_str}]...{Colors.ENDC}")

        # 1. Get History (before saving new one, to see previous state)
        history = get_history(conn, card_name, set_code, collector_number)

        # 2. Get Current Price
        current_price = get_card_price(card_name, set_code, collector_number)

        # 3. Recommendation
        recommendation = generate_recommendation(current_price, target_price, history)

        # Colorize Recommendation
        rec_color = Colors.ENDC
        if "BUY" in recommendation:
            rec_color = Colors.GREEN
        elif "WAIT" in recommendation:
            rec_color = Colors.YELLOW
        elif "UNKNOWN" in recommendation:
            rec_color = Colors.RED

        # 4. Save to DB
        if current_price is not None:
            save_price(conn, card_name, set_code, current_price, collector_number)

        # 5. Output
        price_str = f"${current_price:.2f}" if current_price else "N/A"
        print(f"  Current Price: {Colors.CYAN}{price_str}{Colors.ENDC}")
        print(f"  Target Price:  ${target_price:.2f}")
        print(f"  Recommendation: {rec_color}{recommendation}{Colors.ENDC}")

        # Refetch history to include the latest save for the graph (and get more points)
        history = get_history(conn, card_name, set_code, collector_number, limit=14)

        if history:
            render_ascii_graph(history)
        else:
            print("  No history available.")

        # Rate limiting behavior
        time.sleep(0.1)

    conn.close()
    print("\nDone.")
