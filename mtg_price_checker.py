#! /usr/bin/env python3

"""
MTG Price Checker
-----------------
A script to fetch and track Magic: The Gathering card prices from Scryfall.
Stores history in a local SQLite database and provides BUY/WAIT recommendations.
"""

import sys
import argparse
import json
import sqlite3
import datetime
import time
from typing import List, Tuple, Optional, Any, TypedDict
import requests  # type: ignore

# --- Types ---

class CardData(TypedDict):
    """Structure for card data from JSON or CLI."""
    name: str
    set: str
    target_price: Optional[float]
    collector_number: Optional[str]

# --- Database Functions ---

def setup_database(db_path: str) -> sqlite3.Connection:
    """Creates the price_history table if it doesn't exist and handles migrations."""
    conn = sqlite3.connect(db_path)
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
    except sqlite3.OperationalError:
        # Column likely already exists
        pass

    conn.commit()
    return conn

def save_price(conn: sqlite3.Connection, card: CardData, price_usd: float) -> None:
    """
    Saves the fetched price to the database, ensuring only one entry per day per card.
    """
    cursor = conn.cursor()

    params: Tuple[Any, ...]
    if card['collector_number']:
        query = '''
            SELECT id FROM price_history
            WHERE card_name = ?
              AND set_code = ?
              AND collector_number = ?
              AND date(fetched_at) = date('now')
        '''
        params = (card['name'], card['set'], card['collector_number'])
    else:
        query = '''
            SELECT id FROM price_history
            WHERE card_name = ?
              AND set_code = ?
              AND collector_number IS NULL
              AND date(fetched_at) = date('now')
        '''
        params = (card['name'], card['set'])

    cursor.execute(query, params)

    if cursor.fetchone():
        print(f"  [Info] Price for {card['name']} already saved today.")
        return

    cursor.execute('''
        INSERT INTO price_history (card_name, set_code, price_usd, collector_number)
        VALUES (?, ?, ?, ?)
    ''', (card['name'], card['set'], price_usd, card['collector_number']))
    conn.commit()

def get_history(conn: sqlite3.Connection, card: CardData, limit: int = 5) \
        -> List[Tuple[Optional[float], str]]:
    """Retrieves the last N price entries for a specific card/set."""
    cursor = conn.cursor()

    if card['collector_number']:
        cursor.execute('''
            SELECT price_usd, fetched_at FROM price_history
            WHERE card_name = ? AND set_code = ? AND collector_number = ?
            ORDER BY fetched_at DESC
            LIMIT ?
        ''', (card['name'], card['set'], card['collector_number'], limit))
    else:
        cursor.execute('''
            SELECT price_usd, fetched_at FROM price_history
            WHERE card_name = ? AND set_code = ? AND collector_number IS NULL
            ORDER BY fetched_at DESC
            LIMIT ?
        ''', (card['name'], card['set'], limit))

    return cursor.fetchall()

# --- API Functions ---

def get_card_price(card: CardData) -> Optional[float]:
    """
    Fetches the current non-foil market price for a specific card printing from Scryfall.
    Returns: price (float) or None if not found/no price.
    """
    if card['collector_number']:
        # Fetch by Set + Collector Number (Specific Printing)
        url = f"https://api.scryfall.com/cards/{card['set'].lower()}/{card['collector_number']}"
        params = {}
    else:
        # Fetch by Name + Set
        url = "https://api.scryfall.com/cards/named"
        params = {
            "exact": card['name'],
            "set": card['set']
        }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status() # Raise error for 404, 500 etc
        data = response.json()

        # Validate name if we fetched by collector number to avoid "wrong card" issues
        fetched_name = data.get("name")
        if (card['collector_number'] and fetched_name and
                card['name'].lower() not in fetched_name.lower()):
            print(f"  {Colors.RED}Error: Collector number {card['collector_number']} returned "
                  f"'{fetched_name}', expected '{card['name']}'{Colors.ENDC}")
            return None

        # Scryfall prices are in 'prices' object
        prices = data.get("prices", {})
        price_usd = prices.get("usd")

        if price_usd:
            return float(price_usd)

        print(f"  Warning: No USD price found for {card['name']} ({card['set']})")
        return None

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            cn_info = f" #{card['collector_number']}" if card['collector_number'] else ""
            print(f"  Error: Card not found: {card['name']} ({card['set']}{cn_info})")
        else:
            print(f"  API Error for {card['name']}: {e}")
        return None
    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(f"  Unexpected error for {card['name']}: {e}")
        return None

# --- Recommendation Logic ---

def generate_recommendation(current_price: Optional[float],
                            target_price: Optional[float],
                            history: List[Tuple[Optional[float], str]]) -> str:
    """
    Generates a BUY/WAIT/NEUTRAL recommendation.
    """
    if current_price is None:
        return "UNKNOWN"

    if target_price is not None and current_price <= target_price:
        return "BUY (Below Target)"

    # Check if it's an historical low
    valid_history_prices: List[float] = [h[0] for h in history if h[0] is not None]

    if len(valid_history_prices) >= 3:
        min_hist = min(valid_history_prices)
        if current_price < min_hist:
            return "BUY (New Low)"

    if target_price is not None and current_price > target_price:
        return "WAIT"

    return "NEUTRAL"

# --- Visualization ---

def render_ascii_graph(history: List[Tuple[Optional[float], str]]) -> None:
    """
    Renders a simple ASCII bar chart for the given history.
    History is expected to be a list of (price, date_str).
    """
    if not history:
        return

    # Sort by date ascending (Oldest first)
    chrono_history = sorted(history, key=lambda x: x[1])

    prices: List[float] = [h[0] for h in chrono_history if h[0] is not None]
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

        ascii_bar = '#' * bar_len
        print(f"    {date_short}: {ascii_bar:<30} ${price:.2f}")

# --- Colors ---
class Colors:
    """ANSI color codes for terminal output."""
    # pylint: disable=too-few-public-methods
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

def load_cards_from_file(filepath: str) -> List[CardData]:
    """Loads card list from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Basic validation
            cards: List[CardData] = []
            for item in data:
                cards.append({
                    "name": item.get("name"),
                    "set": item.get("set"),
                    "target_price": item.get("target_price"),
                    "collector_number": item.get("collector_number")
                })
            return cards
    except FileNotFoundError:
        print(f"{Colors.RED}Error: File '{filepath}' not found.{Colors.ENDC}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"{Colors.RED}Error: Failed to parse '{filepath}': {e}{Colors.ENDC}")
        sys.exit(1)

def process_cards(cards: List[CardData], conn: Optional[sqlite3.Connection],
                  use_db: bool) -> None:
    """Iterates through cards and processes them."""
    for card in cards:
        cn_str = f" #{card['collector_number']}" if card['collector_number'] else ""
        print(f"\n{Colors.BOLD}Checking: {card['name']} [{card['set']}{cn_str}]...{Colors.ENDC}")

        # 1. Get History (if DB)
        history: List[Tuple[Optional[float], str]] = []
        if conn:
            history = get_history(conn, card)

        # 2. Get Current Price
        current_price = get_card_price(card)

        # 3. Recommendation
        # Only show recommendation if we have a target price or history
        recommendation = "N/A"
        if use_db or card['target_price'] is not None:
            recommendation = generate_recommendation(current_price, card['target_price'],
                                                     history)

        # Colorize Recommendation
        rec_color = Colors.ENDC
        if "BUY" in recommendation:
            rec_color = Colors.GREEN
        elif "WAIT" in recommendation:
            rec_color = Colors.YELLOW
        elif "UNKNOWN" in recommendation:
            rec_color = Colors.RED

        # 4. Save to DB
        if conn and current_price is not None:
            save_price(conn, card, current_price)

        # 5. Output
        price_str = f"${current_price:.2f}" if current_price else "N/A"
        print(f"  Current Price: {Colors.CYAN}{price_str}{Colors.ENDC}")
        if card['target_price'] is not None:
            print(f"  Target Price:  ${card['target_price']:.2f}")

        if recommendation != "N/A":
            print(f"  Recommendation: {rec_color}{recommendation}{Colors.ENDC}")

        # Refetch history for graph
        if conn:
            history = get_history(conn, card, limit=14)
            if history:
                render_ascii_graph(history)
            else:
                print("  No history available.")

        # Rate limiting behavior
        time.sleep(0.1)

def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description="MTG Price Checker")
    parser.add_argument("--db", help="Path to sqlite database file", default=None)
    parser.add_argument("--list", help="Path to JSON card list file", default=None)
    parser.add_argument("card_info", nargs="*", help="[Name] [Set] [TargetPrice] [CollectorNum]")

    args = parser.parse_args()

    cards_to_check: List[CardData] = []

    # Logic Priority:
    # 1. List file
    if args.list:
        cards_to_check = load_cards_from_file(args.list)

    # 2. Positional Args (Single Card)
    elif args.card_info and len(args.card_info) >= 2:
        name = args.card_info[0]
        set_code = args.card_info[1]
        target_price: Optional[float] = None
        collector_number: Optional[str] = None

        if args.db and len(args.card_info) < 3:
            print(f"{Colors.RED}Error: When using --db, TargetPrice is required "
                  f"for single card mode.{Colors.ENDC}")
            print("Usage: mtg_price_checker.py --db DB 'Name' 'Set' 'TargetPrice' [CollectorNum]")
            sys.exit(1)

        if len(args.card_info) >= 3:
            try:
                target_price = float(args.card_info[2])
            except ValueError:
                print(f"{Colors.RED}Error: TargetPrice must be a number.{Colors.ENDC}")
                print(f"{Colors.YELLOW}Hint: If your card name has spaces, make sure to "
                      f"wrap it in quotes (e.g., \"Black Lotus\").{Colors.ENDC}")
                sys.exit(1)

        if len(args.card_info) >= 4:
            collector_number = args.card_info[3]

        cards_to_check = [{
            "name": name,
            "set": set_code,
            "target_price": target_price,
            "collector_number": collector_number
        }]

    # 3. No inputs -> Error
    else:
        parser.print_help()
        sys.exit(1)

    print(f"{Colors.HEADER}--- MTG Price Fetcher ---{Colors.ENDC}")
    print(f"Timestamp: {datetime.datetime.now()}")

    conn: Optional[sqlite3.Connection] = None
    if args.db:
        conn = setup_database(args.db)

    process_cards(cards_to_check, conn, bool(args.db))

    if conn:
        conn.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
