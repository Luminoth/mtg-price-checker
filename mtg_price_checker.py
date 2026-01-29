#! /usr/bin/env python3

"""
MTG Price Checker
-----------------
A script to fetch and track Magic: The Gathering card prices from Scryfall.
Stores history in a local SQLite database or DynamoDB, and provides BUY/WAIT recommendations.
"""

import sys
import argparse
import json
import sqlite3
import datetime
import time
from decimal import Decimal
from typing import List, Tuple, Optional, Any, TypedDict, Dict
import requests

# Try to import boto3
try:
    import boto3  # type: ignore[import]
    from botocore.exceptions import ClientError  # type: ignore[import]
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

# --- Constants ---

DYNAMODB_TABLE_NAME = "mtg_prices"

# --- Types ---

class CardData(TypedDict):
    """Structure for card data from JSON or CLI."""
    name: str
    set: str
    target_price: Optional[float]
    collector_number: Optional[str]

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

# --- Database Interface ---

class DatabaseInterface:
    """Abstract base class for storage backends."""
    def setup(self) -> None:
        """Prepare the database (create tables, etc)."""
        raise NotImplementedError

    def save_price(self, card: CardData, price_usd: float) -> None:
        """Save a new price entry."""
        raise NotImplementedError

    def get_history(self, card: CardData, limit: int = 5) -> List[Tuple[Optional[float], str]]:
        """Retrieve historical price data."""
        raise NotImplementedError

    def remove_card(self, card_name: str) -> None:
        """Remove all data for a card."""
        raise NotImplementedError

    def close(self) -> None:
        """Close any open connections."""

class SQLiteBackend(DatabaseInterface):
    """SQLite implementation of storage backend."""
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.conn: Optional[sqlite3.Connection] = None

    def setup(self) -> None:
        """Creates the price_history table if it doesn't exist and handles migrations."""
        self.conn = sqlite3.connect(self.db_file)
        cursor = self.conn.cursor()
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
        self.conn.commit()

    def save_price(self, card: CardData, price_usd: float) -> None:
        if not self.conn:
            return
        cursor = self.conn.cursor()

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
        self.conn.commit()

    def get_history(self, card: CardData, limit: int = 5) \
            -> List[Tuple[Optional[float], str]]:
        if not self.conn:
            return []
        cursor = self.conn.cursor()

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

    def remove_card(self, card_name: str) -> None:
        if not self.conn:
            return
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM price_history WHERE card_name = ?", (card_name,))
        deleted_count = cursor.rowcount
        self.conn.commit()
        print(f"{Colors.GREEN}Deleted {deleted_count} entries for card '{card_name}' "
              f"from SQLite.{Colors.ENDC}")

    def close(self) -> None:
        if self.conn:
            self.conn.close()

class DynamoDBBackend(DatabaseInterface):
    """DynamoDB implementation of storage backend."""
    def __init__(self, table_name: str):
        if not BOTO3_AVAILABLE:
            print(f"{Colors.RED}Error: boto3 is not installed. "
                  f"Please install it to use DynamoDB.{Colors.ENDC}")
            sys.exit(1)
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(table_name)

    def setup(self) -> None:
        """Checks table accessibility."""
        # We assume table is created via Terraform/external means, but we can check if it exists
        try:
            self.table.load()
        except ClientError as err:
            print(f"{Colors.RED}Error accessing DynamoDB table '{self.table_name}': "
                  f"{err}{Colors.ENDC}")
            sys.exit(1)

    @staticmethod
    def _get_card_id(card_name: str, set_code: str,
                     collector_number: Optional[str] = None) -> str:
        """Generates a composite key for the card."""
        parts = [card_name, set_code]
        if collector_number:
            parts.append(collector_number)
        return "|".join(parts)

    def save_price(self, card: CardData, price_usd: float) -> None:
        card_id = self._get_card_id(card['name'], card['set'], card['collector_number'])
        now_iso = datetime.datetime.utcnow().isoformat()

        today_prefix = now_iso.split('T')[0]

        try:
            # Check for existing entry today
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('card_id').eq(card_id) &
                                       boto3.dynamodb.conditions.Key('fetched_at')
                                       .begins_with(today_prefix)
            )

            if response.get('Items'):
                print(f"  [Info] Price for {card['name']} already saved today (DynamoDB).")
                return

            # Put Item
            item: Dict[str, Any] = {
                'card_id': card_id,
                'fetched_at': now_iso,
                'card_name': card['name'],
                'set_code': card['set'],
                'price_usd': Decimal(str(price_usd)),
            }
            if card['collector_number']:
                item['collector_number'] = card['collector_number']

            self.table.put_item(Item=item)

        except ClientError as err:
            print(f"  {Colors.RED}DynamoDB Error saving {card['name']}: {err}{Colors.ENDC}")

    def get_history(self, card: CardData, limit: int = 5) -> List[Tuple[Optional[float], str]]:
        card_id = self._get_card_id(card['name'], card['set'], card['collector_number'])

        try:
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('card_id').eq(card_id),
                ScanIndexForward=False, # Descending order
                Limit=limit
            )

            items = response.get('Items', [])
            history: List[Tuple[Optional[float], str]] = []
            for item in items:
                price = float(item['price_usd'])
                fetched_at = item['fetched_at']
                history.append((price, fetched_at))

            return history

        except ClientError as err:
            print(f"  {Colors.RED}DynamoDB Error fetching history for {card['name']}: "
                  f"{err}{Colors.ENDC}")
            return []

    def remove_card(self, card_name: str) -> None:
        print(f"Scanning DynamoDB for items with card_name='{card_name}'...")
        try:
            scan_kwargs: Dict[str, Any] = {
                'FilterExpression': boto3.dynamodb.conditions.Attr('card_name')
                .eq(card_name),
                'ProjectionExpression': 'card_id, fetched_at'
            }
            done = False
            start_key = None
            items_to_delete = []

            while not done:
                if start_key:
                    scan_kwargs['ExclusiveStartKey'] = start_key
                response = self.table.scan(**scan_kwargs)
                items_to_delete.extend(response.get('Items', []))
                start_key = response.get('LastEvaluatedKey', None)
                done = start_key is None

            if not items_to_delete:
                print(f"No items found for '{card_name}' in DynamoDB.")
                return

            print(f"Found {len(items_to_delete)} items. Deleting...")

            with self.table.batch_writer() as batch:
                for item in items_to_delete:
                    batch.delete_item(
                        Key={
                            'card_id': item['card_id'],
                            'fetched_at': item['fetched_at']
                        }
                    )
            print(f"{Colors.GREEN}Successfully deleted {len(items_to_delete)} items "
                  f"from DynamoDB.{Colors.ENDC}")

        except ClientError as err:
            print(f"{Colors.RED}DynamoDB Error deleting {card_name}: {err}{Colors.ENDC}")

# --- API Functions ---

def get_card_price(card: CardData) -> Optional[float]:
    """
    Fetches the current non-foil market price for a specific card printing from Scryfall.
    Returns: price (float) or None if not found/no price.
    """
    if card['collector_number']:
        # Fetch by Set + Collector Number (Specific Printing)
        url = f"https://api.scryfall.com/cards/{card['set'].lower()}/" \
              f"{card['collector_number']}"
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

    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 404:
            cn_info = f" #{card['collector_number']}" if card['collector_number'] else ""
            print(f"  Error: Card not found: {card['name']} ({card['set']}{cn_info})")
        else:
            print(f"  API Error for {card['name']}: {err}")
        return None
    except Exception as err:  # pylint: disable=broad-except
        print(f"  Unexpected error for {card['name']}: {err}")
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

    # Check for strict downward trend (Current < Prev < PrevPrev)
    valid_history_prices: List[float] = [h[0] for h in history if h[0] is not None]

    if valid_history_prices:
        # If the latest history entry is the same as the current price (e.g., already saved today),
        # skip it so we compare against the *previous* day's price.
        comp_idx = 0
        if abs(valid_history_prices[0] - current_price) < 0.001:
            comp_idx = 1

        # We need at least 2 MORE points to compare against
        if len(valid_history_prices) >= comp_idx + 2:
            last_db_price = valid_history_prices[comp_idx]
            prev_db_price = valid_history_prices[comp_idx + 1]

            if current_price < last_db_price < prev_db_price:
                return "WATCH (Falling Price)"

    if target_price is not None and current_price <= target_price:
        return "BUY (Below Target)"

    if len(valid_history_prices) >= 3:
        min_hist = min(valid_history_prices)
        if current_price < min_hist:
            return "BUY (New Low)"

    if target_price is not None and current_price > target_price:
        return "WAIT"

    return "NEUTRAL"

# --- Visualization ---

def _parse_date_to_local(date_str: str) -> str:
    """Helper to convert UTC date string to local date string."""
    try:
        # Handle potential formats: "YYYY-MM-DD HH:MM:SS" (SQLite) or ISO (DynamoDB)
        if "T" in date_str:
            dt_utc = datetime.datetime.fromisoformat(date_str)
        else:
            dt_utc = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

        # Assume stored time is UTC. Convert to local.
        dt_utc = dt_utc.replace(tzinfo=datetime.timezone.utc)
        dt_local = dt_utc.astimezone()
        return dt_local.strftime('%Y-%m-%d')
    except ValueError:
        return date_str.split(' ')[0]

def render_ascii_graph(history: List[Tuple[Optional[float], str]]) -> None:
    """
    Renders a simple ASCII bar chart for the given history.
    History is expected to be a list of (price, date_str).
    Dates are converted from stored UTC to local time for display.
    """
    # pylint: disable=too-many-locals
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

    print("  Price History (Trend) - Local Time:")
    max_bar_width = 30

    previous_price: Optional[float] = None
    lines: List[str] = []

    for price, date_str in chrono_history:
        if price is None:
            continue

        display_date = _parse_date_to_local(date_str)

        if distinct_range == 0:
            bar_len = max_bar_width // 2
        else:
            pct = (price - min_p) / distinct_range
            bar_len = 1 + int(pct * (max_bar_width - 1))

        # Determine color
        bar_color = Colors.ENDC
        if previous_price is not None:
            if price > previous_price:
                bar_color = Colors.RED
            elif price < previous_price:
                bar_color = Colors.GREEN
            else:
                bar_color = Colors.YELLOW

        previous_price = price

        ascii_bar = '#' * bar_len
        lines.append(f"    {display_date}: {bar_color}{ascii_bar:<30}{Colors.ENDC} ${price:.2f}")

    # Print Newest -> Oldest
    for line in reversed(lines):
        print(line)

# --- Main Execution ---

def load_cards_from_file(filepath: str) -> List[CardData]:
    """Loads card list from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
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
    except json.JSONDecodeError as err:
        print(f"{Colors.RED}Error: Failed to parse '{filepath}': {err}{Colors.ENDC}")
        sys.exit(1)

def process_cards(cards: List[CardData], database: Optional[DatabaseInterface],
                  use_db: bool) -> None:
    """Iterates through cards and processes them."""
    # pylint: disable=too-many-branches
    for card in cards:
        cn_str = f" #{card['collector_number']}" if card['collector_number'] else ""
        print(f"\n{Colors.BOLD}Checking: {card['name']} [{card['set']}{cn_str}]...{Colors.ENDC}")

        # 1. Get History (if DB)
        history: List[Tuple[Optional[float], str]] = []
        if database:
            history = database.get_history(card)

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
        elif "WATCH" in recommendation:
            rec_color = Colors.BLUE
        elif "WAIT" in recommendation:
            rec_color = Colors.YELLOW
        elif "UNKNOWN" in recommendation:
            rec_color = Colors.RED

        # 4. Save to DB
        if database and current_price is not None:
            database.save_price(card, current_price)

        # 5. Output
        price_color = Colors.CYAN
        if card['target_price'] is not None and current_price is not None:
            if current_price <= card['target_price']:
                price_color = Colors.GREEN
            else:
                price_color = Colors.RED

        price_str = f"${current_price:.2f}" if current_price else "N/A"
        print(f"  Current Price: {price_color}{price_str}{Colors.ENDC}")
        if card['target_price'] is not None:
            print(f"  Target Price:  ${card['target_price']:.2f}")

        if recommendation != "N/A":
            print(f"  Recommendation: {rec_color}{recommendation}{Colors.ENDC}")

        # Refetch history for graph
        if database:
            history = database.get_history(card, limit=14)
            if history:
                render_ascii_graph(history)
            else:
                print("  No history available.")

        # Rate limiting behavior
        time.sleep(0.1)

def handle_remove(args: argparse.Namespace, database: DatabaseInterface) -> None:
    """Handles the 'remove' subcommand logic."""
    if not args.force:
        confirm = input(
            f"{Colors.YELLOW}Are you sure you want to delete all entries for "
            f"'{args.name}'? [y/N] {Colors.ENDC}"
        )
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return

    database.remove_card(args.name)

def handle_check(args: argparse.Namespace, database: DatabaseInterface) -> None:
    """Handles the 'check' subcommand logic."""
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

        # Validate Target Price if provided
        if len(args.card_info) >= 3:
            try:
                target_price = float(args.card_info[2])
            except ValueError:
                print(f"{Colors.RED}Error: TargetPrice must be a number.{Colors.ENDC}")
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
    # 3. No inputs -> Error
    else:
        print(f"{Colors.RED}Error: You must provide a list file (--list) "
              f"or card info arguments.{Colors.ENDC}")
        sys.exit(1)

    process_cards(cards_to_check, database, bool(database))

def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description="MTG Price Checker")
    parser.add_argument("--database", choices=["sqlite", "dynamodb"], required=True,
                        help="Database backend to use")
    parser.add_argument("--sqlite-path",
                        help="Path to sqlite database file (default: prices.db)",
                        default="prices.db")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # Subparser: Check
    parser_check = subparsers.add_parser("check", help="Check card prices")
    parser_check.add_argument("--list", help="Path to JSON card list file", default=None)
    parser_check.add_argument("card_info", nargs="*",
                              help="[Name] [Set] [TargetPrice] [CollectorNum]")

    # Subparser: Remove
    parser_remove = subparsers.add_parser("remove",
                                          help="Remove a card from the database")
    parser_remove.add_argument("name", help="Name of the card to remove")
    parser_remove.add_argument("--force", action="store_true",
                               help="Force removal without confirmation")

    args = parser.parse_args()

    # --- Initialize Database ---
    database: Optional[DatabaseInterface] = None
    if args.database == "dynamodb":
        print("[Mode] DynamoDB")
        database = DynamoDBBackend(DYNAMODB_TABLE_NAME)
        database.setup()
    elif args.database == "sqlite":
        print(f"[Mode] SQLite ({args.sqlite_path})")
        database = SQLiteBackend(args.sqlite_path)
        database.setup()

    print(f"{Colors.HEADER}--- MTG Price Fetcher ---{Colors.ENDC}")
    print(f"Timestamp: {datetime.datetime.now()}")

    if database:
        if args.command == "check":
            handle_check(args, database)
        elif args.command == "remove":
            handle_remove(args, database)
        database.close()
    else:
        # Should not happen because database is required and we exit if setup fails
        # but satisfies mypy if we put logical flow together
        pass

    print("\nDone.")

if __name__ == "__main__":
    main()
