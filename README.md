# Magic: The Gathering Price Checker

A CLI tool to track and monitor Magic: The Gathering card prices using the [Scryfall API](https://scryfall.com/docs/api).

This tool allows you to:
- Fetch current market prices for cards.
- Store price history in a local **SQLite** database or **AWS DynamoDB**.
- Get BUY/WAIT/WATCH recommendations based on price trends and target prices.
- Visualize price trends with ASCII graphs.

## Prerequisites

- Python 3.8+
- [Terraform](https://www.terraform.io/) (if using DynamoDB)
- AWS Credentials configured (if using DynamoDB)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mtg-price-checker.git
   cd mtg-price-checker
   ```

2. Install dependencies:
   ```bash
   pip install requests boto3
   ```

3. (Optional) Set up DynamoDB:
   If you want to use DynamoDB, initialize the Terraform configuration:
   ```bash
   terraform init
   terraform apply
   ```

## Usage

The script uses subcommands: `check` to verify prices and `remove` to delete data.

### 1. Check Prices (`check`)

**Using SQLite (Default):**
```bash
# Check a single card
# Syntax: check [Name] [Set] [TargetPrice] [CollectorNum (optional)]
python3 mtg_price_checker.py --database sqlite check "Black Lotus" "LEA" 20000

# Check a list of cards from a JSON file
python3 mtg_price_checker.py --database sqlite check --list my_cards.json
```

**Using DynamoDB:**
```bash
python3 mtg_price_checker.py --database dynamodb check "Force of Will" "ALL" 80
```

### 2. Remove Card Data (`remove`)

Removes all price history for a specific card from the database.

**Using SQLite:**
```bash
python3 mtg_price_checker.py --database sqlite remove "Black Lotus"
# You will be prompted for confirmation.
```

**Using DynamoDB (Force delete):**
```bash
# Use --force to skip confirmation
python3 mtg_price_checker.py --database dynamodb remove "Black Lotus" --force
```

## Arguments

### Global Options
- `--database {sqlite,dynamodb}`: **Required**. Selects the database backend.
- `--sqlite-path PATH`: Path to the SQLite database file (default: `prices.db`).

### Subcommands

#### `check`
- `--list PATH`: Path to a JSON file containing a list of cards to check.
- `[Name] [Set] [TargetPrice] [CollectorNum]`: Positional arguments for checking a single card.

#### `remove`
- `name`: The name of the card to remove.
- `--force`: Skip the confirmation prompt.

## JSON List Format

If using `--list`, the JSON file should look like this:

```json
[
  {
    "name": "Black Lotus",
    "set": "LEA",
    "target_price": 20000.00
  },
  {
    "name": "Maha, Its Feathers Night",
    "set": "BLB",
    "target_price": 20.00,
    "collector_number": "289"
  }
]
```
