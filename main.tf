provider "aws" {
  region = "us-east-1"
}

resource "aws_dynamodb_table" "mtg_prices" {
  name         = "mtg_prices"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "card_id"
  range_key    = "fetched_at"

  attribute {
    name = "card_id"
    type = "S"
  }

  attribute {
    name = "fetched_at"
    type = "S"
  }

  tags = {
    Name        = "mtg-prices-table"
    Environment = "production"
  }
}
