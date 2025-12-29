# OneClick Royalty Calculator

Automated royalty payment calculation from music contracts and streaming statements.

## Overview

OneClick analyzes music contracts using AI and calculates royalty payments by matching contract terms with streaming revenue data. It eliminates manual spreadsheet work and reduces errors in royalty distribution.

## How It Works

### 1. **Contract Analysis**
- Contracts are uploaded and stored in Pinecone vector database
- AI extracts key information:
  - Party names (artists, producers, songwriters)
  - Song titles covered by the contract
  - Royalty percentages for each party
  - Royalty types (streaming, mechanical, sync, etc.)

### 2. **Royalty Statement Processing**
- Supports both CSV and Excel formats
- Auto-detects column names for song titles and revenue amounts
- Aggregates revenue by song title across all rows

### 3. **Payment Calculation**
- Matches songs from contracts to songs in royalty statements (fuzzy matching)
- Applies each party's percentage to the total revenue per song
- Generates detailed payment breakdown showing:
  - Who gets paid
  - How much they receive
  - Which songs generated the revenue
  - What percentage they're entitled to

### 4. **Output**
- Returns structured payment data via API
- Can export to Excel with formatted breakdown
- Includes summary totals and validation

## Key Features

- **Smart Matching**: Fuzzy matching handles variations in song titles
- **Multi-Contract Support**: Merge multiple contracts for the same project
- **Auto-Detection**: Automatically identifies relevant columns in statements
- **Validation**: Checks for missing data and provides helpful error messages
- **Flexible Input**: Accepts various royalty statement formats

## File Structure

```
oneclick/
├── contract_parser.py      # AI-powered contract extraction
├── royalty_calculator.py   # Payment calculation engine
├── helpers.py              # Utility functions (matching, normalization)
└── README.md              # This file
```

## Usage Example

```python
from oneclick.royalty_calculator import RoyaltyCalculator

calculator = RoyaltyCalculator()

# Calculate payments
payments = calculator.calculate_payments(
    contract_path="contract.pdf",
    statement_path="royalties.csv",
    user_id="user-123",
    contract_id="contract-456"
)

# Export to Excel
calculator.save_payments_to_excel(payments, "output.xlsx")
```

## API Integration

The OneClick functionality is exposed via FastAPI endpoint:

**POST** `/oneclick/calculate-royalties`

**Request:**
```json
{
  "contract_id": "uuid",
  "user_id": "uuid",
  "project_id": "uuid",
  "royalty_statement_file_id": "uuid"
}
```

**Response:**
```json
{
  "status": "success",
  "total_payments": 6,
  "payments": [
    {
      "song_title": "Midnight Dreams",
      "party_name": "Luna Rivers",
      "role": "Artist",
      "royalty_type": "Streaming Royalties",
      "percentage": 45.0,
      "total_royalty": 125000.00,
      "amount_to_pay": 56250.00
    }
  ]
}
```

## Requirements

- OpenAI API key (for contract parsing)
- Pinecone API key (for vector storage)
- Python packages: `openpyxl`, `openai`, `python-dotenv`, `PyMuPDF`

## Error Handling

The system validates:
- Contract contains royalty information
- Royalty statement has valid data
- Songs in contract match songs in statement
- All required fields are present

Common errors and solutions are logged with helpful messages to guide troubleshooting.
