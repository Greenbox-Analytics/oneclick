"""Thin OneClick wrappers used by `main.py` for the royalty-calculation endpoints.

The PDF, embedding, section, and table helpers that used to live here moved to
`utils/ingestion/*`. The shared OpenAI client moved to `utils/llm/client.py`.
What remains is purpose-built glue around `oneclick.royalty_calculator` —
`main.py` and a small number of tests still import these names from here.
"""

import os
from dataclasses import asdict

from dotenv import load_dotenv

from oneclick.royalty_calculator import RoyaltyCalculator

# Load environment variables
load_dotenv()


def calculate_royalty_payments(
    contract_path: str,
    statement_path: str,
    user_id: str,
    contract_id: str,
    api_key: str = None,
    contract_ids: list[str] = None,
    contract_markdowns: dict[str, str] = None,
    expenses: list[dict] = None,
    contract_data_by_id: dict = None,
) -> tuple[list[dict], dict | None]:
    """
    Calculate royalty payments from a contract and royalty statement.

    This is a helper function that wraps the RoyaltyCalculator to provide
    a simpler interface for the OneClick feature.

    Args:
        contract_path: Path to the contract PDF file (not used, kept for compatibility)
        statement_path: Path to the royalty statement Excel file
        user_id: User ID for querying Pinecone
        contract_id: Contract ID for querying Pinecone (single)
        api_key: Optional OpenAI API key (uses env var if not provided)
        contract_ids: Optional list of contract IDs for multi-contract calculation

    Returns:
        Tuple of (payment_dicts, review_dict):
            - payment_dicts: list of payment dictionaries with keys:
                - song_title: str
                - party_name: str
                - role: str
                - royalty_type: str
                - percentage: float
                - total_royalty: float
                - amount_to_pay: float
                - terms: Optional[str]
            - review_dict: the advisory split-verification outcome as a dict
              (overall/checked/flagged/findings), or None when the pass
              didn't run. Advisory only — never changes any payout.
    """
    # Initialize calculator
    calculator = RoyaltyCalculator(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    # Calculate payments
    if contract_ids and len(contract_ids) > 0:
        # Multi-contract mode
        output = calculator.calculate_payments_from_contract_ids(
            contract_ids=contract_ids,
            user_id=user_id,
            statement_path=statement_path,
            contract_markdowns=contract_markdowns,
            expenses=expenses,
            contract_data_by_id=contract_data_by_id,
        )
    else:
        # Single contract mode
        full_text = contract_markdowns.get(contract_id) if contract_markdowns and contract_id else None
        contract_data = contract_data_by_id.get(contract_id) if contract_data_by_id and contract_id else None
        output = calculator.calculate_payments(
            contract_path=contract_path,
            statement_path=statement_path,
            full_text=full_text,
            contract_id=contract_id,
            user_id=user_id,
            expenses=expenses,
            contract_data=contract_data,
        )

    # Convert to dictionaries for easier JSON serialization
    payment_dicts = []
    for payment in output.payments:
        payment_dicts.append(
            {
                "song_title": payment.song_title,
                "party_name": payment.party_name,
                "role": payment.role,
                "royalty_type": payment.royalty_type,
                "percentage": payment.percentage,
                "total_royalty": payment.total_royalty,
                "amount_to_pay": payment.amount_to_pay,
                "terms": payment.terms,
                "basis": payment.basis,
                "gross_amount": payment.gross_amount,
                "expenses_applied": payment.expenses_applied,
                "net_amount": payment.net_amount,
                "source_contract_ids": list(getattr(payment, "source_contract_ids", []) or []),
            }
        )

    review_dict = asdict(output.review) if output.review is not None else None
    return payment_dicts, review_dict


def save_royalty_payments_to_excel(payments: list[dict], output_path: str, api_key: str = None) -> None:
    """
    Save royalty payments to an Excel file with formatting.

    Args:
        payments: List of payment dictionaries (from calculate_royalty_payments)
        output_path: Path where Excel file should be saved
        api_key: Optional OpenAI API key (uses env var if not provided)
    """
    from oneclick.royalty_calculator import RoyaltyCalculator, RoyaltyPayment

    # Initialize calculator
    calculator = RoyaltyCalculator(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    # Convert dictionaries back to RoyaltyPayment objects
    payment_objects = []
    for p in payments:
        payment_objects.append(
            RoyaltyPayment(
                song_title=p["song_title"],
                party_name=p["party_name"],
                role=p["role"],
                royalty_type=p["royalty_type"],
                percentage=p["percentage"],
                total_royalty=p["total_royalty"],
                amount_to_pay=p["amount_to_pay"],
                terms=p.get("terms"),
            )
        )

    # Save to Excel
    calculator.save_payments_to_excel(payment_objects, output_path)
