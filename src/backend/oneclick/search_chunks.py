import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add backend directory to path to allow imports
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from oneclick.helpers import find_chunks_with_text

def main():
    # Load environment variables
    load_dotenv()

    # Configuration - Replace these with your actual values or pass as args
    SEARCH_TERM = "TOO FAST"
    CONTRACT_ID = "deca5157-f783-47fe-9140-772febe60105" # Replace with your contract ID
    USER_ID = "0cd78285-efaa-449b-b1f1-8d19f7ebec2e"         # Replace with your user ID
    INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "test-3-small-index")

    print(f"Searching for '{SEARCH_TERM}' in contract {CONTRACT_ID}...")
    print(f"User ID: {USER_ID}")
    print(f"Index: {INDEX_NAME}")
    print("-" * 50)

    matches = find_chunks_with_text(
        search_term=SEARCH_TERM,
        contract_id=CONTRACT_ID,
        user_id=USER_ID,
        index_name=INDEX_NAME
    )

    if not matches:
        print("No matches found.")
    else:
        print(f"Found {len(matches)} matches:")
        for i, match in enumerate(matches):
            print(f"\nMatch #{i+1}")
            print(f"ID: {match['id']}")
            print(f"Score: {match['score']:.4f}")
            print(f"Section: {match['section']}")
            print(f"Text snippet: ...{match['text'][:200]}...") # Print first 200 chars
            print("-" * 30)

if __name__ == "__main__":
    main()

