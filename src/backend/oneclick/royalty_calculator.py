"""
Royalty Payment Calculator
Calculates payments from royalty statements and music contracts

Installation:
pip install openpyxl openai python-dotenv PyMuPDF

Usage:
    from royalty_calculator import RoyaltyCalculator
    
    calculator = RoyaltyCalculator()
    payments = calculator.calculate_payments("contract.pdf", "statement.xlsx")
"""

import os
import json
import csv
import time
import difflib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from .contract_parser import MusicContractParser, ContractData
from .helpers import normalize_title, find_matching_song, normalize_name, simplify_role

import openpyxl
from dotenv import load_dotenv
from openai import OpenAI

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()




@dataclass
class RoyaltyPayment:
    """Represents a calculated royalty payment"""
    song_title: str
    party_name: str
    role: str
    royalty_type: str
    percentage: float
    total_royalty: float
    amount_to_pay: float
    terms: Optional[str] = None


class RoyaltyCalculator:
    """
    Enhanced calculator for royalty payments from statements and contracts.
    
    Features:
    - Auto-detects columns in royalty statements
    - Fuzzy matching between contract works and statement songs
    - Support for single or multiple contracts
    - Intelligent merging of multiple contracts
    - Comprehensive validation and reporting
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the calculator with OpenAI API key.
        
        Args:
            api_key: Optional OpenAI API key. If not provided, will look in environment variables.
        """
        # Load API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key or not self.api_key.startswith("sk-"):
            raise ValueError(
                "❌ OpenAI API key missing or invalid. "
                "Please add it to .env file or pass as parameter."
            )
        
        # Initialize contract parser with explicit API key
        self.contract_parser = MusicContractParser(api_key=self.api_key)
    
    # ========================================================================
    # ROYALTY STATEMENT READING
    # ========================================================================
    
    def read_royalty_statement(
        self, 
        excel_path: str, 
        title_column: Optional[str] = None,
        payable_column: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Read streaming royalty statement and calculate total per song.
        
        Supports both CSV and Excel files. Auto-detects column names if not specified.
        
        Args:
            excel_path: Path to the royalty statement (CSV or Excel)
            title_column: Name of column containing song titles (auto-detects if None)
            payable_column: Name of column containing net payable amounts (auto-detects if None)
            
        Returns:
            Dictionary mapping song titles to total net payable amounts
        """
        try:
            logger.info(f"\n📊 Reading royalty statement: {Path(excel_path).name}")
            
            # Detect file type
            file_ext = Path(excel_path).suffix.lower()
            
            if file_ext == '.csv':
                return self._read_csv_statement(excel_path, title_column, payable_column)
            elif file_ext in ['.xlsx', '.xls']:
                return self._read_excel_statement(excel_path, title_column, payable_column)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}. Please use CSV or Excel files.")
            
        except Exception as e:
            raise Exception(f"Error reading royalty statement: {str(e)}")
    
    def _read_csv_statement(
        self,
        csv_path: str,
        title_column: Optional[str] = None,
        payable_column: Optional[str] = None
    ) -> Dict[str, float]:
        """Read royalty statement from CSV file"""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Get headers (keys from DictReader)
            headers = [h.strip().lower() for h in reader.fieldnames]
            
            logger.info(f"   Found {len(headers)} columns: {', '.join(headers[:5])}...")
            
            # Auto-detect columns if not specified
            if title_column is None:
                title_column = self._find_title_column(headers)
                logger.info(f"   ✓ Auto-detected title column: '{title_column}'")
            else:
                title_column = title_column.lower()
            
            if payable_column is None:
                payable_column = self._find_payable_column(headers)
                logger.info(f"   ✓ Auto-detected payable column: '{payable_column}'")
            else:
                payable_column = payable_column.lower()
            
            # Find original column names (case-sensitive)
            original_headers = {h.strip().lower(): h for h in reader.fieldnames}
            title_col_original = original_headers.get(title_column)
            payable_col_original = original_headers.get(payable_column)
            
            if not title_col_original or not payable_col_original:
                raise ValueError(
                    f"Could not find required columns.\n"
                    f"Looking for: '{title_column}' and '{payable_column}'\n"
                    f"Available columns: {list(original_headers.values())}"
                )
            
            # Read data and sum by song title
            song_totals = {}
            rows_processed = 0
            rows_skipped = 0
            
            for row in reader:
                title_val = row.get(title_col_original, '').strip()
                payable_val = row.get(payable_col_original, '')
                
                if title_val and payable_val:
                    try:
                        # Handle comma-separated numbers (e.g., "1,555")
                        payable_clean = payable_val.replace(',', '')
                        amount = float(payable_clean)
                        song_totals[title_val] = song_totals.get(title_val, 0.0) + amount
                        rows_processed += 1
                    except (ValueError, TypeError):
                        rows_skipped += 1
                        continue
                else:
                    rows_skipped += 1
            
            logger.info(f"   ✓ Processed {rows_processed} rows ({rows_skipped} skipped)")
            logger.info(f"   ✓ Found {len(song_totals)} unique songs")
            
            if not song_totals:
                raise ValueError("No valid royalty data found in statement")
            
            return song_totals
    
    def _read_excel_statement(
        self,
        excel_path: str,
        title_column: Optional[str] = None,
        payable_column: Optional[str] = None
    ) -> Dict[str, float]:
        """Read royalty statement from Excel file"""
        workbook = openpyxl.load_workbook(excel_path, data_only=True)
        sheet = workbook.active
        
        # Extract headers from first row
        headers = []
        for cell in sheet[1]:
            if cell.value:
                headers.append(str(cell.value).strip().lower())
        
        logger.info(f"   Found {len(headers)} columns: {', '.join(headers[:5])}...")
        
        # Auto-detect columns if not specified
        if title_column is None:
            title_column = self._find_title_column(headers)
            logger.info(f"   ✓ Auto-detected title column: '{title_column}'")
        else:
            title_column = title_column.lower()
        
        if payable_column is None:
            payable_column = self._find_payable_column(headers)
            logger.info(f"   ✓ Auto-detected payable column: '{payable_column}'")
        else:
            payable_column = payable_column.lower()
        
        # Find column indices
        try:
            title_idx = headers.index(title_column)
            payable_idx = headers.index(payable_column)
        except ValueError:
            raise ValueError(
                f"Could not find required columns.\n"
                f"Looking for: '{title_column}' and '{payable_column}'\n"
                f"Available columns: {headers}"
            )
        
        # Read data and sum by song title
        song_totals = {}
        rows_processed = 0
        rows_skipped = 0
        
        for row in sheet.iter_rows(min_row=2, values_only=True):
            if row[title_idx] and row[payable_idx] is not None:
                title = str(row[title_idx]).strip()
                try:
                    amount = float(row[payable_idx])
                    song_totals[title] = song_totals.get(title, 0.0) + amount
                    rows_processed += 1
                except (ValueError, TypeError):
                    rows_skipped += 1
                    continue
            else:
                rows_skipped += 1
        
        workbook.close()
        
        logger.info(f"   ✓ Processed {rows_processed} rows ({rows_skipped} skipped)")
        logger.info(f"   ✓ Found {len(song_totals)} unique songs")
        
        if not song_totals:
            raise ValueError("No valid royalty data found in statement")
        
        return song_totals
    
    def _find_title_column(self, headers: List[str]) -> str:
        """Auto-detect the title column from headers"""
        title_variations = [
            'release title',
            'title',
            'song title',
            'track title',
            'song name',
            'track name',
            'release name',
            'track',
            'song'
        ]
        
        for header in headers:
            header_clean = header.lower().strip()
            for var in title_variations:
                if var == header_clean or var in header_clean:
                    return header
        
        raise ValueError(
            f"Could not auto-detect title column.\n"
            f"Available columns: {headers}\n"
            f"Please specify title_column parameter explicitly."
        )
    
    def _find_payable_column(self, headers: List[str]) -> str:
        """
        Auto-detect the net payable column from headers using 3 layers:
        1. Keyword matching (exact/partial)
        2. Fuzzy matching
        3. Semantic search (LLM)
        """
        
        # --- Layer 1: Keyword Matching ---
        
        # Priority variations (more specific matches first)
        priority_variations = [
            'net payable',
            'net payment',
            'net earnings',
            'net pay'
            'total payable',
            'net revenue',
            'net amount',
            'payable to artist',
            'artist payable'
        ]
        
        # Check priority variations first
        for header in headers:
            header_clean = header.lower().strip()
            for var in priority_variations:
                if var == header_clean or var in header_clean:
                    # Exclude withheld/deduction columns
                    if 'withheld' not in header_clean and 'deduction' not in header_clean:
                        return header
        
        # Fallback to general variations
        general_variations = ['payable', 'amount', 'earnings', 'payment', 'revenue']
        
        for header in headers:
            header_clean = header.lower().strip()
            for var in general_variations:
                if var in header_clean:
                    # Exclude unwanted columns
                    excluded_terms = ['withheld', 'deduction', 'fee', 'commission', 'advance']
                    if not any(term in header_clean for term in excluded_terms):
                        return header

        logger.info("   ⚠️  Layer 1 (Keyword) failed to find payable column. Trying Layer 2 (Fuzzy)...")

        # --- Layer 2: Fuzzy Matching ---
        
        target_terms = ['net payable', 'net amount', 'payable', 'total payable', 'royalty amount', 'net pay']
        
        # Get close matches for each target term against all headers
        # We use a cutoff of 0.8 for high confidence
        best_match = None
        highest_ratio = 0.0
        
        for header in headers:
            header_clean = header.lower().strip()
            # Skip likely irrelevant columns to avoid false positives
            if any(x in header_clean for x in ['date', 'isrc', 'upc', 'territory', 'country', 'label', 'artist', 'title']):
                continue
                
            for target in target_terms:
                ratio = difflib.SequenceMatcher(None, header_clean, target).ratio()
                if ratio > highest_ratio and ratio > 0.8:  # 80% similarity threshold
                    highest_ratio = ratio
                    best_match = header
        
        if best_match:
            logger.info(f"   ✓ Layer 2 (Fuzzy) detected payable column: '{best_match}' (confidence: {highest_ratio:.2f})")
            return best_match

        logger.info("   ⚠️  Layer 2 (Fuzzy) failed. Trying Layer 3 (Semantic)...")

        # --- Layer 3: Semantic Search (LLM) ---
        
        if self.api_key:
            try:
                client = OpenAI(api_key=self.api_key)
                
                prompt = (
                    f"Given these column headers from a music royalty statement: {headers}\\n\\n"
                    "Identify the single column that represents the 'Net Payable Amount' or 'Royalty Amount' "
                    "that should be paid to the licensor/artist. "
                    "Ignore columns representing gross revenue, fees, taxes, or deductions unless they are the only option.\\n"
                    "Return ONLY the exact column name from the list. If none match, return 'None'."
                )
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini", # Use a fast/cheap model
                    messages=[
                        {"role": "system", "content": "You are a data analyst helper. Output only the requested column name."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                
                result = response.choices[0].message.content.strip()
                
                # Verify the result is actually in the headers
                # (LLM might strip quotes or change case slightly, so we check loosely)
                if result and result.lower() != 'none':
                    # Try to find exact match first
                    if result in headers:
                        logger.info(f"   ✓ Layer 3 (Semantic) detected payable column: '{result}'")
                        return result
                    
                    # Try case-insensitive match
                    for h in headers:
                        if h.lower() == result.lower():
                            logger.info(f"   ✓ Layer 3 (Semantic) detected payable column: '{h}'")
                            return h
                            
            except Exception as e:
                logger.warning(f"   ⚠️  Layer 3 (Semantic) error: {e}")
        
        raise ValueError(
            f"Could not auto-detect payable column after 3 layers of search.\\n"
            f"Available columns: {headers}\\n"
            f"Please specify payable_column parameter explicitly."
        )
    
    # ========================================================================
    # CONTRACT MERGING
    # ========================================================================
    
    def merge_contracts(self, contracts: List[ContractData]) -> ContractData:
        """
        Intelligently merge multiple ContractData objects.
        
        Handles:
        - Name normalization and deduplication
        - Role preservation
        - Royalty share conflict resolution
        - Summary combination
        
        Args:
            contracts: List of ContractData objects to merge
            
        Returns:
            Single merged ContractData object
        """
        if not contracts:
            raise ValueError("No contracts provided to merge")
        
        if len(contracts) == 1:
            return contracts[0]
        
        logger.info(f"\n🔄 Merging {len(contracts)} contracts...")
        
        merged_parties = []
        merged_works = []
        merged_royalty_shares = []
        summaries = []
        
        # Track seen items for deduplication
        seen_parties = {}  # normalized_name -> Party
        seen_works = {}    # normalized_title -> Work
        seen_shares = {}   # (party, type) -> RoyaltyShare
        
        for idx, contract in enumerate(contracts, 1):
            logger.info(f"   Processing contract {idx}/{len(contracts)}...")
            
            if contract.contract_summary:
                summaries.append(contract.contract_summary)
            
            # Merge parties
            for party in contract.parties:
                norm_name = normalize_name(party.name)
                if norm_name:
                    if norm_name not in seen_parties:
                        seen_parties[norm_name] = party
                        merged_parties.append(party)
                    else:
                        existing = seen_parties[norm_name]
                        # Combine roles from both contracts
                        existing_roles = set(r.strip() for r in existing.role.split(";"))
                        new_roles = set(r.strip() for r in party.role.split(";"))
                        combined = existing_roles | new_roles
                        # Remove generic 'party' if a specific role exists
                        if len(combined) > 1:
                            combined.discard('party')
                        existing.role = "; ".join(sorted(combined))
            
            # Merge works
            for work in contract.works:
                norm_title = normalize_title(work.title)
                if norm_title:
                    if norm_title not in seen_works:
                        seen_works[norm_title] = work
                        merged_works.append(work)
                    else:
                        # Update work_type if current one is more specific
                        existing = seen_works[norm_title]
                        if work.work_type != 'work' and existing.work_type == 'work':
                            existing.work_type = work.work_type
            
            # Merge royalty shares with conflict resolution
            for share in contract.royalty_shares:
                norm_name = normalize_name(share.party_name)
                
                # Check for existing share with same name AND similar type
                existing_share_for_party = None
                
                # Determine if current share is streaming-related
                is_streaming_share = 'streaming' in share.royalty_type.lower() or 'digital' in share.royalty_type.lower()
                
                for existing in merged_royalty_shares:
                    if normalize_name(existing.party_name) == norm_name:
                        # Check if percentages match
                        if abs(existing.percentage - share.percentage) < 0.01:
                             # CRITICAL: Only consider it a duplicate if the royalty TYPE is also similar.
                             # If one is "Publishing" and one is "Streaming", they are different entitlements
                             # even if they have the same percentage (e.g. 50% Pub / 50% Master).
                             
                             is_existing_streaming = 'streaming' in existing.royalty_type.lower() or 'digital' in existing.royalty_type.lower()
                             
                             # If both are streaming or both are NOT streaming (e.g. both publishing), likely a duplicate
                             if is_streaming_share == is_existing_streaming:
                                 existing_share_for_party = existing
                                 break
                
                if existing_share_for_party:
                    logger.info(f"      ℹ️  Duplicate share found for {share.party_name} ({share.percentage}%) - skipping")
                    continue

                # If no exact duplicate found, add it
                merged_royalty_shares.append(share)
        
        # Simplify combined roles
        for party in merged_parties:
            party.role = simplify_role(party.role)
        
        # Combine summaries
        merged_summary = "\n\n".join([s for s in summaries if s.strip()])
        
        logger.info(f"   ✓ Merged to: {len(merged_parties)} parties, {len(merged_works)} works, {len(merged_royalty_shares)} royalty entries")
        
        return ContractData(
            parties=merged_parties,
            works=merged_works,
            royalty_shares=merged_royalty_shares,
            contract_summary=merged_summary if merged_summary else None
        )
    
    
    # ========================================================================
    # PAYMENT CALCULATION
    # ========================================================================
    
    def calculate_payments(
        self,
        contract_path: str,
        statement_path: str,
        user_id: str = None,
        contract_id: str = None,
        title_column: Optional[str] = None,
        payable_column: Optional[str] = None
    ) -> List[RoyaltyPayment]:
        """
        Calculate payments for single contract and statement.
        
        Args:
            contract_path: Path to the contract file (not used, kept for compatibility)
            statement_path: Path to the royalty statement (Excel)
            user_id: User ID for querying Pinecone
            contract_id: Contract ID for querying Pinecone
            title_column: Optional - Name of title column in statement
            payable_column: Optional - Name of payable column in statement
            
        Returns:
            List of RoyaltyPayment objects with calculated amounts
        """
        logger.info("\n" + "="*80)
        logger.info("ROYALTY PAYMENT CALCULATION")
        logger.info("="*80)
        
        total_start = time.time()

        # Step 1: Parse contract from Pinecone
        logger.info("\n📄 Step 1: Extracting contract data from Pinecone...")
        t0 = time.time()
        contract_data = self.contract_parser.parse_contract(
            path=contract_path,
            user_id=user_id,
            contract_id=contract_id
        )
        logger.info(f"⏱️  Step 1 took: {time.time() - t0:.2f}s")
        
        # Step 2: Read royalty statement
        logger.info("\n💵 Step 2: Reading royalty statement...")
        t0 = time.time()
        song_totals = self.read_royalty_statement(
            statement_path, 
            title_column, 
            payable_column
        )
        logger.info(f"⏱️  Step 2 took: {time.time() - t0:.2f}s")
        
        # Step 3: Calculate payments
        logger.info("\n🔍 DEBUG CHECK — Contract vs Statement")
        logger.info("Contract works:")
        for w in contract_data.works:
            logger.info(f"   → {w.title}")
        logger.info("\nStatement songs:")
        for s in list(song_totals.keys())[:10]:
            logger.info(f"   → {s}")
        
        t0 = time.time()
        payments = self._calculate_payments_from_data(
            contract_data,
            song_totals
        )
        logger.info(f"⏱️  Step 3 took: {time.time() - t0:.2f}s")
        
        logger.info(f"\n✅ Total calculation process took: {time.time() - total_start:.2f}s")

        return payments
    
    def calculate_payments_from_contract_ids(
        self,
        contract_ids: List[str],
        user_id: str,
        statement_path: str,
        title_column: Optional[str] = None,
        payable_column: Optional[str] = None
    ) -> List[RoyaltyPayment]:
        """
        Parse multiple contracts from Pinecone in PARALLEL, merge their data, and calculate payments.
        
        Args:
            contract_ids: List of contract IDs to query from Pinecone
            user_id: User ID for Pinecone namespace
            statement_path: Path to the royalty statement file
            title_column: Optional column name for song titles
            payable_column: Optional column name for payable amounts
            
        Returns:
            List of RoyaltyPayment objects with combined results
        """
        logger.info("\n" + "="*80)
        logger.info(f"MULTI-CONTRACT ROYALTY CALCULATION ({len(contract_ids)} contracts)")
        logger.info("="*80)
        
        # Step 1: Parse all contracts in PARALLEL
        logger.info(f"\n📄 Step 1: Parsing {len(contract_ids)} contracts from Pinecone (Parallel)...")
        all_contracts_data = []
        
        # Use ThreadPoolExecutor for parallel processing
        # We limit max_workers to avoid hitting API rate limits too hard
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            future_to_cid = {
                executor.submit(self.contract_parser.parse_contract, path=None, user_id=user_id, contract_id=cid): cid 
                for cid in contract_ids
            }
            
            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_cid), 1):
                cid = future_to_cid[future]
                try:
                    # logger.info(f"   ...Finished processing a contract...")
                    data = future.result()
                    all_contracts_data.append(data)
                    logger.info(f"   ✓ Contract parsed successfully ({len(data.parties)} parties, {len(data.works)} works)")
                except Exception as e:
                    logger.error(f"   ⚠️  Failed to parse contract {cid}: {e}")
        
        if not all_contracts_data:
            raise ValueError("❌ No valid contracts could be parsed. Please check your files.")
        
        # Step 2: Merge contracts
        merged_data = self.merge_contracts(all_contracts_data)
        
        # Step 3: Read royalty statement
        logger.info("\n💵 Step 2: Reading royalty statement...")
        song_totals = self.read_royalty_statement(
            statement_path,
            title_column,
            payable_column
        )
        
        # Step 4: Calculate payments
        payments = self._calculate_payments_from_data(
            merged_data,
            song_totals
        )
        
        return payments

    def calculate_payments_from_contracts(
        self,
        contract_paths: List[str],
        statement_path: str,
        title_column: Optional[str] = None,
        payable_column: Optional[str] = None
    ) -> List[RoyaltyPayment]:
        """
        Parse multiple contracts, merge their data, and calculate payments.
        
        This is useful when multiple contracts cover the same works with
        different contributors (e.g., producer agreement, artist agreement).
        
        Args:
            contract_paths: List of paths to contract files
            statement_path: Path to the royalty statement file
            title_column: Optional column name for song titles
            payable_column: Optional column name for payable amounts
            
        Returns:
            List of RoyaltyPayment objects with combined results
        """
        logger.info("\n" + "="*80)
        logger.info(f"MULTI-CONTRACT ROYALTY CALCULATION ({len(contract_paths)} contracts)")
        logger.info("="*80)
        
        # Step 1: Parse all contracts
        logger.info(f"\n📄 Step 1: Parsing {len(contract_paths)} contracts...")
        all_contracts_data = []
        
        for idx, path in enumerate(contract_paths, 1):
            try:
                logger.info(f"\n   Contract {idx}/{len(contract_paths)}: {Path(path).name}")
                data = self.contract_parser.parse_contract(path)
                all_contracts_data.append(data)
                
                # Quick preview
                logger.info(f"      → {len(data.parties)} parties, {len(data.works)} works, {len(data.royalty_shares)} shares")
                
            except Exception as e:
                logger.error(f"      ⚠️  Failed to parse: {e}")
                continue
        
        if not all_contracts_data:
            raise ValueError("❌ No valid contracts could be parsed. Please check your files.")
        
        # Step 2: Merge contracts
        merged_data = self.merge_contracts(all_contracts_data)
        
        # Step 3: Read royalty statement
        logger.info("\n💵 Step 2: Reading royalty statement...")
        song_totals = self.read_royalty_statement(
            statement_path,
            title_column,
            payable_column
        )
        
        # Step 4: Calculate payments
        payments = self._calculate_payments_from_data(
            merged_data,
            song_totals
        )
        
        return payments
    
    def _calculate_payments_from_data(
        self,
        contract_data: ContractData,
        song_totals: Dict[str, float]
    ) -> List[RoyaltyPayment]:
        """
        Internal method to calculate payments from parsed contract data.
        
        Args:
            contract_data: Parsed ContractData object
            song_totals: Dictionary of song titles to amounts
            
        Returns:
            List of RoyaltyPayment objects
        """
        logger.info("\n💰 Step 3: Calculating payments...\n")
        
        # Validate inputs
        if not contract_data.works:
            raise ValueError("❌ No works found in contract data")
        
        if not contract_data.royalty_shares:
            raise ValueError("❌ No royalty shares found in contract data")
        
        if not song_totals:
            raise ValueError("❌ No songs found in royalty statement")
        
        # Filter for streaming royalties only
        streaming_shares = [
            share for share in contract_data.royalty_shares
            if 'streaming' in share.royalty_type.lower()
        ]
        
        if not streaming_shares:
            logger.warning("   ⚠️  No streaming royalty shares found in contract")
            return []
        
        logger.info(f"   Found {len(streaming_shares)} streaming royalty shares")
        logger.info(f"   Found {len(contract_data.works)} works to match")
        
        # Calculate payments for each work
        payments = []
        matched_count = 0
        unmatched_works = []
        
        for work in contract_data.works:
            # Find matching song in statement
            matching_song, total_royalty = find_matching_song(
                work.title,
                song_totals
            )
            
            if matching_song:
                matched_count += 1
                logger.info(f"\n   ✓ '{work.title}'")
                logger.info(f"      Matched to: '{matching_song}'")
                logger.info(f"      Total royalties: ${total_royalty:,.2f}")
                
                # Calculate payment for each party with streaming shares
                for share in streaming_shares:
                    amount_to_pay = total_royalty * (share.percentage / 100.0)
                    
                    # Find party details
                    party = next(
                        (p for p in contract_data.parties
                         if normalize_name(p.name) == normalize_name(share.party_name)),
                        None
                    )
                    role = party.role if party else "unknown"
                    
                    payment = RoyaltyPayment(
                        song_title=work.title,
                        party_name=share.party_name,
                        role=role,
                        royalty_type=share.royalty_type,
                        percentage=share.percentage,
                        total_royalty=total_royalty,
                        amount_to_pay=amount_to_pay,
                        terms=share.terms
                    )
                    payments.append(payment)
                    
                    logger.info(f"         → {share.party_name} ({role}): {share.percentage}% = ${amount_to_pay:,.2f}")
            else:
                unmatched_works.append(work.title)
        
        # Summary
        logger.info(f"\n   📊 Matching Summary:")
        logger.info(f"      ✓ Matched: {matched_count}/{len(contract_data.works)} works")
        
        if unmatched_works:
            logger.warning(f"      ⚠️  Unmatched works:")
            for title in unmatched_works:
                logger.warning(f"         - {title}")
            logger.info(f"\n      💡 Tip: Check for typos or verify these songs are in the statement")
        
        logger.info(f"\n   ✅ Calculated {len(payments)} total payments")
        
        return payments
    
    # ========================================================================
    # OUTPUT METHODS
    # ========================================================================
    
    def save_payments_to_excel(
        self,
        payments: List[RoyaltyPayment],
        output_path: str
    ):
        """
        Save calculated payments to an Excel file with formatting.
        
        Args:
            payments: List of RoyaltyPayment objects
            output_path: Path to save the Excel file
        """
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Royalty Payments"
        
        # Headers
        headers = [
            "Song Title", "Payee Name", "Role", "Royalty Type",
            "Share %", "Total Royalty", "Amount to Pay", "Terms"
        ]
        sheet.append(headers)
        
        # Format headers
        header_font = openpyxl.styles.Font(bold=True, size=11)
        header_fill = openpyxl.styles.PatternFill(
            start_color="366092",
            end_color="366092",
            fill_type="solid"
        )
        header_font_white = openpyxl.styles.Font(bold=True, size=11, color="FFFFFF")
        
        for cell in sheet[1]:
            cell.font = header_font_white
            cell.fill = header_fill
        
        # Add data
        for payment in payments:
            sheet.append([
                payment.song_title,
                payment.party_name,
                payment.role,
                payment.royalty_type,
                f"{payment.percentage}%",
                payment.total_royalty,
                payment.amount_to_pay,
                payment.terms or ""
            ])
        
        # Format currency columns
        for row in sheet.iter_rows(min_row=2, min_col=6, max_col=7):
            for cell in row:
                cell.number_format = '$#,##0.00'
        
        # Auto-adjust column widths
        for column in sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            
            adjusted_width = min(max_length + 3, 50)
            sheet.column_dimensions[column_letter].width = adjusted_width
        
        # Add summary row
        if payments:
            summary_row = sheet.max_row + 2
            sheet[f'A{summary_row}'] = "TOTAL"
            sheet[f'A{summary_row}'].font = openpyxl.styles.Font(bold=True)
            
            total_formula = f'=SUM(G2:G{sheet.max_row - 1})'
            sheet[f'G{summary_row}'] = total_formula
            sheet[f'G{summary_row}'].font = openpyxl.styles.Font(bold=True)
            sheet[f'G{summary_row}'].number_format = '$#,##0.00'
        
        workbook.save(output_path)
        logger.info(f"\n💾 Payment breakdown saved to {output_path}")
    
    def save_payments_to_json(
        self,
        payments: List[RoyaltyPayment],
        output_path: str
    ):
        """
        Save calculated payments to a JSON file.
        
        Args:
            payments: List of RoyaltyPayment objects
            output_path: Path to save the JSON file
        """
        data = {
            'payments': [asdict(payment) for payment in payments],
            'summary': {
                'total_payments': len(payments),
                'total_amount': sum(p.amount_to_pay for p in payments),
                'unique_payees': len(set(p.party_name for p in payments)),
                'unique_songs': len(set(p.song_title for p in payments))
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Payment data saved to {output_path}")
    
    def print_payment_summary(self, payments: List[RoyaltyPayment]):
        """Print a formatted summary of payments to console"""
        
        logger.info("\n" + "="*80)
        logger.info("PAYMENT SUMMARY")
        logger.info("="*80)
        
        if not payments:
            logger.warning("\n⚠️  No payments calculated")
            return
        
        # Group by payee
        payee_totals = {}
        for payment in payments:
            if payment.party_name not in payee_totals:
                payee_totals[payment.party_name] = {
                    'role': payment.role,
                    'total': 0.0,
                    'details': []
                }
            payee_totals[payment.party_name]['total'] += payment.amount_to_pay
            payee_totals[payment.party_name]['details'].append(payment)
        
        # Print summary for each payee
        for payee, data in sorted(payee_totals.items()):
            logger.info(f"\n👤 {payee} ({data['role'].title()})")
            logger.info(f"   Total Payment: ${data['total']:,.2f}")
            logger.info(f"   Breakdown:")
            
            for detail in data['details']:
                logger.info(
                    f"      • {detail.song_title}: "
                    f"{detail.percentage}% of ${detail.total_royalty:,.2f} "
                    f"= ${detail.amount_to_pay:,.2f}"
                )
        
        # Grand total
        grand_total = sum(p.amount_to_pay for p in payments)
        logger.info(f"\n{'='*80}")
        logger.info(f"GRAND TOTAL: ${grand_total:,.2f}")
        logger.info(f"Total Payments: {len(payments)}")
        logger.info(f"Unique Payees: {len(payee_totals)}")
        logger.info(f"Unique Songs: {len(set(p.song_title for p in payments))}")
        logger.info(f"{'='*80}\n")
