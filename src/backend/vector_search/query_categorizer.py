"""
Query Categorization Module

This module provides intelligent query categorization for contract retrieval.
It uses an LLM to determine which contract sections are relevant to a user's query.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Optional
import json

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-5-mini")

# Define available contract section categories
SECTION_CATEGORIES = {
    "ROYALTY_CALCULATIONS": "Royalty calculations, revenue splits, compensation, payment percentages",
    "PUBLISHING_RIGHTS": "Publishing rights, songwriter credits, composition ownership",
    "PERFORMANCE_RIGHTS": "Performance rights, live performance, production services",
    "COPYRIGHT": "Copyright ownership, intellectual property rights",
    "TERMINATION": "Contract termination, term duration, end conditions",
    "MASTER_RIGHTS": "Master recording rights, sound recording ownership",
    "OWNERSHIP_RIGHTS": "General ownership, synchronization rights, licensing",
    "ACCOUNTING_AND_CREDIT": "Accounting procedures, credit attribution, promotion",
    "OTHER": "General contract terms, miscellaneous clauses"
}

# Keyword mappings for fast categorization (no LLM needed)
CATEGORY_KEYWORDS = {
    "ROYALTY_CALCULATIONS": [
        "royalty", "royalties", "revenue", "split", "splits", "percentage", 
        "compensation", "payment", "streaming", "net revenue", "gross revenue",
        "share", "shares", "paid", "pay", "earnings", "master points",
        "royalty participation", "revenue participation", "royalty share",
        "net master revenue", "net master royalty", "gross master royalty",
        "royalty percentage", "sound recording royalty splits", "producer royalties",
        "master royalties", "points"
    ],
    "PUBLISHING_RIGHTS": [
        "publishing", "songwriter", "composition", "mechanical", "sync rights",
        "publishing rights", "writer", "composer"
    ],
    "PERFORMANCE_RIGHTS": [
        "performance", "live", "production services", "concert", "show",
        "performing rights", "PRO", "ASCAP", "BMI", "SOCAN"
    ],
    "COPYRIGHT": [
        "copyright", "intellectual property", "IP rights", "copyrighted",
        "ownership of copyright"
    ],
    "TERMINATION": [
        "termination", "terminate", "term", "duration", "end date", 
        "expiration", "expire", "cancel"
    ],
    "MASTER_RIGHTS": [
        "master", "master rights", "master recording", "sound recording",
        "masters", "recording rights"
    ],
    "OWNERSHIP_RIGHTS": [
        "ownership", "synchronization", "sync", "licensing", "license",
        "owns", "owned by", "rights granted"
    ],
    "ACCOUNTING_AND_CREDIT": [
        "accounting", "credit", "promotion", "audit", "statement",
        "credited as", "promotional"
    ],
}


def categorize_query_fast(query: str) -> Dict:
    """
    Fast keyword-based query categorization (no LLM calls).
    
    This is much faster than LLM-based categorization and works well
    for common, predictable queries like those used in OneClick.
    
    Args:
        query (str): The user's question about the contract
        
    Returns:
        Dict: {
            "categories": List[str] - Relevant section categories,
            "is_general": bool - Whether query requires searching all sections,
            "confidence": str - "high", "medium", or "low"
        }
    """
    query_lower = query.lower()
    matched_categories = []
    
    # Check each category's keywords
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            matched_categories.append(category)
    
    # Determine if it's a general query
    general_indicators = [
        "summarize", "summary", "overview", "all", "everything",
        "entire", "whole contract", "obligations", "terms and conditions"
    ]
    is_general = any(indicator in query_lower for indicator in general_indicators)
    
    # If no categories matched, treat as general or OTHER
    if not matched_categories:
        if is_general:
            return {
                "categories": list(SECTION_CATEGORIES.keys()),
                "is_general": True,
                "confidence": "high",
                "reasoning": "General query detected via keywords"
            }
        else:
            matched_categories = ["OTHER"]
    
    # Determine confidence based on number of matches
    if len(matched_categories) == 1:
        confidence = "high"
    elif len(matched_categories) <= 3:
        confidence = "medium"
    else:
        confidence = "low"
    
    return {
        "categories": matched_categories,
        "is_general": is_general,
        "confidence": confidence,
        "reasoning": f"Keyword-based categorization: {len(matched_categories)} categories matched"
    }


def categorize_query(query: str, openai_client: Optional[OpenAI] = None, use_llm: bool = True) -> Dict:
    """
    Categorize a user query to determine which contract sections are relevant.
    
    Args:
        query (str): The user's question about the contract
        openai_client (OpenAI, optional): OpenAI client instance
        use_llm (bool): If False, uses fast keyword matching instead of LLM (default: True)
        
    Returns:
        Dict: {
            "categories": List[str] - Relevant section categories,
            "is_general": bool - Whether query requires searching all sections,
            "confidence": str - "high", "medium", or "low"
        }
    """
    # Use fast keyword-based categorization if requested
    if not use_llm:
        return categorize_query_fast(query)
    
    if openai_client is None:
        openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    
    # Create category descriptions for the prompt
    category_list = "\n".join([
        f"- {cat}: {desc}" 
        for cat, desc in SECTION_CATEGORIES.items()
    ])
    
    system_prompt = """You are an expert at analyzing legal contract queries and determining which sections of a contract are relevant.
Your task is to categorize user queries to help retrieve the most relevant contract sections."""

    user_prompt = f"""Given the following user query about a music contract, determine which contract section categories are most relevant.

User Query: "{query}"

Available Categories:
{category_list}

Instructions:
1. Identify which categories are relevant to answering this query
2. Determine if this is a GENERAL query that requires searching across all sections, or a SPECIFIC query targeting particular sections
3. Assess your confidence level

Return your response in this exact JSON format:
{{
    "categories": ["CATEGORY1", "CATEGORY2"],
    "is_general": false,
    "confidence": "high",
    "reasoning": "Brief explanation of your categorization"
}}

Rules:
- If the query is very specific (e.g., "What are the royalty splits?"), return 1-2 categories and is_general: false
- If the query is broad (e.g., "Summarize this contract", "What are my obligations?"), return is_general: true
- Confidence should be "high" if you're certain, "medium" if somewhat uncertain, "low" if very uncertain
- Always return valid JSON"""

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        # Clean potential leading/trailing non-json characters if not in code block
        if not result_text.startswith("{"):
             start_idx = result_text.find("{")
             if start_idx != -1:
                 result_text = result_text[start_idx:]
        if not result_text.endswith("}"):
             end_idx = result_text.rfind("}")
             if end_idx != -1:
                 result_text = result_text[:end_idx+1]

        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            # Try to fix common JSON errors (like single quotes)
            try:
                import ast
                result = ast.literal_eval(result_text)
            except:
                 raise ValueError(f"Could not parse JSON: {result_text}")
        
        # Validate categories
        valid_categories = [cat for cat in result.get("categories", []) if cat in SECTION_CATEGORIES]
        
        return {
            "categories": valid_categories if valid_categories else ["OTHER"],
            "is_general": result.get("is_general", False),
            "confidence": result.get("confidence", "medium"),
            "reasoning": result.get("reasoning", "")
        }
        
    except Exception as e:
        print(f"Error in query categorization: {e}")
        # Fallback: treat as general query
        return {
            "categories": list(SECTION_CATEGORIES.keys()),
            "is_general": True,
            "confidence": "low",
            "reasoning": f"Error occurred: {str(e)}"
        }


def build_metadata_filter(
    categories: List[str],
    is_general: bool,
    document_name: Optional[str] = None,
    user_id: Optional[str] = None,
    contract_id: Optional[str] = None
) -> Optional[Dict]:
    """
    Build a Pinecone metadata filter based on categorization results.
    
    Args:
        categories: List of relevant section categories
        is_general: Whether this is a general query
        document_name: Optional document name filter
        user_id: Optional user ID filter
        contract_id: Optional contract ID filter
        
    Returns:
        Dict: Pinecone filter object, or None if no filtering needed
    """
    filters = {}
    
    # Add user/contract/document filters if provided
    if user_id:
        filters["user_id"] = {"$eq": user_id}
    if contract_id:
        filters["contract_id"] = {"$eq": contract_id}
    if document_name:
        filters["document_name"] = {"$eq": document_name}
    
    # Add category filter only if it's a specific query
    if not is_general and categories:
        if len(categories) == 1:
            filters["section_category"] = {"$eq": categories[0]}
        else:
            # Multiple categories - use $in operator
            filters["section_category"] = {"$in": categories}
    
    return filters if filters else None


# Example usage and testing
if __name__ == "__main__":
    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    
    # Test queries
    test_queries = [
        "What are the royalty splits in this contract?",
        "Who owns the master rights?",
        "Summarize this entire contract for me",
        "What happens if I want to terminate the agreement?",
        "What are my publishing and performance rights?",
        "Tell me everything about payments and credits"
    ]
    
    print("="*80)
    print("QUERY CATEGORIZATION TESTS")
    print("="*80)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)
        
        result = categorize_query(query, openai_client)
        
        print(f"Categories: {', '.join(result['categories'])}")
        print(f"Is General: {result['is_general']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Reasoning: {result['reasoning']}")
        
        # Show what filter would be built
        filter_obj = build_metadata_filter(
            categories=result['categories'],
            is_general=result['is_general'],
            document_name="test_contract.pdf"
        )
        print(f"Filter: {filter_obj}")
