"""
Contract Chatbot Module (OneClick Smart Retrieval)
RAG-based Q&A system for contract documents with intelligent query categorization.

Features:
- Smart query categorization for targeted retrieval
- Section-aware semantic search
- Similarity threshold enforcement (≥0.50)
- LLM-powered answer generation with grounding
- Support for project-level and contract-level queries
- Session-based conversation history with memory retention
"""

import os
import uuid
import time
import json
import hashlib
import re
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from vector_search.contract_search import ContractSearch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)



@dataclass
class ChatMessage:
    """Represents a single chat message"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


class InMemoryChatMessageHistory:
    """
    In-memory storage for chat message history per session.
    Provides efficient retrieval and automatic cleanup of old sessions.
    """
    
    def __init__(self, max_messages_per_session: int = 20, session_ttl_seconds: int = 3600):
        """
        Initialize the chat message history.
        
        Args:
            max_messages_per_session: Maximum number of messages to keep per session
            session_ttl_seconds: Time-to-live for inactive sessions (default: 1 hour)
        """
        self._sessions: Dict[str, List[ChatMessage]] = defaultdict(list)
        self._session_timestamps: Dict[str, float] = {}
        self.max_messages = max_messages_per_session
        self.session_ttl = session_ttl_seconds
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None) -> None:
        """
        Add a message to the session history.
        
        Args:
            session_id: Unique session identifier
            role: Message role ("user" or "assistant")
            content: Message content
            metadata: Optional metadata (sources, confidence, etc.)
        """
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self._sessions[session_id].append(message)
        self._session_timestamps[session_id] = time.time()
        
        # Trim to max messages (keep most recent)
        if len(self._sessions[session_id]) > self.max_messages:
            self._sessions[session_id] = self._sessions[session_id][-self.max_messages:]
    
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        """
        Get messages for a session.
        
        Args:
            session_id: Unique session identifier
            limit: Maximum number of messages to return (most recent)
            
        Returns:
            List of ChatMessage objects
        """
        messages = self._sessions.get(session_id, [])
        if limit and len(messages) > limit:
            return messages[-limit:]
        return messages
    
    def get_messages_for_llm(self, session_id: str, limit: int = 10) -> List[Dict]:
        """
        Get messages formatted for OpenAI API.
        
        Args:
            session_id: Unique session identifier
            limit: Maximum number of message pairs to include
            
        Returns:
            List of message dicts with 'role' and 'content'
        """
        messages = self.get_messages(session_id, limit=limit * 2)  # user + assistant pairs
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    def clear_session(self, session_id: str) -> None:
        """Clear all messages for a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
        if session_id in self._session_timestamps:
            del self._session_timestamps[session_id]
        if session_id in self._pending_suggestions:
            del self._pending_suggestions[session_id]
    
    def set_pending_suggestion(self, session_id: str, suggestion: str, context_hash: str) -> None:
        """
        Store a pending suggestion for the session.
        
        Args:
            session_id: Unique session identifier
            suggestion: The suggested follow-up question/action
            context_hash: Hash of contracts_discussed to detect context changes
        """
        if not hasattr(self, '_pending_suggestions'):
            self._pending_suggestions: Dict[str, Dict] = {}
        self._pending_suggestions[session_id] = {
            'suggestion': suggestion,
            'context_hash': context_hash,
            'timestamp': time.time()
        }
    
    def get_pending_suggestion(self, session_id: str) -> Optional[Dict]:
        """
        Get the pending suggestion for a session.
        
        Returns:
            Dict with 'suggestion' and 'context_hash', or None if no pending suggestion
        """
        if not hasattr(self, '_pending_suggestions'):
            self._pending_suggestions: Dict[str, Dict] = {}
            return None
        return self._pending_suggestions.get(session_id)
    
    def clear_pending_suggestion(self, session_id: str) -> None:
        """Clear the pending suggestion for a session."""
        if hasattr(self, '_pending_suggestions') and session_id in self._pending_suggestions:
            del self._pending_suggestions[session_id]


# Global conversation memory instance
_conversation_memory = InMemoryChatMessageHistory(max_messages_per_session=20, session_ttl_seconds=3600)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None
)

# Configuration
DEFAULT_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-5-mini")  # Updated to stable model
MIN_SIMILARITY_THRESHOLD = 0.30
DEFAULT_TOP_K = 8
MAX_CONTEXT_LENGTH = 8000  # Characters to send to LLM


class ContractChatbot:
    """RAG-based chatbot for contract Q&A with session-based memory"""
    
    # Conversational patterns that don't require document search
    CONVERSATIONAL_PATTERNS = [
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon", 
        "good evening", "howdy", "what's up", "whats up", "sup",
        "how are you", "how's it going", "hows it going",
        "thanks", "thank you", "thx", "appreciated",
        "bye", "goodbye", "see you", "later", "cya",
        "help", "what can you do", "who are you", "what are you",
        "nice", "great", "awesome", "cool", "ok", "okay",
    ]
    
    # Affirmative patterns that indicate user wants to proceed with a pending suggestion
    AFFIRMATIVE_PATTERNS = [
        "yes", "yeah", "yep", "yup", "sure", "please", "please do", 
        "go ahead", "do it", "yes please", "sounds good", "that would be great",
        "absolutely", "definitely", "of course", "ok", "okay", "alright",
        "let's do it", "show me", "tell me", "i'd like that", "yes do that"
    ]
    
    def __init__(self, llm_model: str = DEFAULT_LLM_MODEL):
        """
        Initialize the contract chatbot
        
        Args:
            llm_model: LLM model to use for answer generation
        """
        self.llm_model = llm_model
        self.search_engine = ContractSearch()
        self.memory = _conversation_memory  # Use global memory instance
    
    def _is_affirmative_response(self, query: str) -> bool:
        """
        Check if the query is an affirmative response to a pending suggestion.
        
        Args:
            query: User's message
            
        Returns:
            True if it's an affirmative response
        """
        query_lower = query.lower().strip()
        query_clean = ''.join(c for c in query_lower if c.isalnum() or c.isspace()).strip()
        
        for pattern in self.AFFIRMATIVE_PATTERNS:
            if query_clean == pattern or query_clean.startswith(pattern + " ") or query_clean.endswith(" " + pattern):
                return True
        
        return False
    
    def _compute_context_hash(self, context: Optional[Dict]) -> str:
        """
        Compute a hash of the contracts discussed to detect context changes.
        
        Args:
            context: Conversation context
            
        Returns:
            Hash string of contract IDs
        """
        if not context:
            return ""
        
        contracts_discussed = context.get('contracts_discussed', [])
        contract_ids = sorted([c.get('id', '') for c in contracts_discussed])
        return hashlib.md5(json.dumps(contract_ids).encode()).hexdigest()
    
    def _extract_structured_data(self, answer: str, query: str) -> Dict:
        """
        Extract structured data from the LLM answer using pattern matching.
        This replaces client-side regex extraction with server-side extraction.
        
        Args:
            answer: The LLM-generated answer
            query: The original query (to determine what data to extract)
            
        Returns:
            Dict with extracted data fields
        """
        extracted = {}
        query_lower = query.lower()
        
        # Determine royalty type from query
        royalty_type = self._detect_royalty_type(query_lower)
        
        # Extract royalty splits if discussing royalties/splits/percentages
        if any(kw in query_lower for kw in ['royalty', 'split', 'percentage', 'share', '%']):
            # Pattern: "Name: 35%" or "Name - 35%" or "Name (35%)"
            split_patterns = [
                r'([A-Za-z][A-Za-z\s\.\'-]+?):\s*(\d+(?:\.\d+)?)\s*%',
                r'([A-Za-z][A-Za-z\s\.\'-]+?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*%',
                r'([A-Za-z][A-Za-z\s\.\'-]+?)\s*\((\d+(?:\.\d+)?)\s*%\)',
            ]
            splits = []
            for pattern in split_patterns:
                matches = re.findall(pattern, answer)
                for match in matches:
                    party = match[0].strip()
                    percentage = float(match[1])
                    # Avoid duplicates
                    if not any(s['party'].lower() == party.lower() for s in splits):
                        splits.append({'party': party, 'percentage': percentage})
            if splits:
                # Store under the specific royalty type
                extracted['royalty_splits'] = {royalty_type: splits}
        
        # Extract parties/signatories
        if any(kw in query_lower for kw in ['parties', 'who signed', 'signatories', 'between']):
            # Pattern: quoted names or bullet points
            party_matches = re.findall(r'"([^"]+)"|•\s*([^\n•]+)|^\s*[-*]\s*([^\n]+)', answer, re.MULTILINE)
            parties = []
            for match in party_matches:
                for group in match:
                    if group and group.strip():
                        parties.append(group.strip())
            if parties:
                extracted['parties'] = parties
        
        # Extract payment terms
        if any(kw in query_lower for kw in ['payment', 'terms', 'net', 'days']):
            if len(answer) > 0 and len(answer) < 500:
                extracted['payment_terms'] = answer.strip()
        
        # Extract term length
        if any(kw in query_lower for kw in ['term', 'duration', 'length', 'period', 'years', 'months']):
            term_match = re.search(r'(\d+)\s*(year|month|day)s?', answer, re.IGNORECASE)
            if term_match:
                extracted['term_length'] = f"{term_match.group(1)} {term_match.group(2)}s"
        
        # Extract advances
        if any(kw in query_lower for kw in ['advance', 'upfront', 'signing bonus']):
            advance_match = re.search(r'\$[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD)?', answer)
            if advance_match:
                extracted['advances'] = advance_match.group(0)
        
        return extracted
    
    def _detect_royalty_type(self, query_lower: str) -> str:
        """
        Detect the type of royalty being discussed from the query.
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            Royalty type string (streaming, publishing, etc.)
        """
        royalty_types = {
            'streaming': ['streaming', 'stream', 'spotify', 'apple music', 'dsp'],
            'publishing': ['publishing', 'publish', 'songwriter', 'composition'],
            'mechanical': ['mechanical', 'mechanicals'],
            'sync': ['sync', 'synchronization', 'synch'],
            'master': ['master', 'masters', 'recording'],
            'performance': ['performance', 'performing', 'pro', 'ascap', 'bmi'],
        }
        
        for royalty_type, keywords in royalty_types.items():
            if any(kw in query_lower for kw in keywords):
                return royalty_type
        
        return 'general'  # Default for unspecified royalty types
    
    def _extract_suggestion_from_answer(self, answer: str) -> Tuple[str, Optional[str]]:
        """
        Extract and separate a follow-up suggestion from the answer.
        
        Args:
            answer: The full LLM answer
            
        Returns:
            Tuple of (clean_answer, suggestion) where suggestion may be None
        """
        # Pattern to match "Would you like me to..." suggestions at the end
        suggestion_pattern = r'\n*(?:Would you like (?:me to|to)?|Shall I|Should I|Do you want me to)\s*([^?]+\??)\s*$'
        match = re.search(suggestion_pattern, answer, re.IGNORECASE)
        
        if match:
            suggestion = match.group(0).strip()
            clean_answer = answer[:match.start()].strip()
            return clean_answer, suggestion
        
        return answer, None
    
    def _is_conversational_query(self, query: str) -> bool:
        """
        Check if the query is a conversational message (greeting, thanks, etc.)
        that doesn't require document search.
        
        Args:
            query: User's message
            
        Returns:
            True if it's a conversational query
        """
        query_lower = query.lower().strip()
        # Remove punctuation for matching
        query_clean = ''.join(c for c in query_lower if c.isalnum() or c.isspace())
        
        # Check for exact or partial matches
        for pattern in self.CONVERSATIONAL_PATTERNS:
            if query_clean == pattern or query_clean.startswith(pattern + " ") or query_clean.endswith(" " + pattern):
                return True
            # Also check if the entire query is just the pattern with punctuation
            if pattern in query_clean and len(query_clean) < len(pattern) + 10:
                return True
        
        return False
    
    def _handle_conversational_query(self, query: str, session_id: Optional[str] = None) -> Dict:
        """
        Handle conversational queries with a friendly response.
        
        Args:
            query: User's conversational message
            session_id: Session ID for memory
            
        Returns:
            Dict with response
        """
        query_lower = query.lower().strip()
        
        # Track if we should show quick action buttons
        show_quick_actions = False
        
        # Generate appropriate response based on query type
        if any(g in query_lower for g in ["hello", "hi", "hey", "greetings", "howdy", "good morning", "good afternoon", "good evening"]):
            response = "Hello! I'm Zoe, your contract analysis assistant. How can I help you today?"
            show_quick_actions = True
        elif any(g in query_lower for g in ["how are you", "how's it going", "hows it going", "what's up", "whats up"]):
            response = "I'm doing well, thanks! What would you like to know about your contracts?"
            show_quick_actions = True
        elif any(g in query_lower for g in ["thanks", "thank you", "thx", "appreciated"]):
            response = "You're welcome! Is there anything else you'd like to know?"
        elif any(g in query_lower for g in ["bye", "goodbye", "see you", "later", "cya"]):
            response = "Goodbye! Feel free to come back anytime."
        elif any(g in query_lower for g in ["help", "what can you do", "who are you", "what are you"]):
            response = "I'm Zoe, your AI contract assistant. Here are some things I can help with:"
            show_quick_actions = True
        else:
            response = "I'm here to help! What would you like to know about your contracts?"
            show_quick_actions = True
        
        # Store in memory
        self._add_to_memory(session_id, "user", query)
        self._add_to_memory(session_id, "assistant", response)
        
        return {
            "query": query,
            "answer": response,
            "confidence": "conversational",
            "sources": [],
            "search_results_count": 0,
            "session_id": session_id,
            "show_quick_actions": show_quick_actions
        }
    
    def _classify_query(self, query: str) -> str:
        """
        Use LLM to classify whether a query is about artist info or contracts.
        
        Args:
            query: User's question
            
        Returns:
            "artist" if query is about artist profile/info
            "contract" if query is about contracts/agreements
        """
        system_prompt = """You are a query classifier. Determine if the user's question is about:

1. ARTIST - Questions about the artist's profile, bio, social media, streaming links, genres, contact info, EPK, press kit, etc.
   Examples: "What's the artist's bio?", "What are their social media links?", "What genre do they make?", "What's their Spotify?"

2. CONTRACT - Questions about contracts, agreements, royalties, payment terms, splits, advances, legal terms, parties, clauses, etc.
   Examples: "What are the royalty splits?", "Who are the parties?", "What's the advance amount?", "When does the contract end?"

Respond with ONLY one word: either "artist" or "contract"."""

        try:
            response = openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_completion_tokens=10
            )
            
            classification = response.choices[0].message.content.strip().lower()
            
            # Ensure we get a valid response
            if "artist" in classification:
                return "artist"
            else:
                return "contract"
                
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Default to contract if classification fails
            return "contract"
    
    def _handle_artist_query(self, query: str, artist_data: Dict, session_id: Optional[str] = None) -> Dict:
        """
        Handle artist-related queries using artist data.
        
        Args:
            query: User's question about the artist
            artist_data: Artist information from database
            session_id: Session ID for memory
            
        Returns:
            Dict with response
        """
        # Store user message in memory
        self._add_to_memory(session_id, "user", query)
        
        # Format artist data as context for LLM
        artist_context = self._format_artist_context(artist_data)
        
        # Generate answer using LLM
        system_prompt = """You are a helpful assistant providing concise, direct information about an artist. 
Answer the user's question based on the artist profile information provided.

CRITICAL RULES:
1. Answer ONLY with the exact information from the provided artist data
2. Be direct and concise - do NOT add extra commentary, interpretation, or embellishment
3. Do NOT infer or add information not explicitly stated in the artist data
4. If asked for a specific piece of information (like bio), provide ONLY that information
5. Keep responses brief - typically 1-2 sentences unless the data itself is longer
6. Format links as plain URLs when mentioning them
7. If information is not available, simply say so

Examples:
- If asked "What's the artist's bio?" and bio is "House DJ" → Answer: "House DJ"
- If asked "What's their genre?" and genre is "Electronic" → Answer: "Electronic"
- If asked "What are their socials?" → List only the available social media links"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Based on the following artist profile, answer this question:

{query}

Artist Profile:
{artist_context}"""}
        ]
        
        try:
            response = openai_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_completion_tokens=500
            )
            
            answer = response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating artist answer: {e}")
            answer = f"I have information about {artist_data.get('name', 'the artist')}, but encountered an error. Please try again."
        
        # Store assistant response in memory
        self._add_to_memory(session_id, "assistant", answer)
        
        return {
            "query": query,
            "answer": answer,
            "confidence": "artist_data",
            "sources": [],
            "search_results_count": 0,
            "session_id": session_id,
            "show_quick_actions": False
        }
    
    def _format_artist_context(self, artist_data: Dict) -> str:
        """
        Format artist data as context string for LLM.
        
        Args:
            artist_data: Artist information dictionary
            
        Returns:
            Formatted string with artist details
        """
        parts = []
        
        if artist_data.get("name"):
            parts.append(f"Name: {artist_data['name']}")
        
        if artist_data.get("email"):
            parts.append(f"Email: {artist_data['email']}")
        
        if artist_data.get("bio"):
            parts.append(f"Bio: {artist_data['bio']}")
        
        if artist_data.get("genres"):
            genres = artist_data["genres"]
            if isinstance(genres, list):
                parts.append(f"Genres: {', '.join(genres)}")
            else:
                parts.append(f"Genres: {genres}")
        
        # Social media links
        socials = []
        if artist_data.get("social_instagram"):
            socials.append(f"Instagram: {artist_data['social_instagram']}")
        if artist_data.get("social_tiktok"):
            socials.append(f"TikTok: {artist_data['social_tiktok']}")
        if artist_data.get("social_youtube"):
            socials.append(f"YouTube: {artist_data['social_youtube']}")
        if socials:
            parts.append("Social Media:\n  " + "\n  ".join(socials))
        
        # DSP links
        dsps = []
        if artist_data.get("dsp_spotify"):
            dsps.append(f"Spotify: {artist_data['dsp_spotify']}")
        if artist_data.get("dsp_apple_music"):
            dsps.append(f"Apple Music: {artist_data['dsp_apple_music']}")
        if artist_data.get("dsp_soundcloud"):
            dsps.append(f"SoundCloud: {artist_data['dsp_soundcloud']}")
        if dsps:
            parts.append("Streaming Platforms:\n  " + "\n  ".join(dsps))
        
        # Additional links
        additional = []
        if artist_data.get("additional_epk"):
            additional.append(f"EPK: {artist_data['additional_epk']}")
        if artist_data.get("additional_press_kit"):
            additional.append(f"Press Kit: {artist_data['additional_press_kit']}")
        if artist_data.get("additional_linktree"):
            additional.append(f"Linktree: {artist_data['additional_linktree']}")
        if additional:
            parts.append("Additional Links:\n  " + "\n  ".join(additional))
        
        # Custom links
        if artist_data.get("custom_links"):
            custom = artist_data["custom_links"]
            if isinstance(custom, list) and custom:
                custom_parts = [f"{link.get('label', 'Link')}: {link.get('url', '')}" for link in custom if link.get('url')]
                if custom_parts:
                    parts.append("Other Links:\n  " + "\n  ".join(custom_parts))
        
        return "\n\n".join(parts) if parts else "No artist information available."
    
    def _get_conversation_context(self, session_id: Optional[str], max_turns: int = 5) -> List[Dict]:
        """
        Get conversation history for context.
        
        Args:
            session_id: Session identifier
            max_turns: Maximum conversation turns to include
            
        Returns:
            List of message dicts for LLM
        """
        if not session_id:
            return []
        return self.memory.get_messages_for_llm(session_id, limit=max_turns)
    
    def _add_to_memory(self, session_id: Optional[str], role: str, content: str, metadata: Dict = None) -> None:
        """Add a message to session memory."""
        if session_id:
            self.memory.add_message(session_id, role, content, metadata)
    
    def _should_use_context(self, query: str, context: Optional[Dict]) -> Tuple[bool, str]:
        """
        Helper method to check if query should be answered from context.
        Centralizes the context check logic and logging for both smart_ask and multi_contract_ask.
        
        Args:
            query: User's question
            context: Conversation context from frontend
            
        Returns:
            Tuple of (should_use_context: bool, reason: str)
        """
        can_answer, reason = self._can_answer_from_context(query, context)
        logger.info(f"[Context] Can answer from context: {can_answer}, reason: {reason}")
        
        # Log why we're falling through to document search
        if not can_answer and reason.startswith("missing_"):
            logger.info(f"[Context] Missing data in context ({reason}) - will search documents")
        
        return can_answer, reason
    
    def _can_answer_from_context(self, query: str, context: Optional[Dict]) -> Tuple[bool, str]:
        """
        Determine if query can be answered from conversation context
        without searching documents. Returns tuple for better debugging.
        
        Args:
            query: User's question
            context: Conversation context from frontend
            
        Returns:
            Tuple of (can_answer: bool, reason: str)
        """
        if not context:
            logger.info("[Context] No context provided, cannot answer from context")
            return False, "no_context"
        
        contracts_discussed = context.get('contracts_discussed', [])
        logger.info(f"[Context] Checking if can answer from context. Contracts discussed: {len(contracts_discussed)}")
        
        query_lower = query.lower()
        
        # Summary/recap questions about the conversation - always use context
        summary_keywords = ['summarize', 'summary', 'what did we discuss', 'recap', 
                           'what have we talked about', 'what do we know', 'overview']
        if any(kw in query_lower for kw in summary_keywords):
            if len(contracts_discussed) > 0:
                return True, "summary_request"
            return False, "no_contracts_for_summary"
        
        # "Earlier you mentioned" or "you said" type follow-ups
        followup_keywords = ['you said', 'you mentioned', 'earlier', 'previously', 
                            'before you said', 'we discussed', 'you told me']
        if any(kw in query_lower for kw in followup_keywords):
            if len(contracts_discussed) > 0:
                return True, "followup_reference"
            return False, "no_contracts_for_followup"
        
        # Check if asking about royalties - need to verify we have the SPECIFIC type
        royalty_keywords = ['royalty', 'split', 'percentage', 'share']
        if any(kw in query_lower for kw in royalty_keywords):
            royalty_type = self._detect_royalty_type(query_lower)
            logger.info(f"[Context] Query is about royalty type: {royalty_type}")
            
            # Check if we have this specific royalty type in context
            has_required_type = False
            for contract in contracts_discussed:
                data = contract.get('data_extracted', {})
                royalty_splits = data.get('royalty_splits', {})
                
                if isinstance(royalty_splits, dict):
                    # New structure: {streaming: [...], publishing: [...]}
                    if royalty_splits.get(royalty_type):
                        has_required_type = True
                        logger.info(f"[Context] Found {royalty_type} splits in {contract.get('name')}")
                        break
                    # Also check 'general' as fallback
                    if royalty_type == 'general' and any(royalty_splits.values()):
                        has_required_type = True
                        break
                elif isinstance(royalty_splits, list) and len(royalty_splits) > 0:
                    # Legacy structure: [{party, percentage}, ...]
                    # Can only use if asking for 'general' or unspecified royalties
                    if royalty_type == 'general':
                        has_required_type = True
                        logger.info(f"[Context] Found legacy splits in {contract.get('name')}")
                        break
            
            if has_required_type:
                # Check for comparison or re-ask
                comparison_keywords = ['compare', 'difference', 'versus', 'vs', 'between', 
                                       'which one', 'both', 'all', 'total', 'across']
                reask_indicators = ['again', 'remind me', 'what was', 'what were', 'tell me again']
                
                if any(kw in query_lower for kw in comparison_keywords):
                    # Need multiple contracts with this data type
                    contracts_with_type = sum(1 for c in contracts_discussed 
                                              if self._contract_has_royalty_type(c, royalty_type))
                    if contracts_with_type >= 2:
                        return True, f"comparison_{royalty_type}"
                    return False, f"need_more_contracts_for_{royalty_type}_comparison"
                
                if any(ind in query_lower for ind in reask_indicators):
                    return True, f"reask_{royalty_type}"
            
            logger.info(f"[Context] Missing {royalty_type} splits - need document lookup")
            return False, f"missing_{royalty_type}_splits"
        
        # Check for other data types
        other_data_checks = [
            (['parties', 'who signed', 'signatories'], 'parties'),
            (['payment', 'net', 'days'], 'payment_terms'),
            (['term', 'duration', 'length', 'period'], 'term_length'),
            (['advance', 'upfront', 'signing bonus'], 'advances'),
        ]
        
        for keywords, field_name in other_data_checks:
            if any(kw in query_lower for kw in keywords):
                for contract in contracts_discussed:
                    data = contract.get('data_extracted', {})
                    if data.get(field_name):
                        # Check for re-ask
                        reask_indicators = ['again', 'remind me', 'what was', 'what were']
                        if any(ind in query_lower for ind in reask_indicators):
                            return True, f"reask_{field_name}"
                return False, f"missing_{field_name}"
        
        return False, "no_matching_pattern"
    
    def _contract_has_royalty_type(self, contract: Dict, royalty_type: str) -> bool:
        """Check if a contract has extracted data for a specific royalty type."""
        data = contract.get('data_extracted', {})
        royalty_splits = data.get('royalty_splits', {})
        
        if isinstance(royalty_splits, dict):
            return bool(royalty_splits.get(royalty_type))
        elif isinstance(royalty_splits, list) and royalty_type == 'general':
            return len(royalty_splits) > 0
        return False
    
    def _answer_from_context(self, query: str, context: Dict, session_id: Optional[str] = None) -> Dict:
        """
        Answer question using structured context and conversation history.
        Used when the answer can be derived from previously extracted data.
        May suggest context-only follow-ups based on available data.
        
        Args:
            query: User's question
            context: Conversation context containing discussed contracts and extracted data
            session_id: Session ID for memory
            
        Returns:
            Dict with answer and metadata
        """
        logger.info(f"[Context] _answer_from_context called with query: {query}")
        
        # Build context summary for LLM
        context_summary = self._format_context_for_llm(context)
        logger.info(f"[Context] Formatted context summary:\n{context_summary}")
        
        # Get conversation history
        history = self._get_conversation_context(session_id, max_turns=10)
        logger.info(f"[Context] Conversation history: {len(history)} messages")
        
        # Analyze what data is available for suggestions
        available_data = self._get_available_data_types(context)
        suggestion_guidance = self._build_suggestion_guidance(available_data)
        
        system_prompt = f"""You are answering a follow-up question using information from the conversation context.

The user is asking about information that was previously discussed in this conversation. 
Use the structured context and conversation history to provide a comprehensive answer.

CRITICAL: You can compare, summarize, or analyze the information provided without needing to search documents again.

CONTEXT-ONLY SUGGESTIONS:
After answering, you MAY suggest ONE follow-up question, but ONLY if:
1. The answer can be derived ENTIRELY from the provided context data
2. It's genuinely relevant to what the user just asked
3. It adds value (comparison, deeper analysis, related insight)

Format suggestions as: "Would you like me to [specific action]?"

{suggestion_guidance}

If no context-only follow-up makes sense, don't suggest anything.

RULES:
1. Base your answer ONLY on the provided context and conversation history
2. If comparing contracts, clearly distinguish between them
3. Be specific about which contract each piece of information came from
4. If information is incomplete, acknowledge what you don't have
5. Format comparisons in a clear, readable way (use bullet points or tables)
6. Only suggest follow-ups that can be answered from existing context data"""

        messages = [{"role": "system", "content": system_prompt}]
        # Add conversation history first, then the current query with context
        messages.extend(history)
        messages.append({"role": "user", "content": f"Context:\n{context_summary}\n\nQuestion: {query}"})
        
        try:
            response = openai_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_completion_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Handle None response
            if answer is None:
                answer = "I couldn't generate a response. Please try rephrasing your question."
            
        except Exception as e:
            logger.error(f"Error generating context-based answer: {e}")
            answer = "I encountered an error while processing your question. Please try again."
        
        # Extract any suggestion from the answer for tracking
        clean_answer, suggestion = self._extract_suggestion_from_answer(answer)
        
        # Store pending suggestion if one was made
        pending_suggestion = None
        if suggestion and session_id:
            context_hash = self._compute_context_hash(context)
            # Extract the action from "Would you like me to X?"
            action_match = re.search(r'(?:Would you like (?:me to|to)?|Shall I|Should I)\s*(.+?)\??$', suggestion, re.IGNORECASE)
            if action_match:
                pending_suggestion = action_match.group(1).strip()
                self.memory.set_pending_suggestion(session_id, pending_suggestion, context_hash)
                logger.info(f"[Context] Stored pending suggestion: {pending_suggestion}")
        
        # Store in memory
        self._add_to_memory(session_id, "user", query)
        self._add_to_memory(session_id, "assistant", answer)
        
        # Extract structured data from the answer
        extracted_data = self._extract_structured_data(answer, query)
        
        return {
            "query": query,
            "answer": answer,
            "confidence": "context_based",
            "sources": [],
            "search_results_count": 0,
            "answered_from": "conversation_context",
            "session_id": session_id,
            "extracted_data": extracted_data if extracted_data else None,
            "pending_suggestion": pending_suggestion
        }
    
    def _get_available_data_types(self, context: Dict) -> Dict:
        """
        Analyze what types of data are available in the context.
        
        Args:
            context: Conversation context
            
        Returns:
            Dict with available data types per contract
        """
        available = {
            'contracts': [],
            'has_royalty_splits': False,
            'has_parties': False,
            'has_payment_terms': False,
            'has_term_length': False,
            'has_advances': False,
            'multiple_contracts': False,
            'multiple_data_types': False
        }
        
        contracts_discussed = context.get('contracts_discussed', [])
        data_types_found = set()
        
        for contract in contracts_discussed:
            data = contract.get('data_extracted', {})
            if not data:
                continue
                
            contract_info = {'name': contract.get('name', 'Unknown'), 'data_types': []}
            
            if data.get('royalty_splits'):
                available['has_royalty_splits'] = True
                data_types_found.add('royalty_splits')
                contract_info['data_types'].append('royalty_splits')
            if data.get('parties'):
                available['has_parties'] = True
                data_types_found.add('parties')
                contract_info['data_types'].append('parties')
            if data.get('payment_terms'):
                available['has_payment_terms'] = True
                data_types_found.add('payment_terms')
                contract_info['data_types'].append('payment_terms')
            if data.get('term_length'):
                available['has_term_length'] = True
                data_types_found.add('term_length')
                contract_info['data_types'].append('term_length')
            if data.get('advances'):
                available['has_advances'] = True
                data_types_found.add('advances')
                contract_info['data_types'].append('advances')
            
            if contract_info['data_types']:
                available['contracts'].append(contract_info)
        
        available['multiple_contracts'] = len(available['contracts']) >= 2
        available['multiple_data_types'] = len(data_types_found) >= 2
        
        return available
    
    def _build_suggestion_guidance(self, available_data: Dict) -> str:
        """
        Build guidance for the LLM about what suggestions are valid.
        
        Args:
            available_data: Dict from _get_available_data_types
            
        Returns:
            String guidance for the system prompt
        """
        valid_suggestions = []
        invalid_note = ""
        
        if available_data['multiple_contracts']:
            contract_names = [c['name'] for c in available_data['contracts']]
            valid_suggestions.append(f"Compare data between contracts: {', '.join(contract_names)}")
        
        if available_data['multiple_data_types']:
            if available_data['has_royalty_splits'] and available_data['has_payment_terms']:
                valid_suggestions.append("Compare royalty splits with payment terms")
            if available_data['has_royalty_splits']:
                valid_suggestions.append("Calculate total percentages across categories")
                valid_suggestions.append("Show all parties and their splits")
        
        if not valid_suggestions:
            invalid_note = "\nNOTE: Limited context data available - avoid suggesting follow-ups."
        
        if valid_suggestions:
            return f"""VALID context-only suggestions (data IS available):
{chr(10).join('✅ ' + s for s in valid_suggestions)}

INVALID suggestions (would require document lookup - DO NOT suggest these):
❌ Anything about data not listed in the context above
❌ "Check the payment terms" (unless payment_terms is in context)
❌ "Find the advance amount" (unless advances is in context)
❌ "See who else is in the contract" (unless parties is in context){invalid_note}"""
        else:
            return invalid_note
    
    def _format_context_for_llm(self, context: Dict) -> str:
        """
        Format structured conversation context as readable text for LLM.
        
        Args:
            context: Conversation context dictionary
            
        Returns:
            Formatted string with context details
        """
        parts = []
        
        if context.get('artist'):
            parts.append(f"Current Artist: {context['artist'].get('name', 'Unknown')}")
        
        if context.get('project'):
            parts.append(f"Current Project: {context['project'].get('name', 'Unknown')}")
        
        if context.get('contracts_discussed'):
            parts.append("\nContracts Discussed:")
            for contract in context['contracts_discussed']:
                parts.append(f"\n📄 {contract.get('name', 'Unknown Contract')}")
                data_extracted = contract.get('data_extracted', {})
                if data_extracted:
                    for key, value in data_extracted.items():
                        if value:  # Only include non-empty values
                            if isinstance(value, list):
                                parts.append(f"  • {key.replace('_', ' ').title()}:")
                                for item in value:
                                    if isinstance(item, dict):
                                        item_str = ", ".join(f"{k}: {v}" for k, v in item.items())
                                        parts.append(f"    - {item_str}")
                                    else:
                                        parts.append(f"    - {item}")
                            else:
                                parts.append(f"  • {key.replace('_', ' ').title()}: {value}")
        
        if context.get('context_switches'):
            recent_switches = context['context_switches'][-3:]  # Last 3 switches
            if recent_switches:
                parts.append("\nRecent Context Changes:")
                for switch in recent_switches:
                    from_val = switch.get('from_value') or switch.get('from', 'None')
                    parts.append(f"  • {switch['type'].title()}: {from_val} → {switch['to']}")
        
        return "\n".join(parts) if parts else "No context available."

    def _check_similarity_threshold(self, search_results: Dict) -> bool:
        """
        Check if highest similarity score meets threshold
        
        Args:
            search_results: Results from ContractSearch
            
        Returns:
            True if threshold is met, False otherwise
        """
        if not search_results["matches"]:
            return False
        
        highest_score = search_results["matches"][0]["score"]
        return highest_score >= MIN_SIMILARITY_THRESHOLD
    
    def _create_system_prompt(self) -> str:
        """
        Create system prompt for LLM with grounding instructions
        
        Returns:
            System prompt string
        """
        return """You are a specialized contract analysis assistant for the music industry. Your role is to answer questions about music contracts accurately and precisely.

CRITICAL RULES:
1. ONLY answer based on the provided contract contracts - do not use external knowledge
2. If the answer is not explicitly stated in the contracts, respond with: "I don't know based on the available documents."
3. Always cite the source (contract file name) when providing information
4. Be precise with numbers, percentages, dates, and legal terms
5. If multiple contracts contain relevant information, clearly distinguish between them
6. Do not make assumptions or inferences beyond what is explicitly stated
7. If asked about something not in the contracts, acknowledge the limitation

Your answers should be:
- Accurate and grounded in the provided text
- Clear and concise
- Properly cited with sources
- Professional and helpful"""
    
    def _format_context(self, search_results: Dict) -> str:
        """
        Format search results as context for LLM.
        Uses the same format as oneclick_retrieval.py for consistency.
        
        Args:
            search_results: Results from ContractSearch
            
        Returns:
            Formatted context string
        """
        if not search_results["matches"]:
            return "No relevant contract contracts found."
        
        # Format exactly like oneclick_retrieval.py
        context_parts = []
        for match in search_results["matches"]:
            section = match.get('section_heading', 'N/A')
            text = match.get('text', '')
            context_parts.append(f"[Section: {section}]\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str, search_results: Dict, session_id: Optional[str] = None) -> Dict:
        """
        Generate answer using LLM with conversation history for context.
        
        Args:
            query: User's question
            context: Formatted context from search results
            search_results: Original search results
            session_id: Session ID for conversation history
            
        Returns:
            Dict with answer and metadata
        """
        # Check similarity threshold
        if not self._check_similarity_threshold(search_results):
            return {
                "answer": "I don't know based on the available documents.",
                "confidence": "low",
                "reason": "No sufficiently relevant information found (similarity threshold not met)",
                "highest_score": search_results["matches"][0]["score"] if search_results["matches"] else 0.0,
                "threshold": MIN_SIMILARITY_THRESHOLD,
                "sources": []
            }
        
        # System prompt with strict focus on the question asked - NO follow-up suggestions
        # Suggestions are only made in _answer_from_context when we have context data
        system_prompt = """You are a legal contract analyst specializing in music industry agreements. 
Your task is to answer questions about contract documents based on the provided context.

CRITICAL RULES:
1. Answer ONLY the specific question asked - nothing more, nothing less
2. Do NOT automatically add comparisons, summaries, or extra information unless explicitly requested
3. Do NOT reference or compare to previous topics in the conversation unless the user asks for it
4. Be precise and only include information that is explicitly stated in the provided context
5. If you cannot answer based on the documents, say so clearly
6. Do NOT suggest follow-up questions - the system handles this separately based on extracted context

CONVERSATION AWARENESS:
- Use conversation history ONLY to understand pronouns and references (e.g., "this contract" = the one just discussed)
- Do NOT proactively bring up or compare to previous topics

Your answers should be:
- Accurate and grounded in the provided text
- Clear and concise
- Properly cited with sources
- Professional and helpful"""

        # Build messages list with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history for context
        conversation_history = self._get_conversation_context(session_id, max_turns=5)
        if conversation_history:
            # Add a context separator
            messages.append({
                "role": "system", 
                "content": "Previous conversation for context (use ONLY for understanding references, NOT for adding unsolicited comparisons):"
            })
            messages.extend(conversation_history)
        
        # Add current query with contract context
        user_prompt = f"""Based on the following contract documents, answer this question:

{query}

Contract documents:
{context}

Remember: Answer ONLY what was asked. Do not suggest follow-up questions."""
        
        messages.append({"role": "user", "content": user_prompt})
        
        # Call LLM
        logger.info(f"\nGenerating answer using {self.llm_model}...")
        logger.info(f"Conversation history: {len(conversation_history)} messages included")
        
        try:
            response = openai_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_completion_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Server-side extraction of structured data
            extracted_data = self._extract_structured_data(answer, query) if answer else None
            
            # Extract sources from search results
            sources = [
                {
                    "contract_file": match["contract_file"],
                    "score": match["score"],
                    "project_name": match["project_name"],
                    "section_heading": match.get("section_heading", ""),
                    "section_category": match.get("section_category", "")
                }
                for match in search_results["matches"]
            ]
            
            return {
                "answer": answer,
                "confidence": "high",
                "highest_score": search_results["matches"][0]["score"],
                "threshold": MIN_SIMILARITY_THRESHOLD,
                "sources": sources,
                "model": self.llm_model,
                "extracted_data": extracted_data if extracted_data else None
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "confidence": "error",
                "error": str(e),
                "sources": []
            }
    
    def smart_ask(self,
                  query: str,
                  user_id: str,
                  project_id: Optional[str] = None,
                  contract_id: Optional[str] = None,
                  top_k: Optional[int] = None,
                  session_id: Optional[str] = None,
                  artist_data: Optional[Dict] = None,
                  context: Optional[Dict] = None) -> Dict:
        """
        Ask a question using smart retrieval with automatic query categorization.
        
        This method automatically:
        1. Checks if question can be answered from conversation context
        2. Checks for conversational queries (greetings, thanks, etc.)
        3. Checks for artist-related queries and answers from artist data
        4. Categorizes the query to determine relevant contract sections
        5. Adjusts top_k based on query type (general vs specific)
        6. Filters search to relevant sections for better precision
        7. Generates a grounded answer based on retrieved context
        8. Maintains conversation history for follow-up questions
        
        Args:
            query: User's question
            user_id: UUID of the user
            project_id: UUID of the project (optional)
            contract_id: UUID of specific contract (optional)
            top_k: Number of search results (auto-adjusted if None)
            session_id: Session ID for conversation memory
            artist_data: Optional artist information for artist-related queries
            context: Optional conversation context for context-based answering
            
        Returns:
            Dict with answer, sources, categorization, and metadata
        """
        logger.info("\n" + "=" * 80)
        logger.info("SMART CONTRACT CHATBOT (with Query Categorization)")
        logger.info("="  * 80)
        logger.info(f"Question: {query}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Artist data available: {artist_data is not None}")
        logger.info(f"Context available: {context is not None}")
        if project_id:
            logger.info(f"Project ID: {project_id}")
        if contract_id:
            logger.info(f"Contract ID: {contract_id}")
        logger.info("-" * 80)
        
        # Check for affirmative response to pending suggestion first
        if session_id and self._is_affirmative_response(query):
            pending = self.memory.get_pending_suggestion(session_id)
            if pending:
                logger.info("[Affirmative] Detected affirmative response with pending suggestion")
                current_hash = self._compute_context_hash(context)
                
                # Validate context hasn't changed
                if pending.get('context_hash') != current_hash:
                    logger.info("[Affirmative] Context hash mismatch - context was cleared")
                    self.memory.clear_pending_suggestion(session_id)
                    return {
                        "query": query,
                        "answer": "It looks like the conversation context was reset. Please refresh the page to start a new session, or ask your question again.",
                        "confidence": "context_cleared",
                        "sources": [],
                        "search_results_count": 0,
                        "session_id": session_id,
                        "context_cleared": True
                    }
                
                # Rewrite query to the pending suggestion and answer from context
                suggestion = pending.get('suggestion', '')
                logger.info(f"[Affirmative] Rewriting query to: {suggestion}")
                self.memory.clear_pending_suggestion(session_id)
                
                # Route to context-based answering with the suggestion as query
                return self._answer_from_context(suggestion, context, session_id)
        
        # Check if we can answer from conversation context first
        can_answer, reason = self._should_use_context(query, context)
        
        if can_answer:
            logger.info("Detected context-based query - answering from conversation context")
            return self._answer_from_context(query, context, session_id)
        
        # Check for conversational queries (greetings, thanks, etc.)
        if self._is_conversational_query(query):
            logger.info("Detected conversational query - handling without document search")
            return self._handle_conversational_query(query, session_id)
        
        # Step 1: Classify the query - is it about artist or contracts?
        query_type = self._classify_query(query)
        logger.info(f"Query classified as: {query_type}")
        
        # Step 2: Route based on classification
        if query_type == "artist":
            # Artist query - use artist data
            if artist_data:
                logger.info("Handling as artist query - using artist data")
                return self._handle_artist_query(query, artist_data, session_id)
            else:
                # No artist data available
                self._add_to_memory(session_id, "user", query)
                response = "I don't have artist information available. Please select an artist from the sidebar."
                self._add_to_memory(session_id, "assistant", response)
                return {
                    "query": query,
                    "answer": response,
                    "confidence": "needs_artist",
                    "sources": [],
                    "search_results_count": 0,
                    "session_id": session_id,
                    "show_quick_actions": True
                }
        
        # Contract query - need a project to search
        if not project_id:
            self._add_to_memory(session_id, "user", query)
            response = "To answer questions about contracts, I need you to select a project first. Please choose a project from the sidebar."
            self._add_to_memory(session_id, "assistant", response)
            return {
                "query": query,
                "answer": response,
                "confidence": "needs_project",
                "sources": [],
                "search_results_count": 0,
                "session_id": session_id,
                "show_quick_actions": True
            }
        
        # Store user message in memory
        self._add_to_memory(session_id, "user", query)
        
        # Step 3: Query Pinecone for contract data
        search_results = self.search_engine.smart_search(
            query=query,
            user_id=user_id,
            project_id=project_id,
            contract_id=contract_id,
            top_k=top_k
        )
        
        # Step 4: If no results, return low confidence
        if not search_results["matches"]:
            no_result_answer = "I don't know based on the available documents."
            self._add_to_memory(session_id, "assistant", no_result_answer)
            return {
                "query": query,
                "answer": no_result_answer,
                "confidence": "low",
                "reason": "No relevant documents found",
                "sources": [],
                "search_results_count": 0,
                "categorization": search_results.get("categorization", {}),
                "session_id": session_id
            }
        
        # Step 3: Format context
        context = self._format_context(search_results)
        
        # Step 4: Generate answer with conversation history
        result = self._generate_answer(query, context, search_results, session_id)
        
        # Step 5: Store assistant response in memory
        self._add_to_memory(session_id, "assistant", result["answer"], {
            "confidence": result["confidence"],
            "sources": result.get("sources", [])
        })
        
        # Step 6: Add query and search metadata
        result["query"] = query
        result["search_results_count"] = search_results["total_results"]
        result["filter"] = search_results["filter"]
        result["categorization"] = search_results.get("categorization", {})
        result["session_id"] = session_id
        
        logger.info("\n" + "=" * 80)
        logger.info("SMART ANSWER GENERATED")
        logger.info("=" * 80)
        logger.info(f"Confidence: {result['confidence']}")
        if result.get('highest_score'):
            logger.info(f"Highest Similarity Score: {result['highest_score']}")
        logger.info(f"Sources Used: {len(result['sources'])}")
        if result.get('categorization'):
            logger.info(f"Query Categories: {result['categorization'].get('categories', [])}")
        logger.info("=" * 80)
        
        return result
    
    def ask_without_project(self,
                           query: str,
                           user_id: str,
                           session_id: Optional[str] = None,
                           artist_data: Optional[Dict] = None,
                           context: Optional[Dict] = None) -> Dict:
        """
        Handle queries when no project is selected.
        Can answer artist-related queries or prompt to select a project for contract queries.
        
        Args:
            query: User's question
            user_id: UUID of the user
            session_id: Session ID for conversation memory
            artist_data: Optional artist information for artist-related queries
            context: Optional conversation context for context-based answering
            
        Returns:
            Dict with answer and metadata
        """
        logger.info("\n" + "=" * 80)
        logger.info("CHATBOT (No Project Selected)")
        logger.info("=" * 80)
        logger.info(f"Question: {query}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Artist data available: {artist_data is not None}")
        logger.info(f"Context available: {context is not None}")
        logger.info("-" * 80)
        
        # Check if we can answer from conversation context first
        if self._can_answer_from_context(query, context):
            logger.info("Detected context-based query - answering from conversation context")
            return self._answer_from_context(query, context, session_id)
        
        # Check for conversational queries
        if self._is_conversational_query(query):
            logger.info("Detected conversational query - handling without document search")
            return self._handle_conversational_query(query, session_id)
        
        # Classify the query - is it about artist or contracts?
        query_type = self._classify_query(query)
        logger.info(f"Query classified as: {query_type}")
        
        if query_type == "artist":
            # Artist query - use artist data if available
            if artist_data:
                logger.info("Handling as artist query - using artist data")
                return self._handle_artist_query(query, artist_data, session_id)
            else:
                # No artist data available
                self._add_to_memory(session_id, "user", query)
                response = "I don't have artist information available. Please select an artist from the sidebar."
                self._add_to_memory(session_id, "assistant", response)
                return {
                    "query": query,
                    "answer": response,
                    "confidence": "needs_artist",
                    "sources": [],
                    "search_results_count": 0,
                    "session_id": session_id,
                    "show_quick_actions": True
                }
        else:
            # Contract query but no project selected - prompt to select project
            self._add_to_memory(session_id, "user", query)
            response = "To answer questions about contracts, I need you to select a project first. Please choose a project from the sidebar."
            self._add_to_memory(session_id, "assistant", response)
            return {
                "query": query,
                "answer": response,
                "confidence": "needs_project",
                "sources": [],
                "search_results_count": 0,
                "session_id": session_id,
                "show_quick_actions": True
            }
    
    def ask_project(self,
                   query: str,
                   user_id: str,
                   project_id: str,
                   top_k: int = DEFAULT_TOP_K,
                   session_id: Optional[str] = None,
                   artist_data: Optional[Dict] = None,
                   context: Optional[Dict] = None) -> Dict:
        """
        Ask a question about a specific project's contracts using smart retrieval.
        
        Args:
            query: User's question
            user_id: UUID of the user
            project_id: UUID of the project
            top_k: Number of search results to retrieve
            session_id: Session ID for conversation memory
            artist_data: Optional artist information for artist-related queries
            context: Optional conversation context for context-based answering
            
        Returns:
            Dict with answer and metadata
        """
        return self.smart_ask(
            query=query,
            user_id=user_id,
            project_id=project_id,
            top_k=top_k,
            session_id=session_id,
            artist_data=artist_data,
            context=context
        )
    
    def ask_multiple_contracts(self,
                              query: str,
                              user_id: str,
                              project_id: str,
                              contract_ids: List[str],
                              top_k: int = DEFAULT_TOP_K,
                              session_id: Optional[str] = None,
                              artist_data: Optional[Dict] = None,
                              context: Optional[Dict] = None) -> Dict:
        """
        Ask a question about multiple specific contracts using smart retrieval.
        Searches across all selected contracts similar to project-wide search.
        
        Args:
            query: User's question
            user_id: UUID of the user
            project_id: UUID of the project
            contract_ids: List of contract UUIDs to search
            top_k: Number of search results to retrieve
            session_id: Session ID for conversation memory
            artist_data: Optional artist information for artist-related queries
            context: Optional conversation context for context-based answering
            
        Returns:
            Dict with answer and metadata
        """
        logger.info("\n" + "=" * 80)
        logger.info("MULTI-CONTRACT CHATBOT")
        logger.info("=" * 80)
        logger.info(f"Question: {query}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Project ID: {project_id}")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Contract IDs: {contract_ids}")
        logger.info(f"Number of contracts: {len(contract_ids)}")
        logger.info(f"Artist data available: {artist_data is not None}")
        logger.info(f"Context available: {context is not None}")
        logger.info("-" * 80)
        
        # Check if we can answer from conversation context first
        can_answer, reason = self._should_use_context(query, context)
        
        if can_answer:
            logger.info("Detected context-based query - answering from conversation context")
            return self._answer_from_context(query, context, session_id)
        
        # Check for conversational queries (greetings, thanks, etc.)
        if self._is_conversational_query(query):
            logger.info("Detected conversational query - handling without document search")
            return self._handle_conversational_query(query, session_id)
        
        # Step 1: Classify the query - is it about artist or contracts?
        query_type = self._classify_query(query)
        logger.info(f"Query classified as: {query_type}")
        
        # Step 2: Route based on classification
        if query_type == "artist":
            # Artist query - use artist data
            if artist_data:
                logger.info("Handling as artist query - using artist data")
                return self._handle_artist_query(query, artist_data, session_id)
            else:
                # No artist data available
                self._add_to_memory(session_id, "user", query)
                response = "I don't have artist information available. Please select an artist from the sidebar."
                self._add_to_memory(session_id, "assistant", response)
                return {
                    "query": query,
                    "answer": response,
                    "confidence": "needs_artist",
                    "sources": [],
                    "search_results_count": 0,
                    "session_id": session_id,
                    "show_quick_actions": True
                }
        
        # Store user message in memory
        self._add_to_memory(session_id, "user", query)
        
        # Step 3: Query Pinecone for contract data
        search_results = self.search_engine.search_multiple_contracts(
            query=query,
            user_id=user_id,
            project_id=project_id,
            contract_ids=contract_ids,
            top_k=top_k
        )
        
        # Step 4: If no results, return low confidence
        if not search_results["matches"]:
            no_result_answer = "I don't know based on the available documents."
            self._add_to_memory(session_id, "assistant", no_result_answer)
            return {
                "query": query,
                "answer": no_result_answer,
                "confidence": "low",
                "reason": "No relevant documents found",
                "sources": [],
                "search_results_count": 0,
                "session_id": session_id
            }
        
        # Format context
        context = self._format_context(search_results)
        
        # Generate answer with conversation history
        result = self._generate_answer(query, context, search_results, session_id)
        
        # Store assistant response in memory
        self._add_to_memory(session_id, "assistant", result["answer"], {
            "confidence": result["confidence"],
            "sources": result.get("sources", [])
        })
        
        # Add query and search metadata
        result["query"] = query
        result["search_results_count"] = search_results["total_results"]
        result["filter"] = search_results["filter"]
        result["session_id"] = session_id
        
        logger.info("\n" + "=" * 80)
        logger.info("MULTI-CONTRACT ANSWER GENERATED")
        logger.info("=" * 80)
        logger.info(f"Confidence: {result['confidence']}")
        if result.get('highest_score'):
            logger.info(f"Highest Similarity Score: {result['highest_score']}")
        logger.info(f"Sources Used: {len(result['sources'])}")
        logger.info(f"Contracts Searched: {len(contract_ids)}")
        logger.info("=" * 80)
        
        return result
    
    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self.memory.clear_session(session_id)
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of message dictionaries
        """
        messages = self.memory.get_messages(session_id)
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in messages
        ]


# Example usage
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = ContractChatbot()
    
    # Create a session for conversation memory demo
    import uuid
    session_id = str(uuid.uuid4())
    logger.info(f"\nSession ID: {session_id}")
    
    # Example 1: Ask about a project
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 1: Project-level question")
    logger.info("=" * 80)
    
    result1 = chatbot.ask_project(
        query="What are the royalty percentage splits in this project?",
        user_id="test-user-123",
        project_id="test-project-456",
        session_id=session_id
    )
    
    logger.info(f"\nQUESTION: {result1['query']}")
    logger.info(f"\nANSWER:\n{result1['answer']}")
    logger.info(f"\nCONFIDENCE: {result1['confidence']}")
    logger.info("\nSOURCES:")
    for source in result1.get("sources", []):
        logger.info(f"  - {source['contract_file']} (Score: {source['score']})")
    
    # Example 2: Follow-up question (uses conversation memory)
    logger.info("\n\n" + "=" * 80)
    logger.info("EXAMPLE 2: Follow-up question (with conversation memory)")
    logger.info("=" * 80)
    
    result2 = chatbot.ask_project(
        query="What about the payment terms?",
        user_id="test-user-123",
        project_id="test-project-456",
        session_id=session_id  # Same session for context
    )
    
    logger.info(f"\nQUESTION: {result2['query']}")
    logger.info(f"\nANSWER:\n{result2['answer']}")
    logger.info(f"\nCONFIDENCE: {result2['confidence']}")
    
    # Show conversation history
    logger.info("\n\n" + "=" * 80)
    logger.info("CONVERSATION HISTORY")
    logger.info("=" * 80)
    history = chatbot.get_session_history(session_id)
    for msg in history:
        logger.info(f"[{msg['role'].upper()}]: {msg['content'][:100]}...")
