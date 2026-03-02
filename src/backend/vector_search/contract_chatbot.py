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
- Pinned facts with confidence scoring (0.0-1.0)
- Assumptions ledger with scope-based invalidation
- Rolling conversation summaries
"""

import os
import uuid
import time
import json
import hashlib
import re
import logging
from typing import List, Dict, Optional, Tuple, Literal, Any
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI
from vector_search.contract_search import ContractSearch
from oneclick.helpers import normalize_name

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Confidence thresholds for graduated answering
class ConfidenceLevel(Enum):
    HIGH = "high"           # >= 0.85 - Answer directly
    MEDIUM = "medium"       # 0.60-0.84 - Answer with caveat
    LOW = "low"             # < 0.60 - Reverify or clarify
    CONTEXT = "context"     # From pinned facts
    CONVERSATIONAL = "conversational"  # Greetings, etc.


@dataclass
class RouteDecision:
    """Routing decision for the current turn."""
    route: Literal["artist", "contract", "disambiguate"] = "contract"
    answer_mode: Literal["context", "retrieve"] = "retrieve"
    confidence: float = 0.0
    reason: str = "default_contract_retrieval"
    retrieval_reason: str = "general_retrieval"


@dataclass
class Scope:
    """Defines the scope for facts and assumptions"""
    artist_id: Optional[str] = None
    project_id: Optional[str] = None
    contract_id: Optional[str] = None
    
    def matches(self, other: 'Scope') -> bool:
        """Check if this scope matches or is a subset of another scope"""
        if self.artist_id and other.artist_id and self.artist_id != other.artist_id:
            return False
        if self.project_id and other.project_id and self.project_id != other.project_id:
            return False
        if self.contract_id and other.contract_id and self.contract_id != other.contract_id:
            return False
        return True
    
    def is_global(self) -> bool:
        """Check if this is a global (unscoped) fact"""
        return not self.artist_id and not self.project_id and not self.contract_id
    
    def to_dict(self) -> Dict:
        return {"artist_id": self.artist_id, "project_id": self.project_id, "contract_id": self.contract_id}


@dataclass
class PinnedFact:
    """
    A verified fact extracted from documents or user input.
    Includes confidence scoring and scope tracking.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fact_type: str = ""  # royalty_split, payment_terms, parties, etc.
    description: str = ""  # Human-readable description
    value: Any = None  # The actual value (can be dict, list, string, number)
    confidence: float = 0.0  # 0.0 to 1.0
    source_type: Literal["document", "user_stated", "inferred"] = "document"
    source_reference: str = ""  # Document name, chunk ID, or "user stated"
    scope: Scope = field(default_factory=Scope)
    extracted_at: datetime = field(default_factory=datetime.now)
    last_verified: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "fact_type": self.fact_type,
            "description": self.description,
            "value": self.value,
            "confidence": self.confidence,
            "source_type": self.source_type,
            "source_reference": self.source_reference,
            "scope": self.scope.to_dict(),
            "extracted_at": self.extracted_at.isoformat(),
            "last_verified": self.last_verified.isoformat()
        }
    
    def get_confidence_level(self) -> ConfidenceLevel:
        """Get the confidence level based on thresholds"""
        if self.confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.60:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


@dataclass
class Assumption:
    """
    An unverified assumption made during conversation.
    Must be tracked separately from facts and invalidated on scope change.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    statement: str = ""  # The assumption being made
    context: str = ""  # What triggered this assumption
    scope: Scope = field(default_factory=Scope)
    introduced_at: datetime = field(default_factory=datetime.now)
    verified: bool = False
    invalidated: bool = False
    invalidated_at: Optional[datetime] = None
    invalidation_reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "context": self.context,
            "scope": self.scope.to_dict(),
            "introduced_at": self.introduced_at.isoformat(),
            "verified": self.verified,
            "invalidated": self.invalidated,
            "invalidated_at": self.invalidated_at.isoformat() if self.invalidated_at else None,
            "invalidation_reason": self.invalidation_reason
        }
    
    def invalidate(self, reason: str) -> None:
        """Mark this assumption as invalidated"""
        self.invalidated = True
        self.invalidated_at = datetime.now()
        self.invalidation_reason = reason


class FactLedger:
    """
    Manages pinned facts per session with confidence scoring and scope filtering.
    Facts are stored per session and can be filtered by current scope.
    """
    
    def __init__(self):
        self._facts: Dict[str, List[PinnedFact]] = defaultdict(list)
        self._session_timestamps: Dict[str, float] = {}
    
    def add_fact(self, session_id: str, fact: PinnedFact) -> None:
        """Add or update a pinned fact for a session"""
        existing_idx = None
        for idx, existing in enumerate(self._facts[session_id]):
            if (existing.fact_type == fact.fact_type and 
                existing.scope.matches(fact.scope) and
                existing.source_reference == fact.source_reference):
                existing_idx = idx
                break
        
        if existing_idx is not None:
            if fact.confidence >= self._facts[session_id][existing_idx].confidence:
                self._facts[session_id][existing_idx] = fact
        else:
            self._facts[session_id].append(fact)
        
        self._session_timestamps[session_id] = time.time()
    
    def get_facts(self, session_id: str, scope: Optional[Scope] = None, 
                  min_confidence: float = 0.0) -> List[PinnedFact]:
        """Get facts for a session, optionally filtered by scope and confidence"""
        facts = self._facts.get(session_id, [])
        
        if scope:
            facts = [f for f in facts if f.scope.matches(scope) or f.scope.is_global()]
        
        if min_confidence > 0:
            facts = [f for f in facts if f.confidence >= min_confidence]
        
        return facts
    
    def get_fact_by_type(self, session_id: str, fact_type: str, 
                         scope: Optional[Scope] = None) -> Optional[PinnedFact]:
        """Get the highest confidence fact of a given type"""
        facts = self.get_facts(session_id, scope)
        matching = [f for f in facts if f.fact_type == fact_type]
        if matching:
            return max(matching, key=lambda f: f.confidence)
        return None
    
    def clear_session(self, session_id: str) -> None:
        """Clear all facts for a session"""
        if session_id in self._facts:
            del self._facts[session_id]
        if session_id in self._session_timestamps:
            del self._session_timestamps[session_id]


class AssumptionLedger:
    """
    Manages assumptions per session with scope-based invalidation.
    Assumptions are tracked separately from verified facts.
    """
    
    def __init__(self):
        self._assumptions: Dict[str, List[Assumption]] = defaultdict(list)
    
    def add_assumption(self, session_id: str, assumption: Assumption) -> None:
        """Add an assumption for a session"""
        self._assumptions[session_id].append(assumption)
    
    def get_active_assumptions(self, session_id: str, scope: Optional[Scope] = None) -> List[Assumption]:
        """Get non-invalidated assumptions, optionally filtered by scope"""
        assumptions = [a for a in self._assumptions.get(session_id, []) if not a.invalidated]
        
        if scope:
            assumptions = [a for a in assumptions if a.scope.matches(scope) or a.scope.is_global()]
        
        return assumptions
    
    def invalidate_for_scope_change(self, session_id: str, old_scope: Scope, 
                                     new_scope: Scope) -> List[Assumption]:
        """
        Invalidate assumptions that are no longer valid due to scope change.
        Returns list of invalidated assumptions.
        """
        invalidated = []
        
        for assumption in self._assumptions.get(session_id, []):
            if assumption.invalidated:
                continue
            
            should_invalidate = False
            reason = ""
            
            if (assumption.scope.artist_id and 
                old_scope.artist_id == assumption.scope.artist_id and
                new_scope.artist_id != assumption.scope.artist_id):
                should_invalidate = True
                reason = f"Artist changed from {old_scope.artist_id} to {new_scope.artist_id}"
            
            if (assumption.scope.project_id and
                old_scope.project_id == assumption.scope.project_id and
                new_scope.project_id != assumption.scope.project_id):
                should_invalidate = True
                reason = f"Project changed from {old_scope.project_id} to {new_scope.project_id}"
            
            if should_invalidate:
                assumption.invalidate(reason)
                invalidated.append(assumption)
        
        return invalidated
    
    def verify_assumption(self, session_id: str, assumption_id: str) -> bool:
        """Mark an assumption as verified"""
        for assumption in self._assumptions.get(session_id, []):
            if assumption.id == assumption_id:
                assumption.verified = True
                return True
        return False
    
    def clear_session(self, session_id: str) -> None:
        """Clear all assumptions for a session"""
        if session_id in self._assumptions:
            del self._assumptions[session_id]


class ConversationSummary:
    """
    Manages rolling conversation summaries per session.
    Compresses older turns into a summary while keeping recent turns verbatim.
    """
    
    def __init__(self, recent_turns_limit: int = 6):
        self._summaries: Dict[str, str] = {}
        self._recent_turns_limit = recent_turns_limit
    
    def get_summary(self, session_id: str) -> str:
        """Get the rolling summary for a session"""
        return self._summaries.get(session_id, "")
    
    def update_summary(self, session_id: str, summary: str) -> None:
        """Update the rolling summary for a session"""
        self._summaries[session_id] = summary
    
    def clear_session(self, session_id: str) -> None:
        """Clear summary for a session"""
        if session_id in self._summaries:
            del self._summaries[session_id]


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

    def estimate_token_count(self, session_id: str) -> int:
        """
        Estimate the token count for a session's history.
        Uses a simple heuristic: ~1 token per 4 characters.

        Args:
            session_id: Unique session identifier

        Returns:
            Estimated token count
        """
        messages = self._sessions.get(session_id, [])
        total_chars = sum(len(msg.content) for msg in messages)
        return total_chars // 4

    def trim_to_recent(self, session_id: str, keep_recent: int = 6) -> List[ChatMessage]:
        """
        Remove older messages and return them, keeping only the most recent ones.

        Args:
            session_id: Unique session identifier
            keep_recent: Number of recent messages to keep

        Returns:
            List of removed (older) messages
        """
        messages = self._sessions.get(session_id, [])
        if len(messages) <= keep_recent:
            return []

        older_messages = messages[:-keep_recent]
        self._sessions[session_id] = messages[-keep_recent:]
        return older_messages


# Global conversation memory instance
_conversation_memory = InMemoryChatMessageHistory(max_messages_per_session=100, session_ttl_seconds=3600)

# Global ledger instances for fact and assumption tracking
_fact_ledger = FactLedger()
_assumption_ledger = AssumptionLedger()
_conversation_summary = ConversationSummary()

# Confidence thresholds
CONFIDENCE_HIGH_THRESHOLD = 0.85  # Answer directly
CONFIDENCE_MEDIUM_THRESHOLD = 0.60  # Answer with caveat

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
    """RAG-based chatbot for contract Q&A with session-based memory, fact tracking, and assumptions"""
    
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

    ARTIST_INTENT_PHRASES = [
        # Profile / bio
        "artist", "bio", "biography", "about the artist", "about this artist",
        "tell me about", "who is", "who are",
        # Social media
        "social", "social media", "instagram", "tiktok", "youtube", "twitter",
        "facebook", "socials",
        # Streaming / DSP
        "spotify", "apple music", "soundcloud", "dsp", "streaming link",
        "where can i listen", "where to listen", "music link",
        # Contact / press
        "email", "contact", "epk", "press kit", "linktree", "website",
        # Genre / style
        "genre", "genres", "what type of music", "what kind of music",
        "music style", "sound like",
        # General artist info
        "artist info", "artist profile", "artist page", "artist details",
        "background", "artist background",
    ]

    CONTRACT_INTENT_KEYWORDS = [
        "contract", "agreement", "clause", "royalty", "royalties", "split", "splits",
        "payment", "advance", "term", "termination", "copyright", "master rights",
        "publishing rights", "parties", "legal", "streaming", "spotify", "apple music",
        "dsp", "revenue", "net revenue", "gross revenue"
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
        self.fact_ledger = _fact_ledger  # Use global fact ledger
        self.assumption_ledger = _assumption_ledger  # Use global assumption ledger
        self.conversation_summary = _conversation_summary  # Use global conversation summary
    
    def _get_current_scope(self, artist_id: Optional[str] = None, 
                           project_id: Optional[str] = None,
                           contract_id: Optional[str] = None) -> Scope:
        """Create a Scope object from current context"""
        return Scope(artist_id=artist_id, project_id=project_id, contract_id=contract_id)
    
    def _calculate_confidence(self, similarity_score: float) -> float:
        """
        Calculate confidence score from vector similarity.
        Maps similarity (typically 0.3-0.95) to confidence (0.0-1.0).
        
        Args:
            similarity_score: Raw similarity score from vector search
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if similarity_score < MIN_SIMILARITY_THRESHOLD:
            return 0.0
        
        # Normalize to 0-1 range based on practical bounds
        normalized = (similarity_score - MIN_SIMILARITY_THRESHOLD) / (0.95 - MIN_SIMILARITY_THRESHOLD)
        # Apply slight curve to favor higher similarities
        confidence = min(1.0, normalized ** 0.8)
        return round(confidence, 3)
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Get the confidence level enum from a confidence score"""
        if confidence >= CONFIDENCE_HIGH_THRESHOLD:
            return ConfidenceLevel.HIGH
        elif confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _create_pinned_fact(self, fact_type: str, description: str, value: Any,
                            confidence: float, source_reference: str,
                            scope: Scope, source_type: str = "document") -> PinnedFact:
        """Helper to create a PinnedFact with proper defaults"""
        return PinnedFact(
            fact_type=fact_type,
            description=description,
            value=value,
            confidence=confidence,
            source_type=source_type,
            source_reference=source_reference,
            scope=scope
        )
    
    def _extract_and_pin_facts(self, answer: str, query: str, sources: List[Dict],
                               session_id: str, scope: Scope) -> List[PinnedFact]:
        """
        Extract facts from an answer and pin them to the ledger.
        
        Args:
            answer: The LLM-generated answer
            query: The original query
            sources: Source documents used
            session_id: Session ID for storage
            scope: Current scope for the facts
            
        Returns:
            List of extracted and pinned facts
        """
        extracted_facts = []
        
        # Use existing extraction logic
        extracted_data = self._extract_structured_data(answer, query)
        
        # Calculate base confidence from sources
        base_confidence = 0.0
        source_ref = "unknown"
        if sources:
            highest_score = max(s.get('score', 0) for s in sources)
            base_confidence = self._calculate_confidence(highest_score)
            source_ref = sources[0].get('contract_file', 'document')
        
        # Pin royalty splits
        if 'royalty_splits' in extracted_data:
            for royalty_type, splits in extracted_data['royalty_splits'].items():
                fact = self._create_pinned_fact(
                    fact_type=f"royalty_splits_{royalty_type}",
                    description=f"{royalty_type.title()} royalty splits",
                    value=splits,
                    confidence=base_confidence,
                    source_reference=source_ref,
                    scope=scope
                )
                self.fact_ledger.add_fact(session_id, fact)
                extracted_facts.append(fact)
        
        # Pin parties
        if 'parties' in extracted_data:
            fact = self._create_pinned_fact(
                fact_type="parties",
                description="Contract parties",
                value=extracted_data['parties'],
                confidence=base_confidence,
                source_reference=source_ref,
                scope=scope
            )
            self.fact_ledger.add_fact(session_id, fact)
            extracted_facts.append(fact)
        
        # Pin payment terms
        if 'payment_terms' in extracted_data:
            fact = self._create_pinned_fact(
                fact_type="payment_terms",
                description="Payment terms",
                value=extracted_data['payment_terms'],
                confidence=base_confidence,
                source_reference=source_ref,
                scope=scope
            )
            self.fact_ledger.add_fact(session_id, fact)
            extracted_facts.append(fact)
        
        # Pin term length
        if 'term_length' in extracted_data:
            fact = self._create_pinned_fact(
                fact_type="term_length",
                description="Contract term length",
                value=extracted_data['term_length'],
                confidence=base_confidence,
                source_reference=source_ref,
                scope=scope
            )
            self.fact_ledger.add_fact(session_id, fact)
            extracted_facts.append(fact)
        
        # Pin advances
        if 'advances' in extracted_data:
            fact = self._create_pinned_fact(
                fact_type="advances",
                description="Advance payment",
                value=extracted_data['advances'],
                confidence=base_confidence,
                source_reference=source_ref,
                scope=scope
            )
            self.fact_ledger.add_fact(session_id, fact)
            extracted_facts.append(fact)
        
        return extracted_facts
    
    def _get_pinned_facts_for_query(self, query: str, session_id: str, 
                                     scope: Scope) -> List[PinnedFact]:
        """
        Get relevant pinned facts for a query based on detected type.
        
        Args:
            query: User's question
            session_id: Session ID
            scope: Current scope
            
        Returns:
            List of relevant facts sorted by confidence
        """
        query_lower = query.lower()
        relevant_facts = []
        
        # Determine what type of fact might be relevant
        if any(kw in query_lower for kw in ['royalty', 'split', 'percentage', 'share']):
            royalty_type = self._detect_royalty_type(query_lower)
            fact = self.fact_ledger.get_fact_by_type(
                session_id, f"royalty_splits_{royalty_type}", scope
            )
            if fact:
                relevant_facts.append(fact)
            # Also check general if specific type not found
            if not fact and royalty_type != 'general':
                fact = self.fact_ledger.get_fact_by_type(
                    session_id, "royalty_splits_general", scope
                )
                if fact:
                    relevant_facts.append(fact)
        
        if any(kw in query_lower for kw in ['parties', 'who signed', 'signatories']):
            fact = self.fact_ledger.get_fact_by_type(session_id, "parties", scope)
            if fact:
                relevant_facts.append(fact)
        
        if any(kw in query_lower for kw in ['payment', 'terms', 'net', 'days']):
            fact = self.fact_ledger.get_fact_by_type(session_id, "payment_terms", scope)
            if fact:
                relevant_facts.append(fact)
        
        if any(kw in query_lower for kw in ['term', 'duration', 'length', 'period']):
            fact = self.fact_ledger.get_fact_by_type(session_id, "term_length", scope)
            if fact:
                relevant_facts.append(fact)
        
        if any(kw in query_lower for kw in ['advance', 'upfront', 'signing']):
            fact = self.fact_ledger.get_fact_by_type(session_id, "advances", scope)
            if fact:
                relevant_facts.append(fact)
        
        # Sort by confidence (highest first)
        relevant_facts.sort(key=lambda f: f.confidence, reverse=True)
        return relevant_facts
    
    def _answer_from_pinned_facts(self, query: str, facts: List[PinnedFact],
                                   session_id: str) -> Optional[Dict]:
        """
        Try to answer a query using pinned facts if confidence is sufficient.
        
        Args:
            query: User's question
            facts: Relevant pinned facts
            session_id: Session ID
            
        Returns:
            Response dict if answerable from facts, None otherwise
        """
        if not facts:
            return None
        
        # Get the highest confidence fact
        best_fact = facts[0]
        confidence_level = best_fact.get_confidence_level()
        
        # Only answer from facts if confidence is high enough
        if confidence_level == ConfidenceLevel.LOW:
            logger.info(f"[Facts] Low confidence ({best_fact.confidence:.2f}) - will search documents")
            return None
        
        # Generate answer from fact
        self._add_to_memory(session_id, "user", query)
        
        # Format the answer based on fact type and confidence level
        value = best_fact.value
        caveat = "" if confidence_level == ConfidenceLevel.HIGH else "Based on what we discussed earlier, "
        
        if best_fact.fact_type.startswith('royalty_splits'):
            if isinstance(value, list):
                splits_text = ", ".join([f"{s['party']}: {s['percentage']}%" for s in value])
                answer = f"{caveat}The royalty splits are: {splits_text}"
            else:
                answer = f"{caveat}The royalty information: {value}"
        elif best_fact.fact_type == 'parties':
            if isinstance(value, list):
                answer = f"{caveat}The contract parties are: {', '.join(value)}"
            else:
                answer = f"{caveat}The contract parties are: {value}"
        elif best_fact.fact_type == 'payment_terms':
            answer = f"{caveat}The payment terms: {value}"
        elif best_fact.fact_type == 'term_length':
            answer = f"{caveat}The contract term is {value}."
        elif best_fact.fact_type == 'advances':
            answer = f"{caveat}The advance is {value}."
        else:
            answer = f"{caveat}{value}"
        
        self._add_to_memory(session_id, "assistant", answer)
        
        return {
            "query": query,
            "answer": answer,
            "confidence": ConfidenceLevel.CONTEXT.value,
            "sources": [{"contract_file": best_fact.source_reference, "score": best_fact.confidence, "project_name": ""}],
            "search_results_count": 0,
            "session_id": session_id,
            "answered_from": "pinned_facts",
            "highest_score": best_fact.confidence,
            "extracted_facts": [f.to_dict() for f in facts],
            "confidence_score": best_fact.confidence
        }
    
    def _add_confidence_caveat(self, answer: str, confidence_level: ConfidenceLevel) -> str:
        """Add appropriate caveat based on confidence level"""
        if confidence_level == ConfidenceLevel.MEDIUM:
            if not answer.lower().startswith("based on"):
                return f"Based on the available information: {answer}"
        return answer
    
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
        Compute a hash of key context entities to detect context changes.
        
        Args:
            context: Conversation context
            
        Returns:
            Hash string of contract IDs
        """
        if not context:
            return ""
        
        contracts_discussed = context.get('contracts_discussed', [])
        artists_discussed = context.get('artists_discussed', [])
        contract_ids = sorted([c.get('id', '') for c in contracts_discussed if isinstance(c, dict)])
        artist_ids = sorted([a.get('id', '') for a in artists_discussed if isinstance(a, dict)])
        payload = {
            "contract_ids": contract_ids,
            "artist_ids": artist_ids
        }
        return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    
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
    
    def _classify_query(self, query: str, artist_data: Optional[Dict] = None) -> str:
        """
        Use LLM to classify whether a query is about artist info or contracts.
        
        Args:
            query: User's question
            artist_data: Optional artist data to help with context (e.g. artist name)
            
        Returns:
            "artist" if query is about artist profile/info
            "contract" if query is about contracts/agreements
        """
        # Add artist context if available to help disambiguate
        artist_context = ""
        if artist_data and artist_data.get('name'):
            artist_name = artist_data.get('name')
            artist_context = f"\nIMPORTANT CONTEXT: The current artist is '{artist_name}'. If the user asks about '{artist_name}', classify it as ARTIST."

        system_prompt = f"""You are a query classifier. Determine if the user's question is about:

1. ARTIST - Questions about the artist's profile, bio, social media, streaming links, genres, contact info, EPK, press kit, etc.
   Examples: "What's the artist's bio?", "What are their social media links?", "What genre do they make?", "What's their Spotify?"{artist_context}

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

    def _is_artist_intent_query(self, query: str) -> bool:
        """Detect artist-intent queries via deterministic phrase matching.

        Returns False if the query also contains contract-related keywords,
        since those ambiguous queries should go through LLM routing instead.
        """
        query_lower = query.lower().strip()
        has_artist_signal = any(phrase in query_lower for phrase in self.ARTIST_INTENT_PHRASES)
        if not has_artist_signal:
            return False
        # If query also has contract keywords, it's ambiguous — don't fast-path
        has_contract_signal = any(keyword in query_lower for keyword in self.CONTRACT_INTENT_KEYWORDS)
        return not has_contract_signal

    def _is_contract_intent_query(self, query: str) -> bool:
        """Detect explicit contract-intent queries via deterministic keyword matching."""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.CONTRACT_INTENT_KEYWORDS)

    def _has_non_empty_value(self, value: Any) -> bool:
        """Determine whether a value is meaningful (not empty/null-like)."""
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) > 0
        return True

    def _has_contract_context(self, context: Optional[Dict]) -> bool:
        """Check whether there is meaningful prior contract context in the conversation."""
        if not context:
            return False
        contracts_discussed = context.get("contracts_discussed", [])
        for contract in contracts_discussed:
            extracted_data = contract.get("data_extracted", {}) if isinstance(contract, dict) else {}
            if isinstance(extracted_data, dict) and any(self._has_non_empty_value(v) for v in extracted_data.values()):
                return True
        return False

    def _build_source_selection_response(self, query: str, session_id: Optional[str] = None) -> Dict:
        """Ask user whether to use artist profile or contract/context as answer source."""
        response = (
            "I can answer this in two ways: using the artist profile from Supabase, "
            "or using the selected contract/context. Which source should I use?"
        )

        self._add_to_memory(session_id, "user", query)
        self._add_to_memory(session_id, "assistant", response)

        return {
            "query": query,
            "answer": response,
            "confidence": "needs_source_selection",
            "sources": [],
            "search_results_count": 0,
            "session_id": session_id,
            "answered_from": "source_disambiguation",
            "show_quick_actions": True,
            "needs_source_selection": True,
            "quick_actions": [
                {
                    "id": "artist_profile",
                    "label": "Use artist profile",
                    "query": query,
                    "source_preference": "artist_profile"
                },
                {
                    "id": "contract_context",
                    "label": "Use contract/context",
                    "query": query,
                    "source_preference": "contract_context"
                }
            ]
        }

    def _build_artist_source_selection_response(self, query: str, session_id: Optional[str] = None) -> Dict:
        """Ask user whether to use artist profile or continue conversation (use history)."""
        response = (
            "I can answer this using the artist profile, or I can continue based on our "
            "conversation so far. Which would you prefer?"
        )

        # Don't store disambiguation in memory — it's a UI flow control message,
        # not part of the actual conversation. Storing it pollutes history.

        return {
            "query": query,
            "answer": response,
            "confidence": "needs_source_selection",
            "sources": [],
            "search_results_count": 0,
            "session_id": session_id,
            "answered_from": "source_disambiguation",
            "show_quick_actions": True,
            "needs_source_selection": True,
            "quick_actions": [
                {
                    "id": "artist_profile",
                    "label": "Artist Profile",
                    "query": query,
                    "source_preference": "artist_profile"
                },
                {
                    "id": "conversation_history",
                    "label": "Conversation",
                    "query": query,
                    "source_preference": "conversation_history"
                }
            ]
        }

    def _resolve_query_route(self,
                             query: str,
                             artist_data: Optional[Dict],
                             context: Optional[Dict],
                             source_preference: Optional[str] = None) -> str:
        """
        Resolve whether query should route to artist or contract handling.

        Returns one of: "artist", "contract", "disambiguate".
        """
        if source_preference == "artist_profile":
            return "artist"
        if source_preference == "contract_context":
            return "contract"

        has_artist_data = artist_data is not None
        has_contract_context = self._has_contract_context(context)
        is_contract_intent = self._is_contract_intent_query(query)
        is_artist_intent = self._is_artist_intent_query(query)

        # Contract intent always takes precedence for legal/royalty questions.
        if is_contract_intent:
            return "contract"

        if has_artist_data and is_artist_intent:
            if has_contract_context:
                return "disambiguate"
            return "artist"

        if has_artist_data and not has_contract_context and not self._is_contract_intent_query(query):
            return "artist"

        return "contract"

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract the first JSON object found in text."""
        if not text:
            return None

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def _build_routing_context_summary(self,
                                       context: Optional[Dict],
                                       artist_data: Optional[Dict],
                                       project_id: Optional[str],
                                       contract_id: Optional[str],
                                       source_preference: Optional[str]) -> str:
        """Build compact context summary for routing decisions."""
        context = context or {}
        artists_discussed = context.get("artists_discussed", [])
        contracts_discussed = context.get("contracts_discussed", [])

        artist_names = []
        for artist in artists_discussed[-5:]:
            if isinstance(artist, dict):
                artist_names.append(artist.get("name") or "Unknown artist")

        contract_names = []
        for contract in contracts_discussed[-5:]:
            if isinstance(contract, dict):
                contract_names.append(contract.get("name") or "Unknown contract")

        current_artist_name = ""
        if artist_data and artist_data.get("name"):
            current_artist_name = artist_data.get("name")

        summary = {
            "source_preference": source_preference,
            "has_project_selected": bool(project_id),
            "has_contract_selected": bool(contract_id),
            "current_artist": current_artist_name,
            "artists_discussed_count": len(artists_discussed),
            "artists_discussed": artist_names,
            "contracts_discussed_count": len(contracts_discussed),
            "contracts_discussed": contract_names,
            "has_contract_context": self._has_contract_context(context)
        }
        return json.dumps(summary, ensure_ascii=False)

    def _build_default_route_decision(self,
                                      query: str,
                                      artist_data: Optional[Dict],
                                      context: Optional[Dict],
                                      source_preference: Optional[str],
                                      project_id: Optional[str],
                                      contract_id: Optional[str]) -> RouteDecision:
        """Fallback route decision when LLM routing is unavailable."""
        route = self._resolve_query_route(query, artist_data, context, source_preference)

        answer_mode: Literal["context", "retrieve"] = "retrieve"
        reason = "fallback_retrieval"
        retrieval_reason = "general_retrieval"

        if route == "artist":
            answer_mode = "context" if context else "retrieve"
            reason = "fallback_artist_route"
        elif route == "disambiguate":
            answer_mode = "context"
            reason = "fallback_disambiguate"
        elif context and not contract_id and not project_id:
            # Without a selected contract/project, context is often the only useful source.
            answer_mode = "context"
            reason = "fallback_context_no_project"

        return RouteDecision(
            route=route,
            answer_mode=answer_mode,
            confidence=0.35,
            reason=reason,
            retrieval_reason=retrieval_reason
        )

    def _llm_route_decision(self,
                            query: str,
                            artist_data: Optional[Dict],
                            context: Optional[Dict],
                            source_preference: Optional[str] = None,
                            project_id: Optional[str] = None,
                            contract_id: Optional[str] = None) -> RouteDecision:
        """
        Use LLM to decide route and whether to answer from context or retrieve.
        """
        if source_preference == "artist_profile":
            return RouteDecision(
                route="artist",
                answer_mode="context",
                confidence=1.0,
                reason="source_preference_artist",
                retrieval_reason="none"
            )

        if source_preference == "contract_context":
            return RouteDecision(
                route="contract",
                answer_mode="context" if context else "retrieve",
                confidence=1.0,
                reason="source_preference_contract_context",
                retrieval_reason="general_retrieval"
            )

        routing_context = self._build_routing_context_summary(
            context=context,
            artist_data=artist_data,
            project_id=project_id,
            contract_id=contract_id,
            source_preference=source_preference
        )

        system_prompt = """You are a routing planner for a music assistant.
Given the user query and current state, return ONLY valid JSON with:
{
  "route": "artist" | "contract" | "disambiguate",
  "answer_mode": "context" | "retrieve",
  "confidence": number between 0 and 1,
  "reason": "short_snake_case_reason",
  "retrieval_reason": "general_retrieval" | "missing_streaming_splits" | "missing_publishing_splits" | "missing_mechanical_splits" | "missing_sync_splits" | "missing_master_splits" | "missing_performance_splits" | "missing_general_splits" | "missing_parties" | "missing_payment_terms" | "missing_term_length" | "missing_advances"
}

Rules:
1) Prefer route="artist" for artist profile/social/bio/genre/link questions.
2) Prefer route="contract" for legal/royalty/term/payment/party questions.
3) For ambiguous artist+contract wording where user intent is truly unclear, use route="disambiguate".
4) If the question can be answered from provided context/history, set answer_mode="context".
5) If evidence likely needs contract retrieval, set answer_mode="retrieve".
6) If selected contract exists and question is contract-specific, lean retrieve unless context clearly contains exact requested facts.
7) Return JSON only, no markdown or prose."""

        user_prompt = f"""User Query:\n{query}\n\nState:\n{routing_context}"""

        try:
            response = openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=250
            )

            content = (response.choices[0].message.content or "").strip()
            parsed = self._extract_json_object(content)
            if not parsed:
                raise ValueError("No valid JSON route decision returned")

            route = parsed.get("route", "contract")
            answer_mode = parsed.get("answer_mode", "retrieve")
            confidence = float(parsed.get("confidence", 0.0) or 0.0)
            reason = parsed.get("reason", "llm_default")
            retrieval_reason = parsed.get("retrieval_reason", "general_retrieval")

            if route not in {"artist", "contract", "disambiguate"}:
                route = "contract"
            if answer_mode not in {"context", "retrieve"}:
                answer_mode = "retrieve"
            confidence = max(0.0, min(1.0, confidence))

            decision = RouteDecision(
                route=route,
                answer_mode=answer_mode,
                confidence=confidence,
                reason=reason,
                retrieval_reason=retrieval_reason
            )
            logger.info(
                f"[Routing] LLM decision route={decision.route}, mode={decision.answer_mode}, "
                f"confidence={decision.confidence:.2f}, reason={decision.reason}, retrieval_reason={decision.retrieval_reason}"
            )
            return decision
        except Exception as e:
            logger.error(f"[Routing] LLM route decision failed: {e}")
            fallback = self._build_default_route_decision(
                query=query,
                artist_data=artist_data,
                context=context,
                source_preference=source_preference,
                project_id=project_id,
                contract_id=contract_id
            )
            logger.info(
                f"[Routing] Fallback decision route={fallback.route}, mode={fallback.answer_mode}, "
                f"confidence={fallback.confidence:.2f}, reason={fallback.reason}"
            )
            return fallback

    def _should_answer_from_context(self,
                                    decision: RouteDecision,
                                    context: Optional[Dict],
                                    contract_id: Optional[str]) -> bool:
        """Apply policy guardrails for using context responses."""
        if not context:
            return False
        if decision.answer_mode != "context":
            return False

        # Soft override: if contract is selected, require stronger confidence for context-only answer.
        threshold = 0.85 if contract_id else 0.6
        if decision.confidence < threshold:
            logger.info(
                f"[Routing] Context mode requested but confidence {decision.confidence:.2f} < threshold {threshold:.2f}; retrieving"
            )
            return False
        return True
    
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
6. If information is not available, simply say so

FORMATTING:
- Use markdown for structure, not decoration — keep it minimal
- Use **bold** for field labels (e.g., **Bio:**, **Genres:**)
- Use bullet points for listing multiple items (socials, links, genres)
- Format URLs as markdown links: [Platform Name](url)
- Only use ### headers when answering about multiple fields — skip headers for single-field answers
- Do NOT add filler phrases, introductions, or conclusions — just the data

Examples:
- If asked "What's the artist's bio?" and bio is "House DJ" → Answer: "House DJ"
- If asked "What's their genre?" and genre is "Electronic" → Answer: "Electronic"
- If asked "What are their socials?" → List as bullet points with markdown links"""

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

            raw_answer = response.choices[0].message.content
            answer = (raw_answer or "").strip()
            if not answer:
                answer = self._build_artist_fallback_answer(artist_data, query)
            
        except Exception as e:
            logger.error(f"Error generating artist answer: {e}")
            answer = self._build_artist_fallback_answer(artist_data, query)
        
        # Store assistant response in memory
        self._add_to_memory(session_id, "assistant", answer)
        
        # Extract structured artist data for frontend tracking
        extracted_artist_data = self._extract_artist_data(artist_data)
        
        return {
            "query": query,
            "answer": answer,
            "confidence": "artist_data",
            "sources": [],
            "search_results_count": 0,
            "session_id": session_id,
            "answered_from": "artist_data",
            "extracted_data": extracted_artist_data,
            "show_quick_actions": False
        }

    def _build_artist_fallback_answer(self, artist_data: Dict, query: str) -> str:
        """Build a deterministic artist answer when LLM output is empty or fails."""
        extracted = self._extract_artist_data(artist_data)
        artist_name = artist_data.get("name", "the artist")
        query_lower = query.lower()

        if "bio" in query_lower and extracted.get("bio"):
            return str(extracted["bio"])

        if any(kw in query_lower for kw in ["genre", "genres"]) and extracted.get("genres"):
            genres = extracted.get("genres", [])
            if isinstance(genres, list):
                return ", ".join([str(genre) for genre in genres])
            return str(genres)

        if any(kw in query_lower for kw in ["social", "instagram", "tiktok", "youtube", "twitter", "facebook"]):
            social = extracted.get("social_media", {})
            if isinstance(social, dict) and social:
                return "\n".join([f"{platform}: {url}" for platform, url in social.items()])

        if any(kw in query_lower for kw in ["stream", "spotify", "apple music", "soundcloud", "dsp"]):
            links = extracted.get("streaming_links", {})
            if isinstance(links, dict) and links:
                return "\n".join([f"{platform}: {url}" for platform, url in links.items()])

        if any(kw in query_lower for kw in ["email", "contact"]) and extracted.get("email"):
            return str(extracted["email"])

        summary_parts = []
        if extracted.get("bio"):
            summary_parts.append(f"Bio: {extracted['bio']}")
        if extracted.get("genres"):
            genres = extracted["genres"]
            if isinstance(genres, list):
                summary_parts.append(f"Genres: {', '.join([str(genre) for genre in genres])}")
            else:
                summary_parts.append(f"Genres: {genres}")

        if summary_parts:
            return f"Here’s what I have for {artist_name}: " + " | ".join(summary_parts)

        return f"I found the artist record for {artist_name}, but profile details are currently empty."
    
    def _extract_artist_data(self, artist_data: Dict) -> Dict:
        """
        Extract structured artist data for frontend tracking.
        
        Args:
            artist_data: Artist information from database
            
        Returns:
            Dict with structured artist data for frontend
        """
        extracted = {}
        
        # Extract bio
        if artist_data.get("bio"):
            extracted["bio"] = artist_data["bio"]
        
        # Extract genres
        if artist_data.get("genres"):
            extracted["genres"] = artist_data["genres"]
        
        # Extract email
        if artist_data.get("email"):
            extracted["email"] = artist_data["email"]
        
        # Extract social media links
        social_media = {}
        if artist_data.get("social_instagram"):
            social_media["instagram"] = artist_data["social_instagram"]
        if artist_data.get("social_tiktok"):
            social_media["tiktok"] = artist_data["social_tiktok"]
        if artist_data.get("social_youtube"):
            social_media["youtube"] = artist_data["social_youtube"]
        if artist_data.get("social_twitter"):
            social_media["twitter"] = artist_data["social_twitter"]
        if artist_data.get("social_facebook"):
            social_media["facebook"] = artist_data["social_facebook"]
        
        if social_media:
            extracted["social_media"] = social_media
        
        # Extract streaming links
        streaming_links = {}
        if artist_data.get("dsp_spotify"):
            streaming_links["spotify"] = artist_data["dsp_spotify"]
        if artist_data.get("dsp_apple_music"):
            streaming_links["apple_music"] = artist_data["dsp_apple_music"]
        if artist_data.get("dsp_soundcloud"):
            streaming_links["soundcloud"] = artist_data["dsp_soundcloud"]
        
        if streaming_links:
            extracted["streaming_links"] = streaming_links
        
        return extracted
    
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
        Get conversation history for context, prepending rolling summary if available.

        Args:
            session_id: Session identifier
            max_turns: Maximum conversation turns to include

        Returns:
            List of message dicts for LLM
        """
        if not session_id:
            return []

        messages = []

        # Prepend rolling summary if one exists (from previous summarization)
        summary = _conversation_summary.get_summary(session_id)
        if summary:
            messages.append({
                "role": "system",
                "content": f"Summary of earlier conversation:\n{summary}"
            })

        messages.extend(self.memory.get_messages_for_llm(session_id, limit=max_turns))
        return messages
    
    def _add_to_memory(self, session_id: Optional[str], role: str, content: str, metadata: Dict = None) -> None:
        """Add a message to session memory."""
        if session_id:
            self.memory.add_message(session_id, role, content, metadata)

    def _summarize_if_needed(self, session_id: Optional[str]) -> None:
        """
        Check if conversation history exceeds token limits and summarize if needed.
        Keeps the most recent 6 messages verbatim and compresses older messages
        into a rolling summary using the existing ConversationSummary class.

        Target: Keep history under 250k tokens (leaving buffer below 300k ceiling).
        """
        if not session_id:
            return

        TOKEN_LIMIT = 250_000
        estimated_tokens = self.memory.estimate_token_count(session_id)

        if estimated_tokens <= TOKEN_LIMIT:
            return

        logger.info(
            f"[Summarization] Token estimate ({estimated_tokens}) exceeds limit ({TOKEN_LIMIT}). "
            f"Summarizing older messages."
        )

        # Get older messages that will be summarized, keeping recent 6 verbatim
        older_messages = self.memory.trim_to_recent(session_id, keep_recent=6)

        if not older_messages:
            return

        # Build text from older messages for summarization
        existing_summary = _conversation_summary.get_summary(session_id)
        messages_text = "\n".join(
            f"{msg.role.upper()}: {msg.content}" for msg in older_messages
        )

        summary_input = ""
        if existing_summary:
            summary_input = f"Previous summary:\n{existing_summary}\n\nNew messages to incorporate:\n{messages_text}"
        else:
            summary_input = f"Conversation to summarize:\n{messages_text}"

        try:
            response = openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize this conversation concisely, preserving all key facts, "
                            "data points (royalty splits, payment terms, parties, etc.), decisions, "
                            "and important details. The summary will be used as context for future "
                            "questions, so accuracy is critical. Keep it under 2000 words."
                        )
                    },
                    {"role": "user", "content": summary_input}
                ],
                max_completion_tokens=2000
            )

            summary = (response.choices[0].message.content or "").strip()
            if summary:
                _conversation_summary.update_summary(session_id, summary)
                logger.info(
                    f"[Summarization] Summarized {len(older_messages)} messages. "
                    f"Summary length: {len(summary)} chars"
                )

        except Exception as e:
            logger.error(f"[Summarization] Error summarizing conversation: {e}")
            # On failure, keep the trimmed state but without summary

    def _get_targeted_query(self, reason: str, original_query: str) -> str:
        """
        Generate a targeted search query based on missing data reason.
        Maps reasons like 'missing_streaming_splits' to focused queries
        like 'streaming royalty splits percentages'.
        
        Args:
            reason: The reason from _can_answer_from_context (e.g., 'missing_streaming_splits')
            original_query: The user's original question
            
        Returns:
            Targeted search query optimized for retrieving the missing data
        """
        # Extract the data type from the reason
        if reason.startswith("missing_"):
            data_type = reason.replace("missing_", "")
            
            # Map data types to targeted search queries
            targeted_queries = {
                "streaming_splits": "streaming royalty splits percentages parties",
                "publishing_splits": "publishing royalty splits percentages parties",
                "mechanical_splits": "mechanical royalty splits percentages parties",
                "sync_splits": "sync synchronization royalty splits percentages parties",
                "master_splits": "master recording royalty splits percentages parties",
                "performance_splits": "performance royalty splits percentages parties",
                "general_splits": "royalty splits percentages parties",
                "parties": "contract parties signatories names",
                "payment_terms": "payment terms net days accounting",
                "term_length": "contract term duration length period years",
                "advances": "advance payment upfront signing bonus amount"
            }
            
            # Return the targeted query if we have a mapping, otherwise use original
            targeted = targeted_queries.get(data_type)
            if targeted:
                logger.info(f"[Targeted Retrieval] Generated targeted query for {data_type}: '{targeted}'")
                return targeted
        
        # If no mapping found, return original query
        logger.info(f"[Targeted Retrieval] No mapping for reason '{reason}', using original query")
        return original_query
    
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
            logger.info(f"[Context] Missing data in context ({reason}) - will use targeted retrieval")
        
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
        artists_discussed = context.get('artists_discussed', [])
        logger.info(f"[Context] Checking if can answer from context. Contracts: {len(contracts_discussed)}, Artists: {len(artists_discussed)}")
        
        query_lower = query.lower()
        
        # Check for artist comparisons
        comparison_keywords = ['compare', 'difference', 'versus', 'vs', 'between', 'both']
        artist_keywords = ['artist', 'bio', 'social', 'streaming', 'genre', 'music']
        
        if any(kw in query_lower for kw in comparison_keywords):
            # Check if asking about artists
            if any(kw in query_lower for kw in artist_keywords) and len(artists_discussed) >= 2:
                logger.info(f"[Context] Artist comparison detected with {len(artists_discussed)} artists")
                return True, "artist_comparison"
            
            # Check if we have 2+ contracts with ANY extracted data for general comparison
            contracts_with_any_data = [c for c in contracts_discussed 
                                       if c.get('data_extracted') and any(c.get('data_extracted', {}).values())]
            if len(contracts_with_any_data) >= 2:
                logger.info(f"[Context] Contract comparison detected with {len(contracts_with_any_data)} contracts having data")
                return True, "contract_comparison"
            
            # Check if asking about contracts (existing logic continues below)

        # Single-artist re-ask: asking about artist data when we have extracted data in context
        artist_data_keywords = ['genre', 'genres', 'bio', 'biography', 'social', 'socials',
                                'instagram', 'tiktok', 'spotify', 'streaming', 'email', 'contact',
                                'website', 'epk', 'linktree', 'apple music', 'soundcloud',
                                'youtube', 'twitter', 'facebook']
        if any(kw in query_lower for kw in artist_data_keywords) and len(artists_discussed) >= 1:
            for artist in artists_discussed:
                if artist.get('data_extracted') and any(artist['data_extracted'].values()):
                    logger.info(f"[Context] Artist data re-ask detected with extracted data for {artist.get('name')}")
                    return True, "artist_data_reask"

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
        royalty_keywords = ['royalty', 'split', 'percentage', 'share', 'payout',
                           'revenue', 'each person', 'each party', 'who gets']
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

                # Check for aggregate/summary queries across multiple contracts
                aggregate_keywords = ['summary', 'everyone', 'each person', 'each party',
                                      'who gets', 'how much', 'payout', 'breakdown',
                                      'combined', 'all parties', 'all contracts']
                if any(kw in query_lower for kw in aggregate_keywords):
                    contracts_with_type = sum(1 for c in contracts_discussed
                                              if self._contract_has_royalty_type(c, royalty_type))
                    if contracts_with_type >= 2:
                        logger.info(f"[Context] Aggregate split query detected with {contracts_with_type} contracts")
                        return True, f"aggregate_{royalty_type}"

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

    def _is_multi_contract_split_query(self, query: str, context: Optional[Dict]) -> bool:
        """
        Detect if query is asking about royalty splits across multiple contracts.
        Returns True when: query mentions split/royalty keywords AND 2+ contracts
        in context have non-empty royalty_splits data.
        """
        if not context:
            return False

        query_lower = query.lower()
        split_keywords = [
            'split', 'splits', 'royalty', 'royalties', 'payout', 'payouts',
            'revenue', 'share', 'shares', 'percentage', 'percentages',
            'who gets', 'each person', 'each party', 'how much does',
            'breakdown', 'combined', 'everyone', 'all parties'
        ]
        if not any(kw in query_lower for kw in split_keywords):
            return False

        contracts_discussed = context.get('contracts_discussed', [])
        contracts_with_splits = 0
        for contract in contracts_discussed:
            data = contract.get('data_extracted', {})
            royalty_splits = data.get('royalty_splits', {})
            if isinstance(royalty_splits, dict) and any(royalty_splits.values()):
                contracts_with_splits += 1
            elif isinstance(royalty_splits, list) and len(royalty_splits) > 0:
                contracts_with_splits += 1

        if contracts_with_splits >= 2:
            logger.info(f"[Aggregation] Multi-contract split query detected: {contracts_with_splits} contracts with splits")
            return True
        return False

    def _extract_revenue_amount(self, query: str) -> Optional[float]:
        """Extract a dollar/revenue amount from the query string."""
        query_lower = query.lower()

        # $10,000 or $10,000.00
        match = re.search(r'\$\s*([\d,]+(?:\.\d{1,2})?)', query)
        if match:
            return float(match.group(1).replace(',', ''))

        # $10k / $10K / 10k dollars
        match = re.search(r'\$?\s*(\d+(?:\.\d+)?)\s*[kK]\b', query)
        if match:
            return float(match.group(1)) * 1000

        # $10m / $10M
        match = re.search(r'\$?\s*(\d+(?:\.\d+)?)\s*[mM]\b', query)
        if match:
            return float(match.group(1)) * 1_000_000

        # "10000 dollars" / "10,000 dollars" / "10000 in revenue"
        match = re.search(r'([\d,]+(?:\.\d{1,2})?)\s*(?:dollars?|usd|in revenue|in earnings|in income)', query_lower)
        if match:
            return float(match.group(1).replace(',', ''))

        return None

    def _aggregate_splits_across_contracts(self, context: Dict, query: str) -> Dict:
        """
        Aggregate royalty splits across multiple contracts by deduplicating per party.
        Each contract describes a portion of the SAME 100% pie — splits are NOT summed
        per person across contracts.

        Returns:
            Dict with keys: parties, total_percentage, revenue_amount, payouts
        """
        royalty_type = self._detect_royalty_type(query.lower())
        contracts_discussed = context.get('contracts_discussed', [])

        # Collect per-party data: {normalized_name: {display_name, percentage, source_contracts}}
        party_map: Dict[str, Dict] = {}

        for contract in contracts_discussed:
            contract_name = contract.get('name', 'Unknown Contract')
            data = contract.get('data_extracted', {})
            royalty_splits = data.get('royalty_splits', {})

            splits_list = []
            if isinstance(royalty_splits, dict):
                # Try specific type first, then general
                splits_list = royalty_splits.get(royalty_type, [])
                if not splits_list and royalty_type != 'general':
                    splits_list = royalty_splits.get('general', [])
                # If still empty, grab the first non-empty type
                if not splits_list:
                    for _type, _splits in royalty_splits.items():
                        if _splits:
                            splits_list = _splits
                            break
            elif isinstance(royalty_splits, list):
                splits_list = royalty_splits

            for split in splits_list:
                if not isinstance(split, dict):
                    continue
                party_name = split.get('party', split.get('name', ''))
                percentage = split.get('percentage', split.get('share', 0))
                if not party_name:
                    continue

                try:
                    percentage = float(percentage)
                except (ValueError, TypeError):
                    continue

                norm_name = normalize_name(party_name)
                if norm_name in party_map:
                    # Take MAX percentage (should be equal across contracts, but defensive)
                    if percentage > party_map[norm_name]['percentage']:
                        party_map[norm_name]['percentage'] = percentage
                    if contract_name not in party_map[norm_name]['source_contracts']:
                        party_map[norm_name]['source_contracts'].append(contract_name)
                else:
                    party_map[norm_name] = {
                        'display_name': party_name.strip(),
                        'percentage': percentage,
                        'source_contracts': [contract_name]
                    }

        # Build result
        parties = []
        total_pct = 0.0
        for _norm, info in sorted(party_map.items(), key=lambda x: x[1]['percentage'], reverse=True):
            parties.append({
                'name': info['display_name'],
                'percentage': info['percentage'],
                'source_contracts': info['source_contracts']
            })
            total_pct += info['percentage']

        revenue_amount = self._extract_revenue_amount(query)
        payouts = None
        if revenue_amount and parties:
            payouts = []
            for p in parties:
                payout = revenue_amount * (p['percentage'] / 100.0)
                payouts.append({'name': p['name'], 'amount': payout})

        result = {
            'parties': parties,
            'total_percentage': total_pct,
            'royalty_type': royalty_type,
            'num_contracts': len(contracts_discussed),
            'revenue_amount': revenue_amount,
            'payouts': payouts
        }
        logger.info(f"[Aggregation] Result: {len(parties)} parties, total={total_pct}%, revenue={revenue_amount}")
        return result

    def _format_aggregated_splits_for_llm(self, aggregation: Dict) -> str:
        """Format pre-computed aggregated splits as a text block for the LLM prompt."""
        royalty_type = aggregation.get('royalty_type', 'general')
        type_label = royalty_type.title() if royalty_type != 'general' else 'General'
        num_contracts = aggregation.get('num_contracts', 0)
        parties = aggregation.get('parties', [])
        total_pct = aggregation.get('total_percentage', 0)
        revenue = aggregation.get('revenue_amount')
        payouts = aggregation.get('payouts')

        lines = [
            f"=== PRE-COMPUTED AGGREGATED ROYALTY SPLITS ({type_label}) ===",
            f"(Merged from {num_contracts} contracts — shares of the SAME 100% pie, NOT summed.)",
            ""
        ]

        for p in parties:
            sources = ", ".join(p['source_contracts'])
            lines.append(f"  {p['name']}: {p['percentage']:.1f}%  (appears in: {sources})")

        lines.append(f"\n  Total accounted for: {total_pct:.1f}%")

        if revenue and payouts:
            lines.append(f"\n--- Payout calculation (total revenue: ${revenue:,.2f}) ---")
            for pay in payouts:
                lines.append(f"  {pay['name']}: ${pay['amount']:,.2f}")

        lines.append("=== END AGGREGATED SPLITS ===")
        return "\n".join(lines)

    def _try_answer_from_history(self, query: str, session_id: Optional[str], context: Optional[Dict] = None) -> Optional[Dict]:
        """
        Try to answer a question from conversation history alone.
        Returns None if history is empty or insufficient, allowing fallback to semantic search.

        Args:
            query: User's question
            session_id: Session ID for conversation memory
            context: Optional conversation context

        Returns:
            Dict with answer if history can answer, None otherwise
        """
        if not session_id:
            return None

        history = self._get_conversation_context(session_id, max_turns=10)
        if not history:
            return None

        logger.info(f"[History] Checking if conversation history ({len(history)} messages) can answer: {query}")

        system_prompt = """You are a music industry assistant reviewing a conversation history and structured context.
Determine if the conversation history or additional structured context contains enough information to answer the user's new question.

RULES:
1. If the conversation history OR the additional structured context contains the information needed, provide a complete answer.
2. If the information is NOT in the history AND NOT in the structured context, respond with exactly: NEEDS_RETRIEVAL
3. When answering, be specific and reference the relevant information from previous messages or context.
4. You may compare, summarize, or re-analyze information that was previously discussed.
5. Do NOT make up or infer information that is not present in the history or structured context.
6. For questions about genuinely NEW topics with NO relevant data in history or context, respond with NEEDS_RETRIEVAL.
7. If the structured context contains extracted data (genres, bio, social media, streaming links, etc.) that answers the question, USE it — do not return NEEDS_RETRIEVAL.
8. Do NOT suggest follow-up questions.
9. Format responses using clean, minimal markdown (bold labels, bullet points, markdown links for URLs). Do not embellish — just the data."""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)

        # Include structured context if available for richer answers
        context_note = ""
        if context:
            context_summary = self._format_context_for_llm(context)
            if context_summary and context_summary != "No context available.":
                context_note = f"\n\nAdditional structured context:\n{context_summary}"

        messages.append({"role": "user", "content": f"New question: {query}{context_note}"})

        try:
            response = openai_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_completion_tokens=1000
            )

            answer = (response.choices[0].message.content or "").strip()

            if not answer or "NEEDS_RETRIEVAL" in answer:
                logger.info("[History] History insufficient - falling through to retrieval")
                return None

            logger.info("[History] Successfully answered from conversation history")

            # Store in memory
            self._add_to_memory(session_id, "user", query)
            self._add_to_memory(session_id, "assistant", answer)

            # Extract structured data from the answer
            extracted_data = self._extract_structured_data(answer, query)

            return {
                "query": query,
                "answer": answer,
                "confidence": "conversation_history",
                "sources": [],
                "search_results_count": 0,
                "answered_from": "conversation_history",
                "session_id": session_id,
                "extracted_data": extracted_data if extracted_data else None
            }

        except Exception as e:
            logger.error(f"[History] Error checking conversation history: {e}")
            return None

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

        # Check if this is a multi-contract split query and pre-aggregate
        aggregated_splits_block = ""
        royalty_rule_prompt = ""
        if self._is_multi_contract_split_query(query, context):
            aggregation = self._aggregate_splits_across_contracts(context, query)
            if aggregation['parties']:
                aggregated_splits_block = self._format_aggregated_splits_for_llm(aggregation)
                context_summary = f"{aggregated_splits_block}\n\n{context_summary}"
                logger.info(f"[Aggregation] Prepended aggregated splits to context:\n{aggregated_splits_block}")
                royalty_rule_prompt = """

CRITICAL ROYALTY SPLIT RULE:
The PRE-COMPUTED AGGREGATED ROYALTY SPLITS above are ALREADY CORRECT. These splits represent portions of the SAME 100% pie — they are NOT summed per person across contracts.
- If a person appears in multiple bilateral contracts (e.g., Bob appears in Contract 1 and Contract 2 both at 40%), their share is STILL 40% — NOT 80%.
- Each contract describes a bilateral agreement between two parties, but all contracts refer to the same underlying revenue pool.
- Use ONLY the pre-computed numbers above. Do NOT re-calculate or sum percentages from individual contracts.
- If a payout calculation is included above, present those exact numbers."""

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
{royalty_rule_prompt}

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
3. If comparing artists, always provide a comparison answer using available context
4. If artist context is sparse, explicitly state there is not enough information for a full comparison and list what is missing
5. Be specific about which contract each piece of information came from
6. If information is incomplete, acknowledge what you don't have
7. Format responses using clean, minimal markdown:
   - Use ### headers to separate each artist or contract being compared
   - Use **bold** for field labels, bullet points for attributes
   - Format URLs as markdown links: [Platform Name](url)
   - For comparisons, use headers per entity then a short ### Answer section for the user's specific question
   - Do NOT pad with filler text, introductions, or unnecessary transitions — just structured data and a direct answer
8. Only suggest follow-ups that can be answered from existing context data"""

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
        
        # Determine the specific answered_from value
        query_lower = query.lower()
        comparison_keywords = ['compare', 'difference', 'versus', 'vs', 'between', 'both']
        artist_keywords = ['artist', 'bio', 'social', 'streaming', 'genre', 'music']
        
        answered_from = "conversation_context"
        if any(kw in query_lower for kw in comparison_keywords) and any(kw in query_lower for kw in artist_keywords):
            artists_discussed = context.get('artists_discussed', [])
            if len(artists_discussed) >= 2:
                answered_from = "artist_comparison"
        
        return {
            "query": query,
            "answer": answer,
            "confidence": "context_based",
            "sources": [],
            "search_results_count": 0,
            "answered_from": answered_from,
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
        
        # Format artists discussed for comparisons
        if context.get('artists_discussed'):
            parts.append("\nArtists Discussed:")
            for artist in context['artists_discussed']:
                parts.append(f"\n🎤 {artist.get('name', 'Unknown Artist')}")
                data_extracted = artist.get('data_extracted', {})
                if data_extracted:
                    if data_extracted.get('bio'):
                        parts.append(f"  • Bio: {data_extracted['bio']}")
                    if data_extracted.get('genres'):
                        genres = data_extracted['genres']
                        if isinstance(genres, list):
                            parts.append(f"  • Genres: {', '.join(genres)}")
                        else:
                            parts.append(f"  • Genres: {genres}")
                    if data_extracted.get('social_media'):
                        parts.append("  • Social Media:")
                        for platform, url in data_extracted['social_media'].items():
                            parts.append(f"    - {platform.title()}: {url}")
                    if data_extracted.get('streaming_links'):
                        parts.append("  • Streaming Links:")
                        for platform, url in data_extracted['streaming_links'].items():
                            parts.append(f"    - {platform.title()}: {url}")
        
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
- Clear, concise, and to the point — no filler or embellishment
- Formatted using minimal markdown (**bold** for key terms, bullet points for lists, ### headers only when multiple sections are needed)
- Format URLs as markdown links: [Display Text](url)"""
    
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
            highest = search_results["matches"][0]["score"] if search_results["matches"] else 0.0
            categories = [m.get("section_category") for m in search_results["matches"]]
            logger.warning(
                f"[Threshold] Rejected — highest score {highest} < {MIN_SIMILARITY_THRESHOLD}. "
                f"Matches: {len(search_results['matches'])}, Categories: {categories}"
            )
            return {
                "answer": "I don't know based on the available documents.",
                "confidence": "low",
                "reason": "No sufficiently relevant information found (similarity threshold not met)",
                "highest_score": highest,
                "threshold": MIN_SIMILARITY_THRESHOLD,
                "sources": []
            }
        
        # Standard system prompt with strict focus on the question asked
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
- Use conversation history to understand context, pronouns, and references (e.g., "this contract" = the one just discussed)
- If the conversation history contains relevant information that complements the search results, incorporate it into your answer
- Do NOT proactively bring up unrelated previous topics unless the user asks

Your answers should be:
- Accurate and grounded in the provided text
- Clear, concise, and to the point — no filler or embellishment
- Formatted using minimal markdown (**bold** for key terms, bullet points for lists, ### headers only when multiple sections are needed)
- Format URLs as markdown links: [Display Text](url)"""

        # Build messages list with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history for context
        conversation_history = self._get_conversation_context(session_id, max_turns=5)
        if conversation_history:
            # Add a context separator
            messages.append({
                "role": "system", 
                "content": "Previous conversation for context (use for understanding references and incorporating relevant prior information):"
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
        logger.info(f"Context being sent to LLM (first 2000 chars):\n{context[:2000]}")
        
        try:
            response = openai_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_completion_tokens=1000
            )

            raw_answer = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            logger.info(f"LLM finish_reason: {finish_reason}")
            logger.info(f"LLM raw response (first 500 chars): {raw_answer[:500] if raw_answer else 'EMPTY'}")
            if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                logger.warning(f"LLM refusal: {response.choices[0].message.refusal}")

            answer = (raw_answer or "").strip()

            # Retry once without conversation history if LLM returned empty
            if not answer:
                logger.warning("[Retry] LLM returned empty — retrying without conversation history")
                retry_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                retry_response = openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=retry_messages,
                    max_completion_tokens=1000
                )
                raw_answer = retry_response.choices[0].message.content
                retry_finish = retry_response.choices[0].finish_reason
                logger.info(f"[Retry] finish_reason: {retry_finish}")
                logger.info(f"[Retry] response (first 500 chars): {raw_answer[:500] if raw_answer else 'EMPTY'}")
                answer = (raw_answer or "").strip()

            if not answer:
                answer = "I don't know based on the available documents."

            # Server-side extraction of structured data
            extracted_data = self._extract_structured_data(answer, query)

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

            is_fallback = answer == "I don't know based on the available documents."
            return {
                "answer": answer,
                "confidence": "low" if is_fallback else "high",
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
                  context: Optional[Dict] = None,
                  source_preference: Optional[str] = None) -> Dict:
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

        # Summarize conversation history if token limit is approaching
        self._summarize_if_needed(session_id)
        
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
        
        # Check for conversational queries (greetings, thanks, etc.)
        if self._is_conversational_query(query):
            logger.info("Detected conversational query - handling without document search")
            return self._handle_conversational_query(query, session_id)

        # Detect artist intent early — these queries skip Tier 2 so they reach
        # the artist routing path (which offers profile vs conversation choice)
        is_artist_intent = (
            not source_preference
            and artist_data
            and self._is_artist_intent_query(query)
        )

        # Try to answer from conversation history first (skip for artist intent queries)
        skip_artist_disambiguation = False
        if not is_artist_intent and (not source_preference or source_preference == "conversation_history"):
            history_answer = self._try_answer_from_history(query, session_id, context)
            if history_answer:
                logger.info("[History] Answered from conversation history")
                return history_answer
            # If user explicitly chose conversation_history but it couldn't answer,
            # fall through to normal routing but skip artist disambiguation
            if source_preference == "conversation_history":
                # History alone couldn't answer — try structured context before giving up
                can_use_context, reason = self._should_use_context(query, context)
                if can_use_context:
                    logger.info(f"[History] History insufficient but structured context can answer ({reason})")
                    return self._answer_from_context(query, context, session_id)
                source_preference = None
                skip_artist_disambiguation = True

        # Fast path: deterministic artist routing for obvious artist queries
        if not source_preference and artist_data and self._is_artist_intent_query(query):
            logger.info("[FastPath] Artist query detected via keyword matching - skipping LLM routing")
            route_decision = RouteDecision(
                route="artist",
                answer_mode="context",
                confidence=0.9,
                reason="keyword_artist_match"
            )
        else:
            # LLM-based routing: artist/contract/disambiguation + context/retrieve policy
            route_decision = self._llm_route_decision(
                query=query,
                artist_data=artist_data,
                context=context,
                source_preference=source_preference,
                project_id=project_id,
                contract_id=contract_id
            )

        if route_decision.route == "disambiguate":
            logger.info("[Routing] Disambiguation requested")
            return self._build_source_selection_response(query, session_id)

        if route_decision.route == "artist":
            if artist_data:
                # If user explicitly chose artist profile, skip disambiguation
                if source_preference == "artist_profile":
                    logger.info("[Routing] Handling query with artist profile data (explicit choice)")
                    return self._handle_artist_query(query, artist_data, session_id)
                # If user chose conversation history but it was insufficient, don't fall back to artist profile
                if skip_artist_disambiguation:
                    logger.info("[Routing] User chose conversation history but it was insufficient")
                    fallback_msg = "I wasn't able to find anything related to that in our conversation. Could you rephrase your question, or would you like me to look up the artist's profile instead?"
                    self._add_to_memory(session_id, "user", query)
                    self._add_to_memory(session_id, "assistant", fallback_msg)
                    return {
                        "query": query,
                        "answer": fallback_msg,
                        "confidence": "low",
                        "sources": [],
                        "search_results_count": 0,
                        "session_id": session_id,
                        "answered_from": "conversation_history_insufficient",
                        "show_quick_actions": False
                    }
                # If conversation history exists, let user choose between profile and conversation
                history = self._get_conversation_context(session_id, max_turns=1)
                if history:
                    logger.info("[Routing] Artist query with history - offering source choice")
                    return self._build_artist_source_selection_response(query, session_id)
                logger.info("[Routing] Handling query with artist profile data")
                return self._handle_artist_query(query, artist_data, session_id)

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

        if self._should_answer_from_context(route_decision, context, contract_id):
            logger.info("[Routing] Answering from conversation context")
            return self._answer_from_context(query, context, session_id)
        
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
        
        # Step 3: Determine search query from LLM retrieval reason
        search_query = query
        retrieval_reason = route_decision.retrieval_reason or "general_retrieval"
        if retrieval_reason.startswith("missing_"):
            # Generate targeted query for the missing data
            search_query = self._get_targeted_query(retrieval_reason, query)
            logger.info(f"[Targeted Retrieval] Using targeted query for search: '{search_query}'")
            logger.info(f"[Targeted Retrieval] Original query will be used for answer generation: '{query}'")
        
        # Step 4: Query Pinecone for contract data
        search_results = self.search_engine.smart_search(
            query=search_query,  # Use targeted query for search
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
        
        # Step 3: Format context (use a different variable name to avoid shadowing the `context` parameter)
        formatted_context = self._format_context(search_results)
        
        # Step 5: Generate answer with conversation history
        result = self._generate_answer(query, formatted_context, search_results, session_id)
        
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
                           context: Optional[Dict] = None,
                           source_preference: Optional[str] = None) -> Dict:
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

        # Summarize conversation history if token limit is approaching
        self._summarize_if_needed(session_id)

        # Check for conversational queries
        if self._is_conversational_query(query):
            logger.info("Detected conversational query - handling without document search")
            return self._handle_conversational_query(query, session_id)

        # Detect artist intent early — these queries skip Tier 2 so they reach
        # the artist routing path (which offers profile vs conversation choice)
        is_artist_intent = (
            not source_preference
            and artist_data
            and self._is_artist_intent_query(query)
        )

        # Try to answer from conversation history first (skip for artist intent queries)
        skip_artist_disambiguation = False
        if not is_artist_intent and (not source_preference or source_preference == "conversation_history"):
            history_answer = self._try_answer_from_history(query, session_id, context)
            if history_answer:
                logger.info("[History] Answered from conversation history")
                return history_answer
            # If user explicitly chose conversation_history but it couldn't answer,
            # fall through to normal routing but skip artist disambiguation
            if source_preference == "conversation_history":
                # History alone couldn't answer — try structured context before giving up
                can_use_context, reason = self._should_use_context(query, context)
                if can_use_context:
                    logger.info(f"[History] History insufficient but structured context can answer ({reason})")
                    return self._answer_from_context(query, context, session_id)
                source_preference = None
                skip_artist_disambiguation = True

        # Fast path: deterministic artist routing for obvious artist queries
        if not source_preference and artist_data and self._is_artist_intent_query(query):
            logger.info("[FastPath] Artist query detected via keyword matching - skipping LLM routing")
            route_decision = RouteDecision(
                route="artist",
                answer_mode="context",
                confidence=0.9,
                reason="keyword_artist_match"
            )
        else:
            route_decision = self._llm_route_decision(
                query=query,
                artist_data=artist_data,
                context=context,
                source_preference=source_preference,
                project_id=None,
                contract_id=None
            )

        if route_decision.route == "disambiguate":
            logger.info("[Routing] Disambiguation requested")
            return self._build_source_selection_response(query, session_id)

        if route_decision.route == "artist":
            if artist_data:
                # If user explicitly chose artist profile, skip disambiguation
                if source_preference == "artist_profile":
                    logger.info("[Routing] Handling query with artist profile data (explicit choice)")
                    return self._handle_artist_query(query, artist_data, session_id)
                # If user chose conversation history but it was insufficient, don't fall back to artist profile
                if skip_artist_disambiguation:
                    logger.info("[Routing] User chose conversation history but it was insufficient")
                    fallback_msg = "I wasn't able to find anything related to that in our conversation. Could you rephrase your question, or would you like me to look up the artist's profile instead?"
                    self._add_to_memory(session_id, "user", query)
                    self._add_to_memory(session_id, "assistant", fallback_msg)
                    return {
                        "query": query,
                        "answer": fallback_msg,
                        "confidence": "low",
                        "sources": [],
                        "search_results_count": 0,
                        "session_id": session_id,
                        "answered_from": "conversation_history_insufficient",
                        "show_quick_actions": False
                    }
                # If conversation history exists, let user choose between profile and conversation
                history = self._get_conversation_context(session_id, max_turns=1)
                if history:
                    logger.info("[Routing] Artist query with history - offering source choice")
                    return self._build_artist_source_selection_response(query, session_id)
                logger.info("[Routing] Handling query with artist profile data")
                return self._handle_artist_query(query, artist_data, session_id)

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

        if self._should_answer_from_context(route_decision, context, contract_id=None):
            logger.info("[Routing] Answering from conversation context")
            return self._answer_from_context(query, context, session_id)

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
                   context: Optional[Dict] = None,
                   source_preference: Optional[str] = None) -> Dict:
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
            context=context,
            source_preference=source_preference
        )
    
    def ask_multiple_contracts(self,
                              query: str,
                              user_id: str,
                              project_id: str,
                              contract_ids: List[str],
                              top_k: int = DEFAULT_TOP_K,
                              session_id: Optional[str] = None,
                              artist_data: Optional[Dict] = None,
                              context: Optional[Dict] = None,
                              source_preference: Optional[str] = None) -> Dict:
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

        # Summarize conversation history if token limit is approaching
        self._summarize_if_needed(session_id)

        # Check for conversational queries (greetings, thanks, etc.)
        if self._is_conversational_query(query):
            logger.info("Detected conversational query - handling without document search")
            return self._handle_conversational_query(query, session_id)

        # Detect artist intent early — these queries skip Tier 2 so they reach
        # the artist routing path (which offers profile vs conversation choice)
        is_artist_intent = (
            not source_preference
            and artist_data
            and self._is_artist_intent_query(query)
        )

        # Try to answer from conversation history first (skip for artist intent queries)
        skip_artist_disambiguation = False
        if not is_artist_intent and (not source_preference or source_preference == "conversation_history"):
            history_answer = self._try_answer_from_history(query, session_id, context)
            if history_answer:
                logger.info("[History] Answered from conversation history")
                return history_answer
            # If user explicitly chose conversation_history but it couldn't answer,
            # fall through to normal routing but skip artist disambiguation
            if source_preference == "conversation_history":
                # History alone couldn't answer — try structured context before giving up
                can_use_context, reason = self._should_use_context(query, context)
                if can_use_context:
                    logger.info(f"[History] History insufficient but structured context can answer ({reason})")
                    return self._answer_from_context(query, context, session_id)
                source_preference = None
                skip_artist_disambiguation = True

        # Fast path: deterministic artist routing for obvious artist queries
        if not source_preference and artist_data and self._is_artist_intent_query(query):
            logger.info("[FastPath] Artist query detected via keyword matching - skipping LLM routing")
            route_decision = RouteDecision(
                route="artist",
                answer_mode="context",
                confidence=0.9,
                reason="keyword_artist_match"
            )
        else:
            route_decision = self._llm_route_decision(
                query=query,
                artist_data=artist_data,
                context=context,
                source_preference=source_preference,
                project_id=project_id,
                contract_id=contract_ids[0] if contract_ids else None
            )

        if route_decision.route == "disambiguate":
            logger.info("[Routing] Disambiguation requested")
            return self._build_source_selection_response(query, session_id)

        if route_decision.route == "artist":
            if artist_data:
                # If user explicitly chose artist profile, skip disambiguation
                if source_preference == "artist_profile":
                    logger.info("[Routing] Handling query with artist profile data (explicit choice)")
                    return self._handle_artist_query(query, artist_data, session_id)
                # If user chose conversation history but it was insufficient, don't fall back to artist profile
                if skip_artist_disambiguation:
                    logger.info("[Routing] User chose conversation history but it was insufficient")
                    fallback_msg = "I wasn't able to find anything related to that in our conversation. Could you rephrase your question, or would you like me to look up the artist's profile instead?"
                    self._add_to_memory(session_id, "user", query)
                    self._add_to_memory(session_id, "assistant", fallback_msg)
                    return {
                        "query": query,
                        "answer": fallback_msg,
                        "confidence": "low",
                        "sources": [],
                        "search_results_count": 0,
                        "session_id": session_id,
                        "answered_from": "conversation_history_insufficient",
                        "show_quick_actions": False
                    }
                # If conversation history exists, let user choose between profile and conversation
                history = self._get_conversation_context(session_id, max_turns=1)
                if history:
                    logger.info("[Routing] Artist query with history - offering source choice")
                    return self._build_artist_source_selection_response(query, session_id)
                logger.info("[Routing] Handling query with artist profile data")
                return self._handle_artist_query(query, artist_data, session_id)

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

        if self._should_answer_from_context(route_decision, context, contract_ids[0] if contract_ids else None):
            logger.info("[Routing] Answering from conversation context")
            return self._answer_from_context(query, context, session_id)
        
        # Store user message in memory
        self._add_to_memory(session_id, "user", query)
        
        # Step 3: Determine search query from LLM retrieval reason
        search_query = query
        retrieval_reason = route_decision.retrieval_reason or "general_retrieval"
        if retrieval_reason.startswith("missing_"):
            # Generate targeted query for the missing data
            search_query = self._get_targeted_query(retrieval_reason, query)
            logger.info(f"[Targeted Retrieval] Using targeted query for search: '{search_query}'")
            logger.info(f"[Targeted Retrieval] Original query will be used for answer generation: '{query}'")
        
        # Step 4: Query Pinecone for contract data
        search_results = self.search_engine.search_multiple_contracts(
            query=search_query,  # Use targeted query for search
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
        
        # Format context (use a different variable name to avoid shadowing the `context` parameter)
        formatted_context = self._format_context(search_results)
        
        # Generate answer with conversation history
        result = self._generate_answer(query, formatted_context, search_results, session_id)
        
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
    
    # ──────────────────────────────────────────────────────────────
    # SSE Streaming Methods
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sse_event(event_type: str, data: Dict) -> str:
        """Format a Server-Sent Events string."""
        payload = {"type": event_type, **data}
        return f"data: {json.dumps(payload)}\n\n"

    def _stream_llm_completion(self, messages: List[Dict], max_tokens: int = 1000):
        """
        Generator that yields token strings from an OpenAI streaming call.
        """
        response = openai_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            max_completion_tokens=max_tokens,
            stream=True
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _generate_answer_stream(self, query: str, context: str,
                                 search_results: Dict,
                                 session_id: Optional[str] = None):
        """
        Generator version of _generate_answer. Yields SSE event strings.
        Streams LLM tokens in real-time, then yields post-processed metadata.
        """
        # Check similarity threshold
        if not self._check_similarity_threshold(search_results):
            highest = search_results["matches"][0]["score"] if search_results["matches"] else 0.0
            yield self._sse_event("complete", {
                "answer": "I don't know based on the available documents.",
                "confidence": "low",
                "sources": [],
                "highest_score": highest
            })
            return

        # Build sources from search results (known before LLM call)
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
        highest_score = search_results["matches"][0]["score"]

        # Yield sources before streaming starts
        yield self._sse_event("sources", {
            "sources": sources,
            "highest_score": highest_score,
            "search_results_count": search_results["total_results"]
        })

        # Build prompt (same as _generate_answer)
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
- Use conversation history to understand context, pronouns, and references (e.g., "this contract" = the one just discussed)
- If the conversation history contains relevant information that complements the search results, incorporate it into your answer
- Do NOT proactively bring up unrelated previous topics unless the user asks

Your answers should be:
- Accurate and grounded in the provided text
- Clear, concise, and to the point — no filler or embellishment
- Formatted using minimal markdown (**bold** for key terms, bullet points for lists, ### headers only when multiple sections are needed)
- Format URLs as markdown links: [Display Text](url)"""

        messages = [{"role": "system", "content": system_prompt}]

        conversation_history = self._get_conversation_context(session_id, max_turns=5)
        if conversation_history:
            messages.append({
                "role": "system",
                "content": "Previous conversation for context (use for understanding references and incorporating relevant prior information):"
            })
            messages.extend(conversation_history)

        user_prompt = f"""Based on the following contract documents, answer this question:

{query}

Contract documents:
{context}

Remember: Answer ONLY what was asked. Do not suggest follow-up questions."""

        messages.append({"role": "user", "content": user_prompt})

        logger.info(f"\n[Stream] Generating answer using {self.llm_model}...")

        try:
            # Stream LLM tokens
            full_answer = ""
            for token in self._stream_llm_completion(messages, 1000):
                full_answer += token
                yield self._sse_event("token", {"content": token})

            answer = full_answer.strip()

            # Retry once without history if empty
            if not answer:
                logger.warning("[Stream][Retry] LLM returned empty — retrying without history")
                retry_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                full_answer = ""
                for token in self._stream_llm_completion(retry_messages, 1000):
                    full_answer += token
                    yield self._sse_event("token", {"content": token})
                answer = full_answer.strip()

            if not answer:
                answer = "I don't know based on the available documents."
                yield self._sse_event("token", {"content": answer})

            # Post-processing (after full answer is available)
            extracted_data = self._extract_structured_data(answer, query)
            is_fallback = answer == "I don't know based on the available documents."

            yield self._sse_event("data", {
                "extracted_data": extracted_data if extracted_data else None,
                "confidence": "low" if is_fallback else "high",
                "highest_score": highest_score,
                "model": self.llm_model
            })

            # Store in memory
            self._add_to_memory(session_id, "assistant", answer, {
                "confidence": "low" if is_fallback else "high",
                "sources": sources
            })

            yield self._sse_event("done", {"answered_from": "document"})

        except Exception as e:
            logger.error(f"[Stream] Error generating answer: {e}")
            yield self._sse_event("error", {"message": str(e)})

    def _handle_artist_query_stream(self, query: str, artist_data: Dict,
                                     session_id: Optional[str] = None):
        """
        Generator version of _handle_artist_query. Yields SSE event strings.
        """
        self._add_to_memory(session_id, "user", query)
        artist_context = self._format_artist_context(artist_data)

        system_prompt = """You are a helpful assistant providing concise, direct information about an artist.
Answer the user's question based on the artist profile information provided.

CRITICAL RULES:
1. Answer ONLY with the exact information from the provided artist data
2. Be direct and concise - do NOT add extra commentary, interpretation, or embellishment
3. Do NOT infer or add information not explicitly stated in the artist data
4. If asked for a specific piece of information (like bio), provide ONLY that information
5. Keep responses brief - typically 1-2 sentences unless the data itself is longer
6. If information is not available, simply say so

FORMATTING:
- Use markdown for structure, not decoration — keep it minimal
- Use **bold** for field labels (e.g., **Bio:**, **Genres:**)
- Use bullet points for listing multiple items (socials, links, genres)
- Format URLs as markdown links: [Platform Name](url)
- Only use ### headers when answering about multiple fields — skip headers for single-field answers
- Do NOT add filler phrases, introductions, or conclusions — just the data"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Based on the following artist profile, answer this question:

{query}

Artist Profile:
{artist_context}"""}
        ]

        try:
            full_answer = ""
            for token in self._stream_llm_completion(messages, 500):
                full_answer += token
                yield self._sse_event("token", {"content": token})

            answer = full_answer.strip()
            if not answer:
                answer = self._build_artist_fallback_answer(artist_data, query)
                yield self._sse_event("token", {"content": answer})

        except Exception as e:
            logger.error(f"[Stream] Error generating artist answer: {e}")
            answer = self._build_artist_fallback_answer(artist_data, query)
            yield self._sse_event("token", {"content": answer})

        self._add_to_memory(session_id, "assistant", answer)
        extracted_artist_data = self._extract_artist_data(artist_data)

        yield self._sse_event("data", {
            "extracted_data": extracted_artist_data if extracted_artist_data else None,
            "confidence": "artist_data",
            "answered_from": "artist_data"
        })
        yield self._sse_event("done", {"answered_from": "artist_data"})

    def _answer_from_context_stream(self, query: str, context: Dict,
                                     session_id: Optional[str] = None):
        """
        Generator version of _answer_from_context. Yields SSE event strings.
        """
        logger.info(f"[Stream][Context] _answer_from_context_stream called with query: {query}")
        context_summary = self._format_context_for_llm(context)

        aggregated_splits_block = ""
        royalty_rule_prompt = ""
        if self._is_multi_contract_split_query(query, context):
            aggregation = self._aggregate_splits_across_contracts(context, query)
            if aggregation['parties']:
                aggregated_splits_block = self._format_aggregated_splits_for_llm(aggregation)
                context_summary = f"{aggregated_splits_block}\n\n{context_summary}"
                royalty_rule_prompt = """

CRITICAL ROYALTY SPLIT RULE:
The PRE-COMPUTED AGGREGATED ROYALTY SPLITS above are ALREADY CORRECT. These splits represent portions of the SAME 100% pie — they are NOT summed per person across contracts.
- If a person appears in multiple bilateral contracts (e.g., Bob appears in Contract 1 and Contract 2 both at 40%), their share is STILL 40% — NOT 80%.
- Each contract describes a bilateral agreement between two parties, but all contracts refer to the same underlying revenue pool.
- Use ONLY the pre-computed numbers above. Do NOT re-calculate or sum percentages from individual contracts.
- If a payout calculation is included above, present those exact numbers."""

        history = self._get_conversation_context(session_id, max_turns=10)

        available_data = self._get_available_data_types(context)
        suggestion_guidance = self._build_suggestion_guidance(available_data)

        system_prompt = f"""You are answering a follow-up question using information from the conversation context.

The user is asking about information that was previously discussed in this conversation.
Use the structured context and conversation history to provide a comprehensive answer.

CRITICAL: You can compare, summarize, or analyze the information provided without needing to search documents again.
{royalty_rule_prompt}

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
3. If comparing artists, always provide a comparison answer using available context
4. If artist context is sparse, explicitly state there is not enough information for a full comparison and list what is missing
5. Be specific about which contract each piece of information came from
6. If information is incomplete, acknowledge what you don't have
7. Format responses using clean, minimal markdown:
   - Use ### headers to separate each artist or contract being compared
   - Use **bold** for field labels, bullet points for attributes
   - Format URLs as markdown links: [Platform Name](url)
   - For comparisons, use headers per entity then a short ### Answer section for the user's specific question
   - Do NOT pad with filler text, introductions, or unnecessary transitions — just structured data and a direct answer
8. Only suggest follow-ups that can be answered from existing context data"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": f"Context:\n{context_summary}\n\nQuestion: {query}"})

        try:
            full_answer = ""
            for token in self._stream_llm_completion(messages, 1000):
                full_answer += token
                yield self._sse_event("token", {"content": token})

            answer = full_answer.strip()
            if not answer:
                answer = "I couldn't generate a response. Please try rephrasing your question."
                yield self._sse_event("token", {"content": answer})

        except Exception as e:
            logger.error(f"[Stream] Error generating context-based answer: {e}")
            answer = "I encountered an error while processing your question. Please try again."
            yield self._sse_event("token", {"content": answer})

        # Extract suggestion from answer (same as non-streaming)
        _clean_answer, suggestion = self._extract_suggestion_from_answer(answer)

        pending_suggestion = None
        if suggestion and session_id:
            context_hash = self._compute_context_hash(context)
            action_match = re.search(r'(?:Would you like (?:me to|to)?|Shall I|Should I)\s*(.+?)\??$', suggestion, re.IGNORECASE)
            if action_match:
                pending_suggestion = action_match.group(1).strip()
                self.memory.set_pending_suggestion(session_id, pending_suggestion, context_hash)

        self._add_to_memory(session_id, "user", query)
        self._add_to_memory(session_id, "assistant", answer)

        yield self._sse_event("data", {
            "confidence": "context",
            "answered_from": "context",
            "pending_suggestion": pending_suggestion
        })
        yield self._sse_event("done", {"answered_from": "context"})

    def ask_stream(self,
                   query: str,
                   user_id: str,
                   project_id: Optional[str] = None,
                   contract_ids: Optional[List[str]] = None,
                   top_k: Optional[int] = None,
                   session_id: Optional[str] = None,
                   artist_data: Optional[Dict] = None,
                   context: Optional[Dict] = None,
                   source_preference: Optional[str] = None):
        """
        Unified streaming entry point for Zoe chatbot.
        Generator that yields SSE event strings.

        Handles all three cases:
        - No project (artist-only queries)
        - Project-wide queries
        - Multi-contract queries

        For instant responses (conversational, disambiguation, etc.), yields a
        single "complete" event. For LLM-generated answers, streams tokens.
        """
        logger.info("\n" + "=" * 80)
        logger.info("STREAMING SMART CONTRACT CHATBOT")
        logger.info("=" * 80)
        logger.info(f"Question: {query}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Session ID: {session_id}")

        yield self._sse_event("start", {"session_id": session_id})

        # Summarize conversation history if token limit is approaching
        self._summarize_if_needed(session_id)

        # ── Affirmative response to pending suggestion ──
        if session_id and self._is_affirmative_response(query):
            pending = self.memory.get_pending_suggestion(session_id)
            if pending:
                current_hash = self._compute_context_hash(context)
                if pending.get('context_hash') != current_hash:
                    self.memory.clear_pending_suggestion(session_id)
                    yield self._sse_event("complete", {
                        "answer": "It looks like the conversation context was reset. Please refresh the page to start a new session, or ask your question again.",
                        "confidence": "context_cleared",
                        "sources": [],
                        "context_cleared": True,
                        "session_id": session_id
                    })
                    return

                suggestion = pending.get('suggestion', '')
                self.memory.clear_pending_suggestion(session_id)
                yield from self._answer_from_context_stream(suggestion, context, session_id)
                return

        # ── Tier 1: Conversational fast path ──
        if self._is_conversational_query(query):
            logger.info("[Stream] Detected conversational query")
            result = self._handle_conversational_query(query, session_id)
            yield self._sse_event("complete", result)
            return

        # ── Detect artist intent early (before Tier 2) ──
        is_artist_intent = (
            not source_preference
            and artist_data
            and self._is_artist_intent_query(query)
        )

        # ── Tier 2: Try conversation history (skip for artist intent queries) ──
        skip_artist_disambiguation = False
        if not is_artist_intent and (not source_preference or source_preference == "conversation_history"):
            history_answer = self._try_answer_from_history(query, session_id, context)
            if history_answer:
                logger.info("[Stream][History] Answered from conversation history")
                yield self._sse_event("complete", history_answer)
                return
            if source_preference == "conversation_history":
                can_use_context, _reason = self._should_use_context(query, context)
                if can_use_context:
                    yield from self._answer_from_context_stream(query, context, session_id)
                    return
                source_preference = None
                skip_artist_disambiguation = True

        # ── Routing decision ──
        contract_id = contract_ids[0] if contract_ids and len(contract_ids) == 1 else None
        if not source_preference and artist_data and self._is_artist_intent_query(query):
            route_decision = RouteDecision(
                route="artist", answer_mode="context",
                confidence=0.9, reason="keyword_artist_match"
            )
        else:
            route_decision = self._llm_route_decision(
                query=query, artist_data=artist_data, context=context,
                source_preference=source_preference,
                project_id=project_id,
                contract_id=contract_id
            )

        # ── Disambiguation ──
        if route_decision.route == "disambiguate":
            yield self._sse_event("complete", self._build_source_selection_response(query, session_id))
            return

        # ── Artist route ──
        if route_decision.route == "artist":
            if artist_data:
                if source_preference == "artist_profile":
                    yield from self._handle_artist_query_stream(query, artist_data, session_id)
                    return
                if skip_artist_disambiguation:
                    fallback_msg = "I wasn't able to find anything related to that in our conversation. Could you rephrase your question, or would you like me to look up the artist's profile instead?"
                    self._add_to_memory(session_id, "user", query)
                    self._add_to_memory(session_id, "assistant", fallback_msg)
                    yield self._sse_event("complete", {
                        "query": query, "answer": fallback_msg, "confidence": "low",
                        "sources": [], "search_results_count": 0, "session_id": session_id,
                        "answered_from": "conversation_history_insufficient",
                        "show_quick_actions": False
                    })
                    return
                history = self._get_conversation_context(session_id, max_turns=1)
                if history:
                    yield self._sse_event("complete", self._build_artist_source_selection_response(query, session_id))
                    return
                yield from self._handle_artist_query_stream(query, artist_data, session_id)
                return

            self._add_to_memory(session_id, "user", query)
            response = "I don't have artist information available. Please select an artist from the sidebar."
            self._add_to_memory(session_id, "assistant", response)
            yield self._sse_event("complete", {
                "query": query, "answer": response, "confidence": "needs_artist",
                "sources": [], "search_results_count": 0, "session_id": session_id,
                "show_quick_actions": True
            })
            return

        # ── Context-based answer ──
        if self._should_answer_from_context(route_decision, context, contract_id):
            yield from self._answer_from_context_stream(query, context, session_id)
            return

        # ── Need a project for contract search ──
        if not project_id:
            self._add_to_memory(session_id, "user", query)
            response = "To answer questions about contracts, I need you to select a project first. Please choose a project from the sidebar."
            self._add_to_memory(session_id, "assistant", response)
            yield self._sse_event("complete", {
                "query": query, "answer": response, "confidence": "needs_project",
                "sources": [], "search_results_count": 0, "session_id": session_id,
                "show_quick_actions": True
            })
            return

        # ── Tier 3: Vector search + streaming LLM generation ──
        self._add_to_memory(session_id, "user", query)

        search_query = query
        retrieval_reason = route_decision.retrieval_reason or "general_retrieval"
        if retrieval_reason.startswith("missing_"):
            search_query = self._get_targeted_query(retrieval_reason, query)

        # Perform search based on contract selection
        if contract_ids and len(contract_ids) > 0:
            search_results = self.search_engine.search_multiple_contracts(
                query=search_query, user_id=user_id,
                project_id=project_id, contract_ids=contract_ids,
                top_k=top_k or DEFAULT_TOP_K
            )
        else:
            search_results = self.search_engine.smart_search(
                query=search_query, user_id=user_id,
                project_id=project_id, contract_id=contract_id,
                top_k=top_k
            )

        if not search_results["matches"]:
            no_result_answer = "I don't know based on the available documents."
            self._add_to_memory(session_id, "assistant", no_result_answer)
            yield self._sse_event("complete", {
                "query": query, "answer": no_result_answer, "confidence": "low",
                "sources": [], "search_results_count": 0, "session_id": session_id
            })
            return

        formatted_context = self._format_context(search_results)

        # Stream the answer generation
        yield from self._generate_answer_stream(query, formatted_context, search_results, session_id)

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
