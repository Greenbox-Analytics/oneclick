import { useEffect, useCallback, useRef } from "react";
import type { Message } from "@/hooks/useStreamingChat";

// ── Types ──

interface PersistedSession {
  sessionId: string;
  messages: Message[];
  selectedArtist: string;
  selectedProject: string;
  selectedContracts: string[];
  conversationContext: unknown;
  savedAt: number;
}

interface PersistenceOptions {
  sessionId: string;
  messages: Message[];
  selectedArtist: string;
  selectedProject: string;
  selectedContracts: string[];
  conversationContext: unknown;
}

interface PersistenceActions {
  /** Save current state to localStorage (called automatically on changes) */
  save: () => void;
  /** Clear the current session from localStorage */
  clearSession: () => void;
}

// ── Constants ──

const STORAGE_KEY_PREFIX = "zoe-session-";
const SESSION_INDEX_KEY = "zoe-sessions";
const MAX_STORED_SESSIONS = 5;
const SAVE_DEBOUNCE_MS = 500;

// ── Helpers ──

function getSessionKey(sessionId: string) {
  return `${STORAGE_KEY_PREFIX}${sessionId}`;
}

function getSessionIndex(): string[] {
  try {
    const raw = localStorage.getItem(SESSION_INDEX_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function setSessionIndex(ids: string[]) {
  localStorage.setItem(SESSION_INDEX_KEY, JSON.stringify(ids));
}

/** Evict oldest sessions if we exceed the cap */
function pruneOldSessions(currentSessionId: string) {
  const index = getSessionIndex();
  // Move current session to front
  const updated = [currentSessionId, ...index.filter((id) => id !== currentSessionId)];
  // Remove excess sessions
  while (updated.length > MAX_STORED_SESSIONS) {
    const evicted = updated.pop()!;
    localStorage.removeItem(getSessionKey(evicted));
  }
  setSessionIndex(updated);
}

// ── Hook ──

/**
 * Persists Zoe conversation state to localStorage.
 *
 * - Automatically saves on every state change (debounced).
 * - Provides `restore()` to load a previous session.
 * - Caps stored sessions to 5.
 */
export function useConversationPersistence(
  options: PersistenceOptions
): PersistenceActions {
  const { sessionId, messages, selectedArtist, selectedProject, selectedContracts, conversationContext } = options;
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Save current state
  const save = useCallback(() => {
    if (!sessionId || messages.length === 0) return;

    // Don't persist messages that are still streaming
    const safeMessages = messages.map((m) =>
      m.isStreaming ? { ...m, isStreaming: false } : m
    );

    const data: PersistedSession = {
      sessionId,
      messages: safeMessages,
      selectedArtist,
      selectedProject,
      selectedContracts,
      conversationContext,
      savedAt: Date.now(),
    };

    try {
      localStorage.setItem(getSessionKey(sessionId), JSON.stringify(data));
      pruneOldSessions(sessionId);
    } catch (e) {
      // localStorage might be full — evict oldest and retry once
      console.warn("Failed to save session, evicting oldest:", e);
      const index = getSessionIndex();
      if (index.length > 1) {
        const oldest = index[index.length - 1];
        localStorage.removeItem(getSessionKey(oldest));
        setSessionIndex(index.slice(0, -1));
        try {
          localStorage.setItem(getSessionKey(sessionId), JSON.stringify(data));
        } catch {
          // Give up
        }
      }
    }
  }, [sessionId, messages, selectedArtist, selectedProject, selectedContracts, conversationContext]);

  // Auto-save on state changes (debounced)
  useEffect(() => {
    if (messages.length === 0) return;

    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(save, SAVE_DEBOUNCE_MS);

    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    };
  }, [save, messages]);

  const clearSession = useCallback(() => {
    localStorage.removeItem(getSessionKey(sessionId));
    const index = getSessionIndex();
    setSessionIndex(index.filter((id) => id !== sessionId));
  }, [sessionId]);

  return { save, clearSession };
}

// ── Static restore function (called before hook, e.g. in initializer) ──

/**
 * Attempt to restore the most recent saved session.
 * Returns null if nothing is found.
 */
export function restoreLatestSession(): PersistedSession | null {
  const index = getSessionIndex();
  if (index.length === 0) return null;

  // Try most recent first
  for (const id of index) {
    try {
      const raw = localStorage.getItem(getSessionKey(id));
      if (raw) {
        const data: PersistedSession = JSON.parse(raw);
        // Ignore sessions older than 2 hours
        if (Date.now() - data.savedAt > 2 * 60 * 60 * 1000) {
          localStorage.removeItem(getSessionKey(id));
          continue;
        }
        return data;
      }
    } catch {
      // Corrupted entry — remove it
      localStorage.removeItem(getSessionKey(id));
    }
  }

  return null;
}
