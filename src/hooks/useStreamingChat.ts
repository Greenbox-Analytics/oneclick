import { useState, useRef, useCallback } from "react";

// ── Types ──

export type SourcePreference =
  | "artist_profile"
  | "contract_context"
  | "conversation_history";

export interface AssistantQuickAction {
  id: string;
  label: string;
  query?: string;
  source_preference?: SourcePreference;
}

export interface MessageSource {
  contract_file: string;
  page_number?: number;
  score: number;
  project_name?: string;
  section_heading?: string;
  section_category?: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  confidence?: string;
  sources?: MessageSource[];
  timestamp: string;
  showQuickActions?: boolean;
  quickActions?: AssistantQuickAction[];
  answeredFrom?: string;
  extractedData?: Record<string, unknown> | null;
  isStreaming?: boolean;
}

interface SendMessageOptions {
  sourcePreference?: SourcePreference;
  /** Display text shown in the user bubble (defaults to query) */
  userDisplayMessage?: string;
  /** If true, suppress the user message bubble (for silent source selection) */
  silent?: boolean;
}

interface ConversationContext {
  session_id: string;
  artist: { id: string; name: string } | null;
  artists_discussed: unknown[];
  project: { id: string; name: string } | null;
  contracts_discussed: unknown[];
  context_switches: unknown[];
}

// SSE event payloads
interface SSEStart {
  type: "start";
  session_id: string;
}
interface SSEToken {
  type: "token";
  content: string;
}
interface SSESources {
  type: "sources";
  sources: MessageSource[];
  highest_score?: number;
  search_results_count?: number;
}
interface SSEData {
  type: "data";
  extracted_data?: Record<string, unknown> | null;
  confidence?: string;
  answered_from?: string;
  highest_score?: number;
  pending_suggestion?: string | null;
}
interface SSEDone {
  type: "done";
  answered_from?: string;
}
interface SSEComplete {
  type: "complete";
  answer: string;
  confidence?: string;
  sources?: MessageSource[];
  session_id?: string;
  show_quick_actions?: boolean;
  quick_actions?: AssistantQuickAction[];
  answered_from?: string;
  extracted_data?: Record<string, unknown> | null;
  context_cleared?: boolean;
  needs_source_selection?: boolean;
  [key: string]: unknown;
}
interface SSEError {
  type: "error";
  message: string;
}

type SSEEvent =
  | SSEStart
  | SSEToken
  | SSESources
  | SSEData
  | SSEDone
  | SSEComplete
  | SSEError;

// ── Hook ──

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";
const MAX_CONVERSATION_MESSAGES = 100;

export function useStreamingChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState("");
  const abortControllerRef = useRef<AbortController | null>(null);

  // ── helpers ──

  const makeId = () =>
    `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

  /** Parse a raw SSE text buffer into events. Returns leftover partial text. */
  const parseSSEBuffer = (buffer: string): [SSEEvent[], string] => {
    const events: SSEEvent[] = [];
    const parts = buffer.split("\n\n");
    // last part may be incomplete
    const leftover = parts.pop() ?? "";

    for (const part of parts) {
      const trimmed = part.trim();
      if (!trimmed) continue;
      for (const line of trimmed.split("\n")) {
        if (line.startsWith("data: ")) {
          try {
            events.push(JSON.parse(line.slice(6)));
          } catch {
            // skip malformed JSON
          }
        }
      }
    }
    return [events, leftover];
  };

  // ── main send function ──

  const sendMessage = useCallback(
    async (
      query: string,
      params: {
        userId: string;
        artistId: string;
        projectId?: string;
        contractIds?: string[];
        sessionId: string;
        context: ConversationContext;
      },
      options: SendMessageOptions = {}
    ): Promise<{
      sessionId: string;
      contextCleared?: boolean;
      extractedData?: Record<string, unknown> | null;
      answeredFrom?: string;
      sources?: MessageSource[];
    }> => {
      const {
        sourcePreference,
        userDisplayMessage,
        silent = false,
      } = options;

      setError("");

      // Add user message bubble (unless silent)
      const userMsg: Message = {
        id: makeId(),
        role: "user",
        content: userDisplayMessage || query,
        timestamp: new Date().toISOString(),
      };
      if (!silent) {
        setMessages((prev) => [...prev, userMsg]);
      }

      // Create empty assistant message placeholder
      const assistantId = makeId();
      const assistantMsg: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        isStreaming: true,
      };
      setMessages((prev) => [...prev, assistantMsg]);

      setIsStreaming(true);

      const controller = new AbortController();
      abortControllerRef.current = controller;

      let returnSessionId = params.sessionId;
      let contextCleared = false;
      let extractedData: Record<string, unknown> | null | undefined;
      let answeredFrom: string | undefined;
      let returnSources: MessageSource[] | undefined;

      try {
        const response = await fetch(`${API_URL}/zoe/ask-stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            query,
            project_id: params.projectId || null,
            contract_ids:
              params.contractIds && params.contractIds.length > 0
                ? params.contractIds
                : null,
            user_id: params.userId,
            session_id: params.sessionId,
            artist_id: params.artistId,
            context: params.context,
            source_preference: sourcePreference || null,
          }),
        });

        if (!response.ok) {
          throw new Error("Failed to get response from Zoe");
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let sseBuffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          sseBuffer += decoder.decode(value, { stream: true });
          const [events, leftover] = parseSSEBuffer(sseBuffer);
          sseBuffer = leftover;

          for (const event of events) {
            switch (event.type) {
              case "start":
                if (event.session_id) returnSessionId = event.session_id;
                break;

              case "token":
                // Append token to the assistant message in-place
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, content: m.content + event.content }
                      : m
                  )
                );
                break;

              case "sources":
                returnSources = event.sources;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, sources: event.sources }
                      : m
                  )
                );
                break;

              case "data":
                extractedData = event.extracted_data;
                answeredFrom = event.answered_from;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? {
                          ...m,
                          confidence: event.confidence ?? m.confidence,
                          answeredFrom: event.answered_from,
                          extractedData: event.extracted_data,
                        }
                      : m
                  )
                );
                break;

              case "done":
                answeredFrom = event.answered_from ?? answeredFrom;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, isStreaming: false, answeredFrom: event.answered_from ?? m.answeredFrom }
                      : m
                  )
                );
                break;

              case "complete": {
                // Non-streamed instant response (tiers 1/2)
                if (event.context_cleared) {
                  contextCleared = true;
                }
                if (event.session_id) returnSessionId = event.session_id;
                extractedData = event.extracted_data;
                answeredFrom = event.answered_from;
                returnSources = event.sources;

                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? {
                          ...m,
                          content: event.answer,
                          confidence: event.confidence,
                          sources: event.sources,
                          showQuickActions: event.show_quick_actions,
                          quickActions: event.quick_actions,
                          answeredFrom: event.answered_from,
                          extractedData: event.extracted_data,
                          isStreaming: false,
                        }
                      : m
                  )
                );
                break;
              }

              case "error":
                setError(event.message);
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? {
                          ...m,
                          content:
                            m.content ||
                            "I'm sorry, I encountered an error. Please try again.",
                          confidence: "error",
                          isStreaming: false,
                        }
                      : m
                  )
                );
                break;
            }
          }
        }

        // Ensure streaming flag is cleared even if "done" event was missed
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, isStreaming: false } : m
          )
        );
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") {
          // User clicked stop — keep the partial answer, just mark as not streaming
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, isStreaming: false } : m
            )
          );
        } else {
          console.error("Error sending message:", err);
          const errorText =
            err instanceof Error ? err.message : "An unexpected error occurred";
          setError(errorText);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    content:
                      m.content ||
                      "I'm sorry, I encountered an error. Please try again.",
                    confidence: "error",
                    isStreaming: false,
                  }
                : m
            )
          );
        }
      } finally {
        setIsStreaming(false);
        abortControllerRef.current = null;
      }

      return {
        sessionId: returnSessionId,
        contextCleared,
        extractedData,
        answeredFrom,
        sources: returnSources,
      };
    },
    []
  );

  // ── stop / retry / clear ──

  const stopGeneration = useCallback(() => {
    abortControllerRef.current?.abort();
  }, []);

  const retryLastMessage = useCallback(
    async (
      params: {
        userId: string;
        artistId: string;
        projectId?: string;
        contractIds?: string[];
        sessionId: string;
        context: ConversationContext;
      }
    ) => {
      // Find the last user message
      const lastUserIndex = [...messages]
        .reverse()
        .findIndex((m) => m.role === "user");
      if (lastUserIndex === -1) return;
      const actualIndex = messages.length - 1 - lastUserIndex;
      const lastUserMsg = messages[actualIndex];

      // Remove last assistant message(s) after it
      setMessages((prev) => prev.slice(0, actualIndex + 1));

      // Re-send
      return sendMessage(lastUserMsg.content, params, { silent: true });
    },
    [messages, sendMessage]
  );

  const addSystemMessage = useCallback((content: string) => {
    const msg: Message = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      role: "system",
      content,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, msg]);
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError("");
  }, []);

  const isAtLimit = messages.length >= MAX_CONVERSATION_MESSAGES;

  return {
    messages,
    setMessages,
    isStreaming,
    error,
    setError,
    sendMessage,
    stopGeneration,
    retryLastMessage,
    addSystemMessage,
    clearMessages,
    isAtLimit,
    MAX_CONVERSATION_MESSAGES,
  };
}
