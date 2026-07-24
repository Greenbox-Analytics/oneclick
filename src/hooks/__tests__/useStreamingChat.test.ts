import { describe, it, expect, vi, afterEach } from "vitest";
import { act, renderHook, cleanup } from "@testing-library/react";

// useStreamingChat talks to the backend via raw fetch (it needs a streaming
// reader, so it can't use apiFetch) but still pulls auth headers from
// @/lib/apiFetch — stub that so the module loads without touching the real
// Supabase client, matching the pattern in useBoards.optimistic.test.ts.
vi.mock("@/lib/apiFetch", () => ({
  API_URL: "http://test",
  getAuthHeaders: vi.fn().mockResolvedValue({}),
  // Mirror the real apiErrorFromBody (see src/lib/apiFetch.ts) so the hook's
  // pre-stream error path derives the same message + attaches raw `detail`,
  // without pulling in the real module (which imports the Supabase client).
  apiErrorFromBody: (
    body: { detail?: unknown } | null | undefined,
    status: number,
    fallback = `Request failed: ${status}`,
  ) => {
    const detail = body?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : (detail as { reason?: string } | undefined)?.reason ?? fallback;
    const err = new Error(message) as Error & { status?: number; detail?: unknown };
    err.status = status;
    err.detail = detail;
    return err;
  },
}));

import { useStreamingChat } from "@/hooks/useStreamingChat";

const baseParams = {
  artistId: "a1",
  sessionId: "s1",
  context: {
    session_id: "s1",
    artist: null,
    artists_discussed: [],
    project: null,
    contracts_discussed: [],
    context_switches: [],
  },
};

/** Builds a fetch Response stand-in whose body streams the given SSE chunks
 * (each already in `data: {...}\n\n` form) then ends. */
function makeStreamResponse(chunks: string[]) {
  let i = 0;
  const encoder = new TextEncoder();
  return {
    ok: true,
    status: 200,
    json: async () => ({}),
    body: {
      getReader: () => ({
        read: async () => {
          if (i < chunks.length) {
            const value = encoder.encode(chunks[i]);
            i += 1;
            return { done: false, value };
          }
          return { done: true, value: undefined };
        },
      }),
    },
  };
}

function makeErrorResponse(status: number, body: unknown) {
  return {
    ok: false,
    status,
    json: async () => body,
  };
}

const lastAssistantMessage = (messages: ReturnType<typeof useStreamingChat>["messages"]) =>
  [...messages].reverse().find((m) => m.role === "assistant");

afterEach(() => cleanup());

describe("useStreamingChat — pre-stream HTTP error handling (licensing follow-ups Task 1)", () => {
  it("map-replaces the placeholder with a credit_wall bubble on a 402 carrying structured detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        makeErrorResponse(402, {
          detail: {
            reason: "You've used this month's credits.",
            upgradeRequired: true,
            resetDate: "2026-08-01T00:00:00Z",
          },
        })
      )
    );

    const { result } = renderHook(() => useStreamingChat());
    await act(async () => {
      await result.current.sendMessage("hi", baseParams);
    });

    const assistantMsg = lastAssistantMessage(result.current.messages);
    expect(assistantMsg?.confidence).toBe("credit_wall");
    // The reason is threaded onto `content` too, so the card is never blank
    // even for legacy shapes that carry no structured `detail` at all.
    expect(assistantMsg?.content).toBe("You've used this month's credits.");
    expect(assistantMsg?.detail).toEqual({
      reason: "You've used this month's credits.",
      upgradeRequired: true,
      resetDate: "2026-08-01T00:00:00Z",
    });
    // The plain-string banner stays quiet for 402s in the UI, but the string
    // state itself still gets the improved message (never the old canned copy).
    expect(result.current.error).toBe("You've used this month's credits.");
  });

  it("still renders a non-blank wall when a legacy 402 carries only a plain-string detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(makeErrorResponse(402, { detail: "Zoe AI is a Pro feature" }))
    );

    const { result } = renderHook(() => useStreamingChat());
    await act(async () => {
      await result.current.sendMessage("hi", baseParams);
    });

    const assistantMsg = lastAssistantMessage(result.current.messages);
    expect(assistantMsg?.confidence).toBe("credit_wall");
    expect(assistantMsg?.content).toBe("Zoe AI is a Pro feature");
    expect(assistantMsg?.detail).toBeUndefined();
  });

  it("keeps today's canned bubble for a non-402 HTTP error, with the improved reason as the banner message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        makeErrorResponse(500, { detail: { reason: "Internal server hiccup" } })
      )
    );

    const { result } = renderHook(() => useStreamingChat());
    await act(async () => {
      await result.current.sendMessage("hi", baseParams);
    });

    const assistantMsg = lastAssistantMessage(result.current.messages);
    expect(assistantMsg?.confidence).toBe("error");
    expect(assistantMsg?.content).toBe(
      "I'm sorry, I encountered an error. Please try again."
    );
    expect(assistantMsg?.detail).toBeUndefined();
    expect(result.current.error).toBe("Internal server hiccup");
  });

  it("shows the generic fallback bubble on a network failure (no HTTP response at all)", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    const { result } = renderHook(() => useStreamingChat());
    await act(async () => {
      await result.current.sendMessage("hi", baseParams);
    });

    const assistantMsg = lastAssistantMessage(result.current.messages);
    expect(assistantMsg?.confidence).toBe("error");
    expect(result.current.error).toBe("Failed to fetch");
  });

  it("leaves the SSE success path untouched — a normal 'complete' event resolves as before", async () => {
    const sseBody = `data: ${JSON.stringify({
      type: "complete",
      answer: "Hi there",
      confidence: "high",
    })}\n\n`;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeStreamResponse([sseBody])));

    const { result } = renderHook(() => useStreamingChat());
    await act(async () => {
      await result.current.sendMessage("hi", baseParams);
    });

    const assistantMsg = lastAssistantMessage(result.current.messages);
    expect(assistantMsg?.confidence).toBe("high");
    expect(assistantMsg?.content).toBe("Hi there");
    expect(result.current.error).toBe("");
  });
});
