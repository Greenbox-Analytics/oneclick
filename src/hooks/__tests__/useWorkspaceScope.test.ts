import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderHook } from "@testing-library/react";
import type { Entitlements } from "@/hooks/useEntitlements";

const mockEntitlements = vi.fn();
vi.mock("@/hooks/useEntitlements", () => ({
  useEntitlements: () => mockEntitlements(),
}));

const { useWorkspaceScope, UNSCOPED_KEY } = await import("@/hooks/useWorkspaceScope");

const ORG = "org-a";

function withPayload(payload: Partial<Entitlements> | undefined, isLoading = false) {
  mockEntitlements.mockReturnValue({ data: payload, isLoading });
  return renderHook(() => useWorkspaceScope()).result.current;
}

beforeEach(() => mockEntitlements.mockReset());

describe("useWorkspaceScope", () => {
  it("sends no scope param when the payload omits workspaceScope", () => {
    // The rollback signal: WORKSPACE_SCOPING_ENABLED off means the backend
    // omits the field, and the frontend must then behave exactly as before.
    const s = withPayload({});
    expect(s.enabled).toBe(false);
    expect(s.scopeParam).toBeUndefined();
    expect(s.withScope("/artists")).toBe("/artists");
  });

  it("uses a distinct cache key when scoping is off", () => {
    // Not "personal" — otherwise flipping the flag would COLLIDE with entries
    // cached under the scoped personal key instead of invalidating them.
    expect(withPayload({}).scopeKey).toBe(UNSCOPED_KEY);
    expect(withPayload({ workspaceScope: { type: "personal" } }).scopeKey).toBe("personal");
  });

  it("appends scope=personal in the personal workspace", () => {
    const s = withPayload({ workspaceScope: { type: "personal" } });
    expect(s.scopeId).toBeNull();
    expect(s.scopeLabel).toBe("Personal");
    expect(s.withScope("/artists")).toBe("/artists?scope=personal");
  });

  it("appends the org id in an org workspace", () => {
    const s = withPayload({ workspaceScope: { type: "org", orgId: ORG, orgName: "Acme" } });
    expect(s.scopeId).toBe(ORG);
    expect(s.scopeKey).toBe(ORG);
    expect(s.scopeLabel).toBe("Acme");
    expect(s.withScope("/artists")).toBe(`/artists?scope=${ORG}`);
  });

  it("uses & when the url already has a query string", () => {
    const s = withPayload({ workspaceScope: { type: "org", orgId: ORG, orgName: "Acme" } });
    expect(s.withScope("/registry/works?artist_id=x")).toBe(`/registry/works?artist_id=x&scope=${ORG}`);
  });

  it("is not ready while entitlements are loading", () => {
    // Queries gate on `ready`. Fetching first would send a param-less request,
    // cache it under the wrong key, then refetch — a double fetch and a visible
    // flash of the wrong workspace on every cold load.
    expect(withPayload(undefined, true).ready).toBe(false);
  });

  it("becomes ready — unscoped — when entitlements fail", () => {
    // No payload and not loading = the fetch errored. Degrade to "send no
    // param" so the server falls back to the stored context, rather than
    // blocking every list behind a failed request forever.
    const s = withPayload(undefined, false);
    expect(s.ready).toBe(true);
    expect(s.enabled).toBe(false);
  });
});
