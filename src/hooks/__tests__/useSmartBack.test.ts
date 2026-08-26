import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook } from "@testing-library/react";

const navigate = vi.fn();
vi.mock("react-router-dom", () => ({ useNavigate: () => navigate }));

import { useSmartBack } from "@/hooks/useSmartBack";

/** Stand in for a history entry at stack position `idx`. */
const atIndex = (idx: number | null | undefined) => {
  const state = idx === undefined ? null : { idx, usr: null, key: "abc" };
  vi.spyOn(window.history, "state", "get").mockReturnValue(state);
};

describe("useSmartBack", () => {
  beforeEach(() => navigate.mockClear());

  it("goes back when there is an in-app entry behind this one", () => {
    atIndex(3);
    renderHook(() => useSmartBack("/fallback")).result.current();
    expect(navigate).toHaveBeenCalledWith(-1);
  });

  it("uses the fallback on the first entry — the app was opened here", () => {
    atIndex(0);
    renderHook(() => useSmartBack("/fallback")).result.current();
    expect(navigate).toHaveBeenCalledWith("/fallback");
  });

  it("uses the fallback when history carries no router index at all", () => {
    atIndex(undefined);
    renderHook(() => useSmartBack("/fallback")).result.current();
    expect(navigate).toHaveBeenCalledWith("/fallback");
  });

  it("defaults the fallback to the dashboard", () => {
    atIndex(0);
    renderHook(() => useSmartBack()).result.current();
    expect(navigate).toHaveBeenCalledWith("/dashboard");
  });

  // The regression this hook was rewritten for: a deep link bounced through a
  // `replace` (ProtectedRoute → /auth → sign in) lands on stack position 0 with
  // a freshly minted location.key. Keying on the key made Back walk the user
  // out of the app; keying on `idx` keeps them in it.
  it("still uses the fallback after a replace onto the first entry", () => {
    atIndex(0);
    renderHook(() => useSmartBack("/portfolio")).result.current();
    expect(navigate).toHaveBeenCalledWith("/portfolio");
    expect(navigate).not.toHaveBeenCalledWith(-1);
  });
});
