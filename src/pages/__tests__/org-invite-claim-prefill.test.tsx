// The invite email links to /orgs/invite/{token}?email=…&signup=1 so /auth
// can prefill the invitee's address. The claim page must (1) latch and scrub
// those params from the address bar on arrival, whether signed out or not,
// (2) stash the email beside the token, and (3) hand /auth the email + tab
// hint in router state — never in the URL.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { ApiError } from "@/lib/apiFetch";
import { readPendingInvite } from "@/lib/pendingInvite";

const h = vi.hoisted(() => ({
  user: null as { id: string; email: string } | null,
  signOut: vi.fn(),
  accept: vi.fn(),
}));

vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => ({ user: h.user, loading: false, signOut: h.signOut }),
}));
vi.mock("@/integrations/supabase/client", () => ({ supabase: {} }));
vi.mock("@/hooks/useOrgs", () => ({
  useOrgInvitePreview: () => ({ data: undefined }),
  useAcceptOrgInvite: () => ({ mutateAsync: h.accept, isPending: false }),
  useDeclineOrgInvite: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));
vi.mock("sonner", () => ({
  toast: Object.assign(vi.fn(), { success: vi.fn(), info: vi.fn(), error: vi.fn() }),
}));

const { default: OrgInviteClaim } = await import("@/pages/OrgInviteClaim");

const Probe = () => {
  const { pathname, search, state } = useLocation();
  return (
    <>
      <div data-testid="loc">{pathname + search}</div>
      <div data-testid="state">{JSON.stringify(state ?? null)}</div>
    </>
  );
};

function renderAt(url: string) {
  return render(
    <MemoryRouter initialEntries={[url]}>
      <Probe />
      <Routes>
        <Route path="/orgs/invite/:token" element={<OrgInviteClaim />} />
        <Route path="/auth" element={<div>AUTH</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

const loc = () => screen.getByTestId("loc").textContent;
const state = () => JSON.parse(screen.getByTestId("state").textContent!);

beforeEach(() => {
  sessionStorage.clear();
  h.user = null;
  h.signOut.mockResolvedValue(undefined);
});

afterEach(() => {
  cleanup();
});

describe("OrgInviteClaim — signed out", () => {
  it("scrubs the link params, stashes the email, and sends /auth the hint in state", async () => {
    renderAt("/orgs/invite/tok?email=a%2Bb%40c.d&signup=1");

    await waitFor(() => expect(loc()).toBe("/orgs/invite/tok"));
    expect(readPendingInvite()).toEqual({
      token: "tok",
      accepted: false,
      orgName: null,
      kind: null,
      email: "a+b@c.d",
    });

    fireEvent.click(screen.getByRole("button", { name: /sign in \/ create account/i }));
    expect(loc()).toBe("/auth?redirect=%2Forgs%2Finvite%2Ftok");
    expect(state()).toEqual({ email: "a+b@c.d", tab: "signup" });
  });

  it("points an existing user at Sign In", async () => {
    renderAt("/orgs/invite/tok?email=a%40b.c");

    await waitFor(() => expect(loc()).toBe("/orgs/invite/tok"));
    fireEvent.click(screen.getByRole("button", { name: /sign in \/ create account/i }));
    expect(state()).toEqual({ email: "a@b.c", tab: "signin" });
  });

  it("still scrubs a malformed email and passes no prefill", async () => {
    renderAt("/orgs/invite/tok?email=nope&signup=1");

    await waitFor(() => expect(loc()).toBe("/orgs/invite/tok"));
    expect(readPendingInvite()?.email).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /sign in \/ create account/i }));
    expect(state()).toEqual({ email: null, tab: "signup" });
  });

  it("behaves as before without the params", () => {
    renderAt("/orgs/invite/tok");

    expect(loc()).toBe("/orgs/invite/tok");
    expect(readPendingInvite()).toEqual({ token: "tok", accepted: false, orgName: null, kind: null, email: null });
    fireEvent.click(screen.getByRole("button", { name: /sign in \/ create account/i }));
    expect(state()).toEqual({ email: null, tab: "signin" });
  });
});

describe("OrgInviteClaim — signed in with the wrong account", () => {
  beforeEach(() => {
    h.user = { id: "u1", email: "wrong@example.test" };
    h.accept.mockRejectedValue(new ApiError("Invite was sent to a different email", 403));
  });

  it("names the invited address and forwards it to Sign In after signing out", async () => {
    renderAt("/orgs/invite/tok?email=a%40b.c");

    await waitFor(() => expect(loc()).toBe("/orgs/invite/tok"));
    fireEvent.click(screen.getByRole("button", { name: /accept invitation/i }));
    await screen.findByText(/wrong account/i);
    expect(screen.getByText(/this invite was sent to a@b\.c/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /sign out/i }));
    await waitFor(() => expect(loc()).toBe("/auth?redirect=%2Forgs%2Finvite%2Ftok"));
    expect(h.signOut).toHaveBeenCalled();
    expect(state()).toEqual({ email: "a@b.c", tab: "signin" });
    expect(readPendingInvite()?.email).toBe("a@b.c");
  });

  it("keeps the generic copy when no address is known", async () => {
    renderAt("/orgs/invite/tok");

    fireEvent.click(screen.getByRole("button", { name: /accept invitation/i }));
    await screen.findByText(/wrong account/i);
    expect(screen.getByText(/sent to a different email address/i)).toBeInTheDocument();
  });
});
