// An org invitee bounced to /auth used to see an empty email field on a page
// that always opened on Sign In. The claim page now hands the invitee's
// address and a tab hint over in router state (never the URL), with the
// session stash as the fallback for a reload or the Google round-trip.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { MemoryRouter } from "react-router-dom";
import type { InitialEntry } from "react-router-dom";
import { stashPendingInvite } from "@/lib/pendingInvite";

const h = vi.hoisted(() => ({
  signIn: vi.fn(),
  signUp: vi.fn(),
  signInWithGoogle: vi.fn(),
}));

vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => ({ signIn: h.signIn, signUp: h.signUp, signInWithGoogle: h.signInWithGoogle }),
}));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ toast: vi.fn() }) }));

const { default: Auth } = await import("@/pages/Auth");

const INVITE_REDIRECT = "?redirect=%2Forgs%2Finvite%2Ftok";

function renderAt(entry: InitialEntry) {
  return render(
    <MemoryRouter initialEntries={[entry]}>
      <Auth />
    </MemoryRouter>,
  );
}

const tab = (name: string) => screen.getByRole("tab", { name });
const signInEmail = () => document.querySelector<HTMLInputElement>("#signin-email")!;
const signUpEmail = () => document.querySelector<HTMLInputElement>("#signup-email")!;
// Radix activates a tab on pointer-down, not click.
const switchTo = (name: string) => fireEvent.mouseDown(tab(name));

beforeEach(() => {
  sessionStorage.clear();
});

afterEach(() => {
  cleanup();
});

describe("Auth — org invite prefill", () => {
  it("opens on Sign Up with the email prefilled on both tabs for a new invitee", () => {
    renderAt({ pathname: "/auth", search: INVITE_REDIRECT, state: { email: "ada@example.com", tab: "signup" } });

    expect(tab("Sign Up")).toHaveAttribute("aria-selected", "true");
    expect(signUpEmail()).toHaveValue("ada@example.com");

    switchTo("Sign In");
    expect(signInEmail()).toHaveValue("ada@example.com");
  });

  it("opens on Sign In for an existing invitee", () => {
    renderAt({ pathname: "/auth", search: INVITE_REDIRECT, state: { email: "ada@example.com", tab: "signin" } });

    expect(tab("Sign In")).toHaveAttribute("aria-selected", "true");
    expect(signInEmail()).toHaveValue("ada@example.com");
  });

  it("falls back to the session stash when the state is gone (reload, OAuth round-trip)", () => {
    stashPendingInvite({ token: "tok", accepted: false, email: "ada@example.com" });
    renderAt({ pathname: "/auth", search: INVITE_REDIRECT });

    expect(tab("Sign In")).toHaveAttribute("aria-selected", "true");
    expect(signInEmail()).toHaveValue("ada@example.com");
  });

  it("ignores a stash for a different invite or a non-invite redirect", () => {
    stashPendingInvite({ token: "other", accepted: false, email: "ada@example.com" });
    renderAt({ pathname: "/auth", search: INVITE_REDIRECT });
    expect(signInEmail()).toHaveValue("");
    cleanup();

    stashPendingInvite({ token: "tok", accepted: false, email: "ada@example.com" });
    renderAt({ pathname: "/auth", search: "?redirect=%2Fdashboard" });
    expect(signInEmail()).toHaveValue("");
  });

  it("keeps the user's edit when they switch tabs", () => {
    renderAt({ pathname: "/auth", search: INVITE_REDIRECT, state: { email: "ada@example.com", tab: "signup" } });

    fireEvent.change(signUpEmail(), { target: { value: "other@example.com" } });
    switchTo("Sign In");
    switchTo("Sign Up");
    expect(signUpEmail()).toHaveValue("other@example.com");
  });

  it("drops a malformed hint rather than rendering it", () => {
    renderAt({ pathname: "/auth", search: INVITE_REDIRECT, state: { email: "not an email", tab: "signup" } });

    expect(tab("Sign Up")).toHaveAttribute("aria-selected", "true");
    expect(signUpEmail()).toHaveValue("");
  });

  it("is unchanged on a plain visit", () => {
    renderAt("/auth");

    expect(tab("Sign In")).toHaveAttribute("aria-selected", "true");
    expect(signInEmail()).toHaveValue("");
  });

  it("stashes the invite email alongside the token on sign-up", async () => {
    h.signUp.mockResolvedValue(undefined);
    renderAt({ pathname: "/auth", search: INVITE_REDIRECT, state: { email: "ada@example.com", tab: "signup" } });

    fireEvent.change(document.querySelector("#signup-name")!, { target: { value: "Ada" } });
    fireEvent.change(document.querySelector("#signup-password")!, { target: { value: "hunter22" } });
    fireEvent.submit(document.querySelector("#signup-email")!.closest("form")!);

    await vi.waitFor(() => expect(h.signUp).toHaveBeenCalled());
    expect(h.signUp).toHaveBeenCalledWith("ada@example.com", "hunter22", "Ada", "/orgs/invite/tok");
    expect(JSON.parse(sessionStorage.getItem("msanii_pending_org_invite")!)).toEqual({
      token: "tok",
      accepted: false,
      email: "ada@example.com",
    });
  });
});
