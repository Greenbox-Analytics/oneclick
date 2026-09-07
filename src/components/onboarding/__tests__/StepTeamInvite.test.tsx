import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import StepTeamInvite from "../StepTeamInvite";

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const setup = (overrides: Partial<React.ComponentProps<typeof StepTeamInvite>> = {}) => {
  const onNext = vi.fn();
  const onBack = vi.fn();
  render(
    <StepTeamInvite
      orgName="GNS Music"
      kind="self_serve"
      accepted={false}
      onNext={onNext}
      onBack={onBack}
      isLoading={false}
      {...overrides}
    />,
  );
  return { onNext, onBack };
};

describe("StepTeamInvite", () => {
  it("tells an invitee they are joining the named team and never mentions a plan to buy", () => {
    setup();
    expect(screen.getByRole("heading", { name: "You're joining GNS Music" })).toBeInTheDocument();
    expect(screen.getByText(/once you accept the invitation/i)).toBeInTheDocument();
    expect(screen.queryByText(/basic/i)).not.toBeInTheDocument();
  });

  it("switches to past tense once the invite has been accepted", () => {
    setup({ accepted: true });
    expect(screen.getByRole("heading", { name: "You're on GNS Music" })).toBeInTheDocument();
    expect(screen.getByText(/there's no plan to choose/i)).toBeInTheDocument();
  });

  it("falls back to the org noun when the preview could not name the org", () => {
    setup({ orgName: null, kind: "self_serve" });
    expect(screen.getByRole("heading", { name: "You're joining your team" })).toBeInTheDocument();
    cleanup();
    setup({ orgName: null, kind: "enterprise" });
    expect(screen.getByRole("heading", { name: "You're joining your organization" })).toBeInTheDocument();
  });

  it("wires Continue and Back, and disables both while saving", () => {
    const { onNext, onBack } = setup();
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    fireEvent.click(screen.getByRole("button", { name: "Back" }));
    expect(onNext).toHaveBeenCalledTimes(1);
    expect(onBack).toHaveBeenCalledTimes(1);

    cleanup();
    setup({ isLoading: true });
    expect(screen.getByRole("button", { name: "..." })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Back" })).toBeDisabled();
  });
});
