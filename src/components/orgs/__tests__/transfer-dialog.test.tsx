import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
// Registers jest-dom matchers (toBeDisabled, etc.) on vitest's `expect` for
// this file — mirrors ../dissolve-dialog.test.tsx.
import "@testing-library/jest-dom/vitest";
import { TransferCreditsDialog } from "../TransferCreditsDialog";
import { ApiError } from "@/lib/apiFetch";

const transferMutate = vi.fn();
const transferReset = vi.fn();
let mutationState: { isPending: boolean; error: unknown } = { isPending: false, error: null };
let reserveBalance: number | null = 5000;

vi.mock("@/hooks/useOrgs", () => ({
  useTransferCredits: () => ({
    mutate: transferMutate,
    isPending: mutationState.isPending,
    error: mutationState.error,
    reset: transferReset,
  }),
}));

vi.mock("@/hooks/useEntitlements", () => ({
  useEntitlements: () => ({ data: { credits: { reserveBalance } } }),
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  mutationState = { isPending: false, error: null };
  reserveBalance = 5000;
});

function setup() {
  const onOpenChange = vi.fn();
  render(<TransferCreditsDialog orgId="org-1" orgName="Greenbox Analytics" open onOpenChange={onOpenChange} />);
  return { onOpenChange };
}

describe("TransferCreditsDialog", () => {
  it("caps the transfer amount at the caller's reserve balance", () => {
    reserveBalance = 500;
    setup();

    const input = screen.getByLabelText(/credits to transfer/i) as HTMLInputElement;
    expect(input).toHaveAttribute("max", "500");

    fireEvent.change(input, { target: { value: "600" } });
    expect(screen.getByRole("button", { name: /^transfer$/i })).toBeDisabled();

    fireEvent.change(input, { target: { value: "500" } });
    expect(screen.getByRole("button", { name: /^transfer$/i })).not.toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /^transfer$/i }));
    expect(transferMutate).toHaveBeenCalledWith(
      { orgId: "org-1", amount: 500 },
      expect.objectContaining({ onSuccess: expect.any(Function) }),
    );
  });

  it("renders the 409 insufficient-reserve reason verbatim", () => {
    mutationState = {
      isPending: false,
      error: new ApiError("You don't have enough reserve credits for this transfer.", 409),
    };
    setup();

    expect(screen.getByText("You don't have enough reserve credits for this transfer.")).toBeInTheDocument();
  });

  it("zero reserve balance points at buying packs instead of an amount input", () => {
    reserveBalance = 0;
    setup();

    expect(screen.queryByLabelText(/credits to transfer/i)).toBeNull();
    expect(screen.queryByRole("button", { name: /^transfer$/i })).toBeNull();
    expect(screen.getByText(/personal billing/i)).toBeInTheDocument();
    expect(screen.getByText(/don.t have any reserve credits/i)).toBeInTheDocument();
  });
});
