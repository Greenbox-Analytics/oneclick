import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
// Registers jest-dom matchers (toBeDisabled, etc.) on vitest's `expect` for
// this file — mirrors src/components/oneclick/__tests__/gate-dialogs.test.tsx.
import "@testing-library/jest-dom/vitest";
import { DissolveOrgDialog } from "../DissolveOrgDialog";
import type { DissolvePreview } from "@/hooks/useOrgs";

const navigateMock = vi.fn();
vi.mock("react-router-dom", () => ({
  useNavigate: () => navigateMock,
}));

const dissolveMutate = vi.fn();
let previewData: DissolvePreview | undefined;

vi.mock("@/hooks/useOrgs", () => ({
  useDissolvePreview: () => ({ data: previewData, isLoading: false }),
  useDissolveOrg: () => ({ mutate: dissolveMutate, isPending: false }),
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  previewData = undefined;
});

function setup(preview: DissolvePreview) {
  previewData = preview;
  const onOpenChange = vi.fn();
  render(<DissolveOrgDialog orgId="org-1" orgName="Greenbox Analytics" open onOpenChange={onOpenChange} />);
  return { onOpenChange };
}

const emptyPreview: DissolvePreview = { recipients: [], forfeitReserve: 0, inertBundle: 0, memberCount: 0 };

describe("DissolveOrgDialog", () => {
  it("disables the confirm button until the org name is typed exactly", () => {
    setup(emptyPreview);
    const confirmBtn = screen.getByRole("button", { name: /dissolve team/i });
    expect(confirmBtn).toBeDisabled();

    fireEvent.change(screen.getByLabelText(/type/i), { target: { value: "Greenbox" } });
    expect(confirmBtn).toBeDisabled();

    fireEvent.change(screen.getByLabelText(/type/i), { target: { value: "Greenbox Analytics" } });
    expect(confirmBtn).not.toBeDisabled();
  });

  it("renders recipients, using 'returned to you' copy for fallback rows", () => {
    setup({
      recipients: [
        { artistId: "a1", artistName: "Rome", userId: "u1", email: "rome@example.com", fallback: false },
        { artistId: "a2", artistName: "Solo Artist", userId: "u2", email: null, fallback: true },
      ],
      forfeitReserve: 0,
      inertBundle: 0,
      memberCount: 2,
    });

    expect(screen.getByText("Rome")).toBeInTheDocument();
    expect(screen.getByText("rome@example.com")).toBeInTheDocument();
    expect(screen.getByText("Solo Artist")).toBeInTheDocument();
    expect(screen.getByText(/returned to you/i)).toBeInTheDocument();
  });

  it("hides the forfeit line when the reserve balance is zero, but shows the inert-bundle line", () => {
    setup({ recipients: [], forfeitReserve: 0, inertBundle: 500, memberCount: 0 });

    // The dialog's static description also contains the word "forfeited", so
    // match the actual forfeit-line copy ("Purchased credits forfeited"),
    // not just any occurrence of the word.
    expect(screen.queryByText(/purchased credits forfeited/i)).toBeNull();
    expect(screen.getByText(/500 monthly credits expire with the team/i)).toBeInTheDocument();
  });

  it("submits the exact-name confirmation and navigates away on success", () => {
    setup(emptyPreview);
    fireEvent.change(screen.getByLabelText(/type/i), { target: { value: "Greenbox Analytics" } });
    fireEvent.click(screen.getByRole("button", { name: /dissolve team/i }));

    expect(dissolveMutate).toHaveBeenCalledWith(
      { orgId: "org-1", confirmName: "Greenbox Analytics" },
      expect.objectContaining({ onSuccess: expect.any(Function) }),
    );

    // The component navigates from its mutate() onSuccess callback — invoke
    // what it actually passed in to observe that behavior, rather than
    // asserting navigation happened on a mutate() call that (mocked) never
    // resolves on its own.
    dissolveMutate.mock.calls[0][1].onSuccess();
    expect(navigateMock).toHaveBeenCalledWith("/organization");
  });
});
