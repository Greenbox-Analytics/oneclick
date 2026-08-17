import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const mutate = vi.fn();
vi.mock("@/hooks/useBoardsList", () => ({
  useUpdateBoard: () => ({ mutate, isPending: false }),
}));
vi.mock("@/hooks/useOrgs", () => ({
  useOrgRoster: () => ({ data: [{ user_id: "u2", role: "member", full_name: "Bea" }] }),
}));

import { BoardSettingsDialog } from "../BoardSettingsDialog";

// Radix's Popover (MultiSelectCombobox) measures its content via ResizeObserver,
// which jsdom doesn't implement. Stub is enough — nothing here asserts on layout.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver;

const wrap = (ui: React.ReactElement) => (
  <QueryClientProvider client={new QueryClient()}>
    <MemoryRouter>{ui}</MemoryRouter>
  </QueryClientProvider>
);

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("BoardSettingsDialog", () => {
  it("saves name + restricted + member ids", async () => {
    render(
      wrap(
        <BoardSettingsDialog
          open
          onOpenChange={() => {}}
          board={{
            id: "b1",
            name: "Q3",
            team_id: "org1",
            owner_id: "u1",
            restricted: false,
            member_user_ids: [],
          }}
          teamId="org1"
          canManage
        />,
      ),
    );

    fireEvent.click(screen.getByLabelText(/only specific people/i));
    fireEvent.click(screen.getByRole("button", { name: /choose people/i }));
    fireEvent.click(await screen.findByText("Bea"));
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));

    await waitFor(() =>
      expect(mutate).toHaveBeenCalledWith(
        expect.objectContaining({
          boardId: "b1",
          name: "Q3",
          restricted: true,
          member_user_ids: ["u2"],
        }),
        expect.anything(),
      ),
    );
  });

  it("hides visibility controls when the caller can't manage", () => {
    render(
      wrap(
        <BoardSettingsDialog
          open
          onOpenChange={() => {}}
          board={{ id: "b1", name: "Q3", team_id: "org1", owner_id: "u1" }}
          teamId="org1"
          canManage={false}
        />,
      ),
    );
    expect(screen.queryByText(/who can see this board/i)).toBeNull();
  });

  it("omits visibility fields entirely for a personal board", () => {
    render(
      wrap(
        <BoardSettingsDialog
          open
          onOpenChange={() => {}}
          board={{ id: "b2", name: "Solo", team_id: null, owner_id: "u1" }}
          teamId={null}
          canManage
        />,
      ),
    );
    expect(screen.queryByText(/who can see this board/i)).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));
    expect(mutate).toHaveBeenCalledWith({ boardId: "b2", name: "Solo" }, expect.anything());
  });

  it("clears the list when switching back to Everyone (replace-set needs [], not undefined)", async () => {
    // The most common save there is, and it had zero coverage on either side of
    // the wire: `member_user_ids: memberIds` instead of `restricted ? memberIds : []`
    // passes every other test here, and the backend's replace-set would then
    // never clear the narrowing.
    render(
      wrap(
        <BoardSettingsDialog
          open
          onOpenChange={() => {}}
          board={{
            id: "b1",
            name: "Q3",
            team_id: "org1",
            owner_id: "u1",
            restricted: true,
            member_user_ids: ["u2"],
          }}
          teamId="org1"
          canManage
        />,
      ),
    );
    fireEvent.click(screen.getByLabelText(/everyone on the team/i));
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));
    await waitFor(() =>
      expect(mutate).toHaveBeenCalledWith(
        expect.objectContaining({ restricted: false, member_user_ids: [] }),
        expect.anything(),
      ),
    );
  });

  it("keeps a suspended member visible so they can be removed", async () => {
    // useOrgRoster is ACTIVE-only but suspend does NOT purge board_members, so
    // without a synthetic option the id sits in state with no chip and no
    // checkbox — invisible, unremovable, and every later save 422s.
    render(
      wrap(
        <BoardSettingsDialog
          open
          onOpenChange={() => {}}
          board={{
            id: "b1",
            name: "Q3",
            team_id: "org1",
            owner_id: "u1",
            restricted: true,
            member_user_ids: ["u-suspended"],
          }}
          teamId="org1"
          canManage
        />,
      ),
    );
    // Rendered BOTH as a removable chip on the trigger and as a checked option
    // in the list — either alone would leave the admin unable to drop them.
    const shown = await screen.findAllByText(/suspended member/i);
    expect(shown.length).toBeGreaterThanOrEqual(1);
    fireEvent.click(screen.getByRole("button", { name: /choose people/i }));
    expect(screen.getAllByText(/suspended member/i).length).toBeGreaterThan(shown.length);
  });

  it("excludes the board's creator from the pickable options", async () => {
    // The creator always has access; offering them is a no-op the backend
    // would happily store. Dropping the filter keeps every other test green
    // because the mocked roster never contains the owner.
    render(
      wrap(
        <BoardSettingsDialog
          open
          onOpenChange={() => {}}
          board={{ id: "b1", name: "Q3", team_id: "org1", owner_id: "u2", restricted: true, member_user_ids: [] }}
          teamId="org1"
          canManage
        />,
      ),
    );
    fireEvent.click(screen.getByRole("button", { name: /choose people/i }));
    expect(screen.queryByText("Bea")).toBeNull(); // Bea IS u2, the creator
  });
});
