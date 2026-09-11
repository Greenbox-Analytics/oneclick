import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { Board } from "@/types/boards";

let boards: Board[] = [];
vi.mock("@/hooks/useBoardsList", () => ({
  useBoardsList: () => ({ data: boards, isLoading: false, isFetching: false }),
  useCreateBoard: () => ({ mutate: vi.fn(), isPending: false }),
  useArchiveBoard: () => ({ mutate: vi.fn(), isPending: false }),
  useDeleteBoard: () => ({ mutate: vi.fn(), isPending: false }),
  useRestoreBoard: () => ({ mutate: vi.fn(), isPending: false }),
  useArchivedBoards: () => ({ data: [], isLoading: false }),
}));
vi.mock("@/hooks/useOrgs", () => ({
  useMyOrgs: () => ({ data: [], isLoading: false }),
  liveOrgs: () => [],
}));
vi.mock("@/hooks/useWorkspaceScope", () => ({
  useWorkspaceScope: () => ({ enabled: false, scopeId: null, ready: true }),
}));
vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => ({ user: { id: "u1" } }),
}));

import { BoardSwitcher } from "../BoardSwitcher";

const wrap = (ui: React.ReactElement) => (
  <QueryClientProvider client={new QueryClient()}>
    <MemoryRouter>{ui}</MemoryRouter>
  </QueryClientProvider>
);

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("BoardSwitcher — Personal context default", () => {
  // The Personal context used to leave the board unselected ("All personal
  // boards" = a union of every personal board). It now behaves like a team
  // context: a board is always selected, and the default is the artistless
  // "Personal" board, not whichever board happens to sort first.
  it("auto-selects the artistless Personal board", async () => {
    boards = [
      { id: "b-artist", name: "Nova", artist_id: "a1", team_id: null },
      { id: "b-personal", name: "Personal", artist_id: null, team_id: null },
    ];
    const onBoardChange = vi.fn();
    render(wrap(<BoardSwitcher teamId={null} boardId={undefined} onBoardChange={onBoardChange} />));

    await waitFor(() => expect(onBoardChange).toHaveBeenCalledWith("b-personal", null));
  });

  it("falls back to the first board when no artistless board exists", async () => {
    boards = [{ id: "b-artist", name: "Nova", artist_id: "a1", team_id: null }];
    const onBoardChange = vi.fn();
    render(wrap(<BoardSwitcher teamId={null} boardId={undefined} onBoardChange={onBoardChange} />));

    await waitFor(() => expect(onBoardChange).toHaveBeenCalledWith("b-artist", null));
  });

  it("keeps a valid selection and never offers an all-boards option", () => {
    boards = [
      { id: "b-artist", name: "Nova", artist_id: "a1", team_id: null },
      { id: "b-personal", name: "Personal", artist_id: null, team_id: null },
    ];
    const onBoardChange = vi.fn();
    render(wrap(<BoardSwitcher teamId={null} boardId="b-artist" onBoardChange={onBoardChange} />));

    expect(onBoardChange).not.toHaveBeenCalled();
    expect(screen.queryByText(/all personal boards/i)).toBeNull();
  });
});
