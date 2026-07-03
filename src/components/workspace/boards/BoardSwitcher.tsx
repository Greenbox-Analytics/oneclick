import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Archive, History, MoreHorizontal, Pencil, Plus, Trash2, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useTeams } from "@/hooks/useTeams";
import {
  useArchiveBoard,
  useArchivedBoards,
  useBoardsList,
  useCreateBoard,
  useDeleteBoard,
  useRenameBoard,
  useRestoreBoard,
} from "@/hooks/useBoardsList";
import { DeleteConfirmDialog } from "@/components/workspace/boards/DeleteConfirmDialog";

const PERSONAL = "personal";

interface BoardSwitcherProps {
  /** null = Personal context. Controlled by the parent (Workspace). */
  teamId: string | null;
  boardId: string | undefined;
  onBoardChange: (boardId: string | undefined, teamId: string | null) => void;
}

/**
 * Fully controlled: `teamId`/`boardId` live in the parent (Workspace) so the
 * selection survives tab switches (Radix unmounts inactive TabsContent) — the
 * switcher only reads props and emits `onBoardChange`.
 */
export function BoardSwitcher({ teamId, boardId, onBoardChange }: BoardSwitcherProps) {
  const { data: teams, isLoading: teamsLoading } = useTeams();
  const {
    data: boards,
    isLoading: boardsLoading,
    isFetching: boardsFetching,
  } = useBoardsList(teamId);

  const createBoard = useCreateBoard();
  const renameBoard = useRenameBoard();
  const archiveBoard = useArchiveBoard();
  const deleteBoard = useDeleteBoard();

  const [newDialogOpen, setNewDialogOpen] = useState(false);
  const [newBoardName, setNewBoardName] = useState("");
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [renameName, setRenameName] = useState("");
  const [archiveDialogOpen, setArchiveDialogOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [archivedOpen, setArchivedOpen] = useState(false);

  const selectedBoard = boards?.find((b) => b.id === boardId);

  // Delete/Restore + the archived-boards list are gated to the same permission as
  // archiving a team board: Personal (owner) or a team admin. Non-admin team members
  // never see these affordances (the backend enforces the same gate).
  const activeTeam = teams?.find((t) => t.id === teamId);
  const canManage = teamId == null || activeTeam?.my_role === "admin";
  // The empty state is a TEAM concept only. Personal keeps boardId undefined = the
  // personal-boards union (today's behavior), so it never shows "No boards yet".
  const hasNoBoards =
    teamId !== null && !boardsLoading && Array.isArray(boards) && boards.length === 0;

  // Stale-team guard: if the selected team is gone (archived / left), reset the
  // parent to Personal so Workspace doesn't keep rendering the dangling team board.
  const teamIsStale =
    teamId !== null &&
    !teamsLoading &&
    Array.isArray(teams) &&
    !teams.some((t) => t.id === teamId);
  useEffect(() => {
    if (teamIsStale) onBoardChange(undefined, null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [teamIsStale]);

  // In a TEAM context, auto-select the first board once the list resolves, and drop a
  // stale selection (e.g. after archiving the active board) that is no longer present.
  // Gated on !boardsFetching so we never re-select from the STALE cached list during
  // the background refetch an archive invalidation kicks off — only from fresh data.
  // Emits only when the computed selection differs from the prop, so it cannot loop.
  // In Personal (teamId === null) we do NOT auto-select — boardId stays undefined so
  // KanbanBoard renders the personal-boards union (no forced remount / narrowing).
  useEffect(() => {
    if (teamId === null || teamIsStale) return;
    if (boardsLoading || boardsFetching || !boards) return;
    if (boardId && boards.some((b) => b.id === boardId)) return; // current selection still valid
    if (boards.length > 0) {
      onBoardChange(boards[0].id, teamId);
    } else if (boardId) {
      // Selection points at a board that's gone from the (now empty) list — drop it.
      onBoardChange(undefined, teamId);
    }
    // boards.length === 0 with no selection → zero-boards state renders; nothing to emit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [boards, boardsLoading, boardsFetching, teamId, boardId, teamIsStale]);

  const handleContextChange = (value: string) => {
    const newTeamId = value === PERSONAL ? null : value;
    if (newTeamId === teamId) return;
    // Never keep the previous context's board selected under a new label — reset and
    // emit immediately so the parent drops it. The effect re-selects once boards load.
    onBoardChange(undefined, newTeamId);
  };

  const handleBoardChange = (value: string) => {
    onBoardChange(value, teamId);
  };

  const handleCreateBoard = () => {
    if (createBoard.isPending) return;
    const name = newBoardName.trim();
    if (!name) return;
    createBoard.mutate(
      { name, team_id: teamId },
      {
        onSuccess: (newBoard) => {
          onBoardChange(newBoard.id, teamId);
          setNewBoardName("");
          setNewDialogOpen(false);
        },
      },
    );
  };

  const openRename = () => {
    if (!selectedBoard) return;
    setRenameName(selectedBoard.name);
    setRenameDialogOpen(true);
  };

  const handleRenameBoard = () => {
    if (renameBoard.isPending) return;
    const name = renameName.trim();
    if (!name || !boardId) return;
    renameBoard.mutate(
      { boardId, name },
      {
        onSuccess: () => {
          toast.success("Board renamed");
          setRenameDialogOpen(false);
        },
      },
    );
  };

  const handleArchiveBoard = () => {
    if (!boardId) return;
    archiveBoard.mutate(boardId, {
      onSuccess: () => {
        // Drop the archived board; the auto-select effect will pick the next one
        // (or fall into the zero-boards state) once the FRESH list arrives
        // (it is gated on !isFetching, so the stale cache can't re-select it).
        onBoardChange(undefined, teamId);
        setArchiveDialogOpen(false);
      },
    });
  };

  return (
    <div className="mb-4 flex flex-wrap items-center gap-2">
      {/* Context: Personal + teams */}
      <Select value={teamId ?? PERSONAL} onValueChange={handleContextChange}>
        <SelectTrigger className="w-[180px]" aria-label="Board context">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={PERSONAL}>Personal</SelectItem>
          {(teams ?? []).map((team) => (
            <SelectItem key={team.id} value={team.id}>
              {team.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Board within the selected context */}
      {boardsLoading ? (
        <div className="flex h-10 items-center gap-2 px-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          Loading boards…
        </div>
      ) : hasNoBoards ? (
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">No boards yet</span>
          <Button size="sm" variant="outline" onClick={() => setNewDialogOpen(true)}>
            <Plus className="mr-2 h-4 w-4" />
            New board
          </Button>
          {/* Keep Restore reachable even with zero active boards: the board-actions
              menu (the other entry point to the archived list) is hidden in this
              empty state, so an admin who archived the last board could otherwise
              never get back in. */}
          {canManage && (
            <Button size="sm" variant="ghost" onClick={() => setArchivedOpen(true)}>
              <History className="mr-2 h-4 w-4" />
              Archived boards…
            </Button>
          )}
        </div>
      ) : (
        <Select value={boardId ?? ""} onValueChange={handleBoardChange}>
          <SelectTrigger className="w-[200px]" aria-label="Select board">
            <SelectValue placeholder={teamId === null ? "All personal boards" : "Select board"} />
          </SelectTrigger>
          <SelectContent>
            {(boards ?? []).map((board) => (
              <SelectItem key={board.id} value={board.id}>
                {board.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}

      {/* New / Rename / Archive */}
      {!hasNoBoards && (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="icon" title="Board actions" aria-label="Board actions">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onSelect={() => setNewDialogOpen(true)}>
              <Plus className="mr-2 h-4 w-4" />
              New board
            </DropdownMenuItem>
            <DropdownMenuItem disabled={!selectedBoard} onSelect={openRename}>
              <Pencil className="mr-2 h-4 w-4" />
              Rename board
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              disabled={!selectedBoard}
              className="text-destructive focus:text-destructive"
              onSelect={() => setArchiveDialogOpen(true)}
            >
              <Archive className="mr-2 h-4 w-4" />
              Archive board
            </DropdownMenuItem>
            {canManage && (
              <DropdownMenuItem
                disabled={!selectedBoard}
                className="text-destructive focus:text-destructive"
                onSelect={() => setDeleteOpen(true)}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Delete board permanently…
              </DropdownMenuItem>
            )}
            {canManage && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuItem onSelect={() => setArchivedOpen(true)}>
                  <History className="mr-2 h-4 w-4" />
                  Archived boards…
                </DropdownMenuItem>
              </>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      )}

      {/* New board dialog */}
      <Dialog
        open={newDialogOpen}
        onOpenChange={(open) => {
          setNewDialogOpen(open);
          if (!open) setNewBoardName(""); // drop a cancelled draft so it doesn't persist
        }}
      >
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>New board</DialogTitle>
            <DialogDescription>Create a board to organize tasks in this context.</DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <Label htmlFor="new-board-name">Board name</Label>
            <Input
              id="new-board-name"
              placeholder="e.g. Q3 Campaign"
              value={newBoardName}
              onChange={(e) => setNewBoardName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleCreateBoard();
              }}
              autoFocus
            />
            <p className="text-xs text-muted-foreground">
              {teamId ? "This board will belong to the selected team." : "This board will be personal to you."}
            </p>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setNewDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateBoard} disabled={!newBoardName.trim() || createBoard.isPending}>
              {createBoard.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Rename board dialog */}
      <Dialog open={renameDialogOpen} onOpenChange={setRenameDialogOpen}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>Rename board</DialogTitle>
            <DialogDescription>Give this board a new name.</DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <Label htmlFor="rename-board-name">Board name</Label>
            <Input
              id="rename-board-name"
              value={renameName}
              onChange={(e) => setRenameName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleRenameBoard();
              }}
              autoFocus
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenameDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleRenameBoard} disabled={!renameName.trim() || renameBoard.isPending}>
              {renameBoard.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Archive confirm */}
      <AlertDialog open={archiveDialogOpen} onOpenChange={setArchiveDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Archive this board?</AlertDialogTitle>
            <AlertDialogDescription>
              {selectedBoard ? `"${selectedBoard.name}" ` : "This board "}
              and its columns and tasks will be archived. You can no longer add tasks to it.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleArchiveBoard}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Archive
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Permanently delete the ACTIVE board. Active boards carry no task_count
          (list_boards doesn't compute it), so no impact count is passed — the dialog's
          base "…and everything in it" copy already conveys the scope. The mutation
          sends the BARE name; the dialog validates the typed `delete-<name>` locally
          and the server re-compares. */}
      <DeleteConfirmDialog
        open={deleteOpen}
        onOpenChange={setDeleteOpen}
        name={selectedBoard?.name ?? ""}
        resourceType="board"
        isPending={deleteBoard.isPending}
        onConfirm={() =>
          selectedBoard &&
          deleteBoard.mutate(
            { boardId, confirmName: selectedBoard.name },
            {
              onSuccess: () => {
                setDeleteOpen(false);
                onBoardChange(undefined, teamId);
              },
            },
          )
        }
      />

      {/* Archived boards list (admin/owner only) — mounted only when canManage so a
          non-admin team member never fires the (would-be 403) archived fetch. */}
      {canManage && (
        <ArchivedBoardsDialog open={archivedOpen} onOpenChange={setArchivedOpen} teamId={teamId} />
      )}
    </div>
  );
}

interface ArchivedBoardsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  teamId: string | null;
}

/**
 * Lists archived boards for the current context with Restore + permanent Delete.
 * Archived boards DO carry `task_count` (from `list_archived_boards`), so their
 * delete impact shows the real number. Per-row delete state is a single id.
 */
function ArchivedBoardsDialog({ open, onOpenChange, teamId }: ArchivedBoardsDialogProps) {
  const { data: archivedBoards, isLoading } = useArchivedBoards(teamId);
  const restoreBoard = useRestoreBoard();
  const deleteBoard = useDeleteBoard();
  const [archivedDeleteId, setArchivedDeleteId] = useState<string | null>(null);

  const target = archivedBoards?.find((b) => b.id === archivedDeleteId);

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Archived boards</DialogTitle>
            <DialogDescription>Restore a board or delete it permanently.</DialogDescription>
          </DialogHeader>
          <div className="max-h-[60vh] space-y-2 overflow-y-auto">
            {isLoading ? (
              <div className="flex items-center gap-2 px-1 py-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading…
              </div>
            ) : !archivedBoards || archivedBoards.length === 0 ? (
              <p className="px-1 py-2 text-sm text-muted-foreground">No archived boards.</p>
            ) : (
              archivedBoards.map((b) => (
                <div
                  key={b.id}
                  className="flex items-center justify-between gap-2 rounded-md border p-2"
                >
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{b.name}</p>
                    <p className="text-xs text-muted-foreground">{b.task_count ?? 0} tasks</p>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={restoreBoard.isPending}
                      onClick={() => restoreBoard.mutate(b.id)}
                    >
                      Restore
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-destructive hover:text-destructive"
                      onClick={() => setArchivedDeleteId(b.id)}
                    >
                      Delete
                    </Button>
                  </div>
                </div>
              ))
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <DeleteConfirmDialog
        open={!!archivedDeleteId}
        onOpenChange={(o) => !o && setArchivedDeleteId(null)}
        name={target?.name ?? ""}
        resourceType="board"
        impact={target ? `${target.task_count ?? 0} tasks` : undefined}
        isPending={deleteBoard.isPending}
        onConfirm={() =>
          target &&
          deleteBoard.mutate(
            { boardId: target.id, confirmName: target.name },
            { onSuccess: () => setArchivedDeleteId(null) },
          )
        }
      />
    </>
  );
}
