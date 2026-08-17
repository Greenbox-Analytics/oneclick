import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { MultiSelectCombobox } from "./MultiSelectCombobox";
import { useUpdateBoard } from "@/hooks/useBoardsList";
import { useOrgRoster } from "@/hooks/useOrgs";
import type { Board } from "@/types/boards";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  board: Board;
  /** Owning org id; null for a personal board (visibility section hidden). */
  teamId: string | null;
  /** Org admin or the board's creator — the only people who may change visibility. */
  canManage: boolean;
}

/**
 * Board settings (spec 2026-08-16 §3). Name for anyone who can open the board;
 * "who can see this board" for admins/creator on TEAM boards. Membership itself
 * lives in the Teams console — this dialog only picks from the roster.
 */
export function BoardSettingsDialog({ open, onOpenChange, board, teamId, canManage }: Props) {
  const update = useUpdateBoard();
  const { data: roster } = useOrgRoster(teamId);
  const [name, setName] = useState(board.name);
  const [restricted, setRestricted] = useState(!!board.restricted);
  const [memberIds, setMemberIds] = useState<string[]>(board.member_user_ids ?? []);

  // Re-seed on every OPEN, not just when a different board is selected. The
  // component stays mounted for as long as a board is selected (Radix unmounts
  // the content, not the component), so without `open` here a Cancel would
  // leave the abandoned edits in state — and the next Save would silently
  // apply a change the user thought they had discarded.
  useEffect(() => {
    setName(board.name);
    setRestricted(!!board.restricted);
    setMemberIds(board.member_user_ids ?? []);
  }, [open, board.id, board.name, board.restricted, board.member_user_ids]);

  const showVisibility = teamId != null && canManage;
  const rosterIds = new Set((roster ?? []).map((m) => m.user_id));
  const options = [
    ...(roster ?? [])
      .filter((m) => m.user_id !== board.owner_id) // the creator always sees it
      .map((m) => ({ id: m.user_id, label: m.full_name || "Unnamed member" })),
    // Someone on the list whose seat is no longer ACTIVE (suspend deliberately
    // does NOT purge board_members — spec §1). The roster is active-only, so
    // without a synthetic option their id sits in `memberIds` with no chip and
    // no checkbox: invisible, and impossible to remove. Same idiom as
    // TaskFields' "(no longer in team)" assignees.
    ...(board.member_user_ids ?? [])
      .filter((id) => !rosterIds.has(id) && id !== board.owner_id)
      .map((id) => ({ id, label: "Suspended member" })),
  ];

  const save = () => {
    update.mutate(
      {
        boardId: board.id,
        name: name.trim(),
        ...(showVisibility ? { restricted, member_user_ids: restricted ? memberIds : [] } : {}),
      },
      {
        onSuccess: () => {
          toast.success("Board settings saved");
          onOpenChange(false);
        },
      },
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Board settings</DialogTitle>
          <DialogDescription>
            Rename this board{showVisibility ? " and choose who on the team can see it" : ""}.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="board-settings-name">Board name</Label>
            <Input
              id="board-settings-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoFocus
            />
          </div>
          {showVisibility && (
            <div className="space-y-2">
              <Label>Who can see this board</Label>
              <RadioGroup
                value={restricted ? "restricted" : "team"}
                onValueChange={(v) => setRestricted(v === "restricted")}
              >
                <div className="flex items-center gap-2">
                  <RadioGroupItem value="team" id="vis-team" />
                  <Label htmlFor="vis-team" className="font-normal">
                    Everyone on the team
                  </Label>
                </div>
                <div className="flex items-center gap-2">
                  <RadioGroupItem value="restricted" id="vis-restricted" />
                  <Label htmlFor="vis-restricted" className="font-normal">
                    Only specific people
                  </Label>
                </div>
              </RadioGroup>
              {restricted && (
                <MultiSelectCombobox
                  options={options}
                  selected={memberIds}
                  onChange={setMemberIds}
                  placeholder="Choose people"
                  aria-label="Choose people"
                />
              )}
              <p className="text-xs text-muted-foreground">
                Team admins and you always see it. Members are managed in Teams —{" "}
                <Link to="/teams" className="underline">
                  Manage team members →
                </Link>
              </p>
            </div>
          )}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={save} disabled={!name.trim() || update.isPending}>
            {update.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
