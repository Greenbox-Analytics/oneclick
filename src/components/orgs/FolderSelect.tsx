// The folder picker shared by "New API key" and "Move to folder": none, an
// existing folder, or a new one typed inline. Native <select> — no Radix
// portal to fight inside a dialog.
import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useCreatePartnerKeyFolder, type PartnerKeyFolder } from "@/hooks/usePartnerKeys";

/** Sentinel option value: "New folder…" reveals the name input. */
export const NEW_FOLDER = "__new__";

const SELECT =
  "block h-9 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground focus:border-primary focus:outline-none focus:ring-[3px] focus:ring-primary/15";

/** Picker state plus the create-then-assign step, so both callers resolve a
 * folder the same way. */
export function useFolderChoice(orgId: string, initial: string | null = null) {
  const createFolder = useCreatePartnerKeyFolder();
  const [value, setValue] = useState<string>(initial ?? "");
  const [newName, setNewName] = useState("");

  const missing = value === NEW_FOLDER && !newName.trim() ? "Name the new folder" : null;

  const reset = () => {
    setValue(initial ?? "");
    setNewName("");
  };

  /** Hands the caller the folder id to send. Only "New folder…" is async, so
   * the common case stays synchronous. */
  const resolve = (then: (folderId: string | null) => void) => {
    if (value !== NEW_FOLDER) {
      then(value || null);
      return;
    }
    // Idempotent on the name, so a retry is safe.
    createFolder.mutateAsync({ orgId, name: newName.trim() }).then((folder) => then(folder.id), () => {});
  };

  return { value, setValue, newName, setNewName, missing, reset, resolve, isPending: createFolder.isPending };
}

export function FolderSelect({
  id,
  folders,
  choice,
}: {
  id: string;
  folders: PartnerKeyFolder[];
  choice: ReturnType<typeof useFolderChoice>;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor={id}>Folder</Label>
      <select
        id={id}
        className={SELECT}
        value={choice.value}
        onChange={(e) => choice.setValue(e.target.value)}
      >
        <option value="">No folder</option>
        {folders.map((f) => (
          <option key={f.id} value={f.id}>
            {f.name}
          </option>
        ))}
        <option value={NEW_FOLDER}>New folder…</option>
      </select>
      {choice.value === NEW_FOLDER && (
        <Input
          aria-label="New folder name"
          placeholder="Folder name"
          maxLength={80}
          value={choice.newName}
          onChange={(e) => choice.setNewName(e.target.value)}
        />
      )}
      <p className="text-[12px] text-muted-foreground">Groups keys by use case. Spend is reported per folder.</p>
    </div>
  );
}
