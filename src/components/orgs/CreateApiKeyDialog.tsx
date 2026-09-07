// src/components/orgs/CreateApiKeyDialog.tsx
// "New API key": name + optional expiry -> secret shown ONCE. One key type:
// every key is this team's credential. Copy is for the person paying, not a
// developer — "credits", never bearer/hash.
import { useState } from "react";
import { Check, Copy, CalendarIcon, Loader2 } from "lucide-react";
import { format } from "date-fns";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useCreatePartnerKey, type PartnerKeyFolder } from "@/hooks/usePartnerKeys";
import {
  endOfLocalDayIso,
  expiryFromPreset,
  expiryLabel,
  EXPIRY_PRESETS,
  toInputValue,
  tomorrowInputValue,
  type ExpiryPreset,
} from "@/lib/partnerKeys";
import { FolderSelect, useFolderChoice } from "./FolderSelect";

export function CreateApiKeyDialog({
  orgId,
  folders,
  open,
  onOpenChange,
}: {
  orgId: string;
  folders: PartnerKeyFolder[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const create = useCreatePartnerKey();
  const folder = useFolderChoice(orgId);
  const [label, setLabel] = useState("");
  const [preset, setPreset] = useState<ExpiryPreset>("never");
  const [customDate, setCustomDate] = useState<Date | undefined>(undefined);
  const [calendarOpen, setCalendarOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);

  const secret = create.data?.secret ?? null;
  const missing = !label.trim() ? "Add a name" : folder.missing;
  const canSubmit = !missing && !create.isPending && !folder.isPending;
  const tomorrow = new Date(`${tomorrowInputValue()}T00:00:00`);
  // Custom with no date chosen is treated as "never" for the payload, but the
  // helper copy below still nudges toward picking one instead of reading "never".
  const resolvedExpiry = preset === "custom" ? (customDate ? toInputValue(customDate) : null) : expiryFromPreset(preset);
  const expiryHelp =
    preset === "custom" && !customDate ? "Pick a date, or the key won't expire." : expiryLabel(resolvedExpiry);

  const reset = () => {
    setLabel("");
    setPreset("never");
    setCustomDate(undefined);
    folder.reset();
    setCopied(false);
    setCopyFailed(false);
    create.reset();
  };

  const handleOpenChange = (next: boolean) => {
    // While a secret is on screen the ONLY way out is Done — a backdrop click
    // must not throw away the one chance to copy it.
    if (!next && secret) return;
    if (!next) reset();
    onOpenChange(next);
  };

  const handleDone = () => {
    reset();
    onOpenChange(false);
  };

  const handleCreate = () => {
    if (!canSubmit) return;
    // A typed folder name has to exist before the key can point at it.
    folder.resolve((folderId) =>
      create.mutate({
        orgId,
        label: label.trim(),
        folder_id: folderId,
        ...(resolvedExpiry ? { expires_at: endOfLocalDayIso(resolvedExpiry) } : {}),
      }),
    );
  };

  const handleCopy = async () => {
    if (!secret) return;
    try {
      await navigator.clipboard.writeText(secret);
      setCopied(true);
      setCopyFailed(false);
    } catch {
      // navigator.clipboard is absent on a non-secure origin — say so, the
      // secret is unrecoverable once Done is clicked.
      setCopied(false);
      setCopyFailed(true);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent
        onInteractOutside={(e) => secret && e.preventDefault()}
        onEscapeKeyDown={(e) => secret && e.preventDefault()}
      >
        {secret ? (
          <>
            <DialogHeader>
              <DialogTitle>Your new API key</DialogTitle>
              <DialogDescription>Copy this now. For your security, we can't show it again.</DialogDescription>
            </DialogHeader>
            <div className="flex items-center gap-2">
              <code className="flex-1 rounded-md border bg-muted px-3 py-2 text-[13px] font-mono break-all select-all">
                {secret}
              </code>
              <Button type="button" variant="outline" size="sm" onClick={handleCopy}>
                {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                <span className="ml-1.5">{copied ? "Copied" : "Copy"}</span>
              </Button>
            </div>
            {copyFailed && (
              <p role="alert" className="text-[12px] text-destructive">
                Couldn&apos;t copy automatically — select the key above and copy it yourself.
              </p>
            )}
            <DialogFooter>
              <Button type="button" onClick={handleDone}>
                Done
              </Button>
            </DialogFooter>
          </>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>New API key</DialogTitle>
              <DialogDescription>
                Treat it like a password: anyone holding it can run calculations on this team&apos;s credits.
              </DialogDescription>
            </DialogHeader>
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="api-key-name">Name</Label>
                <Input
                  id="api-key-name"
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                  placeholder="Your key name"
                  maxLength={120}
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>Expires</Label>
                <div
                  role="radiogroup"
                  aria-label="Key expiry"
                  className="inline-flex flex-wrap rounded-lg border border-border bg-muted/40 p-0.5"
                >
                  {EXPIRY_PRESETS.map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      role="radio"
                      aria-checked={preset === p.id}
                      onClick={() => setPreset(p.id)}
                      className={`rounded-md px-2.5 py-1 text-[12px] font-semibold transition-colors ${
                        preset === p.id ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
                {preset === "custom" && (
                  <Popover open={calendarOpen} onOpenChange={setCalendarOpen}>
                    <PopoverTrigger asChild>
                      <Button type="button" variant="outline" className="w-fit justify-start text-left font-normal">
                        <CalendarIcon className="mr-2 h-4 w-4" />
                        {customDate ? format(customDate, "MMM d, yyyy") : "Pick a date"}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0" align="start">
                      <Calendar
                        mode="single"
                        selected={customDate}
                        onSelect={(d) => {
                          setCustomDate(d);
                          setCalendarOpen(false);
                        }}
                        disabled={{ before: tomorrow }}
                        defaultMonth={customDate ?? tomorrow}
                      />
                    </PopoverContent>
                  </Popover>
                )}
                <p className="text-[12px] text-muted-foreground">{expiryHelp}</p>
              </div>
              <FolderSelect id="api-key-folder" folders={folders} choice={folder} />
              {create.error && (
                <p role="alert" className="text-[13px] text-destructive">
                  {create.error.message}
                </p>
              )}
              {missing && (
                <p id="api-key-create-hint" className="text-[12px] text-muted-foreground">
                  {missing} to create this key.
                </p>
              )}
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => handleOpenChange(false)}>
                Cancel
              </Button>
              {/* aria-disabled, not disabled: keyboard and screen-reader users can reach it and hear why. */}
              <Button
                type="button"
                onClick={handleCreate}
                aria-disabled={!canSubmit}
                aria-describedby={missing ? "api-key-create-hint" : undefined}
                className={canSubmit ? undefined : "opacity-50"}
              >
                {create.isPending && <Loader2 className="w-4 h-4 mr-1.5 animate-spin" />}
                Create
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
