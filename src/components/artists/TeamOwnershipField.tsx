import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";

/**
 * The ownership choice shown on BOTH artist-creation paths (NewArtist page and
 * NewArtistDialog). One component rather than two copies because the copy is
 * money-adjacent — it tells someone whose credits and whose storage this artist
 * will consume — and two copies drift.
 *
 * Renders nothing when there is no active team: with no org billing context
 * both forms look exactly as they do today.
 */
export function TeamOwnershipField({
  teamName,
  keepPrivate,
  onChange,
  id = "keep-private",
}: {
  teamName: string | null;
  keepPrivate: boolean;
  onChange: (v: boolean) => void;
  id?: string;
}) {
  if (!teamName) return null;
  return (
    <div className="flex items-start gap-2 rounded-lg border border-border bg-muted/40 px-4 py-3">
      <Checkbox
        id={id}
        checked={keepPrivate}
        onCheckedChange={(v) => onChange(v === true)}
        className="mt-0.5"
      />
      <div>
        <Label htmlFor={id} className="font-normal">
          Keep this artist private to me
        </Label>
        <p className="text-xs text-muted-foreground mt-0.5">
          By default this artist is shared with {teamName} — everyone on the team can see it, and
          its storage and AI usage come out of the team&apos;s allowance.
        </p>
      </div>
    </div>
  );
}
