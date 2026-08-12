// src/components/billing/HeaderContextSwitcher.tsx
// "Working as" pill in the page header, left of the credits ticker.
//
// The context is more than a payer switch, which is why it earns header space:
// useActiveTeam() reads it to decide which org a NEW artist is created into
// (NewArtist.tsx / NewArtistDialog.tsx) and whether ArtistProfile offers
// "Move to <org>". Leaving it buried in Profile meant creating team work into
// the wrong owner was a silent, easy mistake.
//
// Renders nothing for the overwhelming majority of users: no seat anywhere (or
// LICENSING_ENABLED off) means one available context, and nothing to switch.
import { Building2, Check, User } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import { useBillingContextSwitcher } from "@/hooks/useBillingContext";

export function HeaderContextSwitcher() {
  const { canSwitch, orgContexts, value, currentLabel, switchTo, isPending } =
    useBillingContextSwitcher();

  if (!canSwitch) return null;

  const isPersonal = value === "personal";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          disabled={isPending}
          title={`Working as ${currentLabel}`}
          aria-label={`Working as ${currentLabel} — change`}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[12.5px] transition-colors",
            "border-border bg-background text-muted-foreground hover:text-foreground hover:bg-muted/60",
            isPending && "opacity-60",
          )}
        >
          {isPersonal ? (
            <User className="w-3.5 h-3.5 flex-none" />
          ) : (
            <Building2 className="w-3.5 h-3.5 flex-none text-primary" />
          )}
          <span className="truncate max-w-[72px] sm:max-w-[140px]">{currentLabel}</span>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-60">
        <DropdownMenuLabel className="font-normal">
          <div className="text-sm font-medium">Working as</div>
          <p className="text-xs text-muted-foreground mt-0.5">
            Whose credits pay, and which team new artists join
          </p>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => switchTo("personal")} className="gap-2">
          <User className="w-4 h-4 text-muted-foreground" />
          <span className="flex-1">Personal</span>
          {isPersonal && <Check className="w-3.5 h-3.5 text-primary" />}
        </DropdownMenuItem>
        {orgContexts.map((o) => (
          <DropdownMenuItem key={o.orgId} onClick={() => switchTo(o.orgId)} className="gap-2">
            <Building2 className="w-4 h-4 text-muted-foreground" />
            <span className="flex-1 min-w-0">
              <span className="block truncate">{o.orgName}</span>
              {o.pending && (
                <span className="block text-[11px] text-muted-foreground">Activating soon</span>
              )}
            </span>
            {value === o.orgId && <Check className="w-3.5 h-3.5 text-primary flex-none" />}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
