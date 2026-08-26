// src/components/admin/AdminPalette.tsx
// ⌘K palette for the admin console — jump to a user (server-side search),
// an organization (already-loaded list), or a view.
//
// Built on Dialog + Command directly rather than CommandDialog because the
// user results are already filtered server-side; cmdk's own scorer would
// re-filter them and drop legitimate matches.
import { useDeferredValue, useState } from "react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { useAdminUsers } from "@/hooks/useAdmin";
import { useAdminOrgs } from "@/hooks/useAdminOrgs";
import { tierLabel } from "@/lib/tiers";

export interface PaletteAction {
  label: string;
  run: () => void;
}

export function AdminPalette({
  open,
  onOpenChange,
  actions,
  onGoUser,
  onGoOrg,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  actions: PaletteAction[];
  onGoUser: (userId: string) => void;
  onGoOrg: (orgId: string) => void;
}) {
  const [query, setQuery] = useState("");
  const term = query.trim().toLowerCase();
  const deferred = useDeferredValue(query.trim());
  const usersQuery = useAdminUsers(deferred, 1, 6);
  const users = deferred.length >= 2 ? (usersQuery.data?.users ?? []) : [];
  const orgs = (useAdminOrgs().data ?? [])
    .filter((o) => term && (o.name ?? "").toLowerCase().includes(term))
    .slice(0, 5);

  const pick = (run: () => void) => {
    onOpenChange(false);
    setQuery("");
    run();
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="overflow-hidden p-0 shadow-lg">
        <Command shouldFilter={false}>
          <CommandInput
            value={query}
            onValueChange={setQuery}
            placeholder="Search users, organizations, or run an action…"
          />
          <CommandList>
            <CommandEmpty>
              {deferred.length >= 2 && usersQuery.isLoading ? "Searching…" : "No matches."}
            </CommandEmpty>
            {users.length > 0 && (
              <CommandGroup heading="Users">
                {users.map((u) => (
                  <CommandItem key={u.id} value={u.id} onSelect={() => pick(() => onGoUser(u.id))}>
                    <span className="truncate">{u.name ?? u.email ?? u.id}</span>
                    <span className="ml-auto truncate pl-3 text-[11px] text-muted-foreground">
                      {u.name ? u.email : tierLabel(u.tier)}
                    </span>
                  </CommandItem>
                ))}
              </CommandGroup>
            )}
            {orgs.length > 0 && (
              <CommandGroup heading="Organizations">
                {orgs.map((o) => (
                  <CommandItem key={o.id} value={o.id} onSelect={() => pick(() => onGoOrg(o.id))}>
                    <span className="truncate">{o.name ?? "Organization"}</span>
                    <span className="ml-auto pl-3 font-mono text-[11px] text-muted-foreground">
                      {o.status} · {o.memberCount} members
                    </span>
                  </CommandItem>
                ))}
              </CommandGroup>
            )}
            <CommandGroup heading="Actions">
              {actions
                .filter((a) => !term || a.label.toLowerCase().includes(term))
                .map((a) => (
                  <CommandItem key={a.label} value={a.label} onSelect={() => pick(a.run)}>
                    {a.label}
                  </CommandItem>
                ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </DialogContent>
    </Dialog>
  );
}
