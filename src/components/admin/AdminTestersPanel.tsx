// src/components/admin/AdminTestersPanel.tsx
// Admin console → Beta testers: grant form (with existing-account
// autocomplete), pending pre-signup invites, and active grants.
import { useDeferredValue, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { useAdminUsers } from "@/hooks/useAdmin";
import {
  useTesterGrants,
  useCreateTesterGrant,
  useRevokeTesterGrant,
  useRevokePendingTesterGrant,
} from "@/hooks/useTesterGrants";
import { tierLabel } from "@/lib/tiers";
import { PanelHeader, QueueRow, shortDate, Tag } from "@/components/admin/ui";

export function AdminTestersPanel({ onSelectUser }: { onSelectUser: (id: string) => void }) {
  const grantsQuery = useTesterGrants();
  const revokeGrant = useRevokeTesterGrant();
  const revokePendingGrant = useRevokePendingTesterGrant();

  const grants = grantsQuery.data ?? [];
  const active = grants.filter((g) => !g.pending);
  const pending = grants.filter((g) => g.pending);

  const handleRevoke = async (userId: string) => {
    try {
      await revokeGrant.mutateAsync(userId);
      toast.success("Tester access revoked.");
    } catch {
      toast.error("Failed to revoke tester access. Try again.");
    }
  };

  const handleRevokePending = async (pendingEmail: string) => {
    try {
      await revokePendingGrant.mutateAsync(pendingEmail);
      toast.success("Tester access revoked.");
    } catch {
      toast.error("Failed to revoke tester access. Try again.");
    }
  };

  return (
    <>
      <GrantForm />

      {pending.length > 0 && (
        <Card className="mb-4 overflow-hidden p-0">
          <PanelHeader
            title="Pending invites"
            subtitle="Granted to an email with no account yet — activates on first verified sign-in"
          >
            <Tag>{pending.length}</Tag>
          </PanelHeader>
          {pending.map((g) => (
            <QueueRow key={g.email}>
              <div className="min-w-0 flex-1">
                <div className="truncate text-[13.5px] font-medium">{g.email}</div>
                <div className="text-xs text-muted-foreground">
                  {g.grant_duration_days
                    ? `${g.grant_duration_days} days after first sign-in`
                    : "No expiry"}
                  {g.reason ? ` · ${g.reason}` : ""}
                </div>
              </div>
              <Button
                size="sm"
                variant="ghost"
                className="text-destructive hover:text-destructive"
                disabled={revokePendingGrant.isPending}
                onClick={() => handleRevokePending(g.email!)}
              >
                Revoke
              </Button>
            </QueueRow>
          ))}
        </Card>
      )}

      <Card className="overflow-hidden p-0">
        <PanelHeader
          title="Active testers"
          subtitle={`${active.length} account${active.length === 1 ? "" : "s"} with tester access`}
        />
        {grantsQuery.isLoading && (
          <div className="px-4 py-10 text-center text-sm text-muted-foreground">Loading…</div>
        )}
        {!grantsQuery.isLoading && active.length === 0 && (
          <div className="px-4 py-10 text-center text-sm text-muted-foreground">
            No active tester grants.
          </div>
        )}
        {active.map((g) => (
          <QueueRow key={g.user_id!} onClick={() => onSelectUser(g.user_id!)}>
            <div className="min-w-0 flex-1">
              <div className="truncate text-[13.5px] font-medium">
                {g.name || g.email || g.user_id}
              </div>
              <div className="truncate text-xs text-muted-foreground">
                {g.name && g.email ? g.email : g.reason || "tester"}
              </div>
            </div>
            <span className="text-xs text-muted-foreground">
              {g.expires_at ? `Expires ${shortDate(g.expires_at)}` : "No expiry"}
            </span>
            <Button
              size="sm"
              variant="ghost"
              className="text-destructive hover:text-destructive"
              disabled={revokeGrant.isPending}
              onClick={(e) => {
                e.stopPropagation();
                handleRevoke(g.user_id!);
              }}
            >
              Revoke
            </Button>
          </QueueRow>
        ))}
      </Card>
    </>
  );
}

// ---------------------------------------------------------------------------
// Grant form — email autocomplete over existing accounts
// ---------------------------------------------------------------------------

function GrantForm() {
  const createGrant = useCreateTesterGrant();

  const [email, setEmail] = useState("");
  const [durationDays, setDurationDays] = useState("");
  const [credits, setCredits] = useState("");
  const [reason, setReason] = useState("");

  // Autocomplete: search existing users by email as the admin types.
  //
  // - useDeferredValue lets React keep the input snappy while the network
  //   request and downstream render lag behind a tick or two (Vercel rule
  //   rerender-use-deferred-value). The React Query call is keyed by the
  //   deferred value, so it doesn't re-fetch on every keystroke when the user
  //   is typing fast.
  // - 2-char minimum avoids a noisy first-keystroke fetch returning every user.
  // - React Query caches by {search, page} for 30s, so backtracking is free.
  const deferredQuery = useDeferredValue(email.trim());
  const suggestionsQuery = useAdminUsers(deferredQuery, 1, 10);
  const showSuggestions = deferredQuery.length >= 2;
  const suggestions = showSuggestions ? (suggestionsQuery.data?.users ?? []) : [];

  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Reset the highlight when the suggestion set changes. Primitive dep
  // (length) per Vercel rule rerender-dependencies — referencing the array
  // would refire every render.
  useEffect(() => {
    setHighlightedIndex(suggestions.length > 0 ? 0 : -1);
  }, [suggestions.length]);

  // Single document-level mousedown listener while the dropdown is open;
  // cleaned up on close so we don't leak listeners across re-renders.
  useEffect(() => {
    if (!isOpen) return;
    const onMouseDown = (e: MouseEvent) => {
      if (!containerRef.current?.contains(e.target as Node)) setIsOpen(false);
    };
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [isOpen]);

  // Keep the highlighted item visible when arrowing past the viewport edge.
  useEffect(() => {
    if (highlightedIndex < 0 || !listRef.current) return;
    const item = listRef.current.children[highlightedIndex] as HTMLElement | undefined;
    item?.scrollIntoView({ block: "nearest" });
  }, [highlightedIndex]);

  const handleSelect = (selectedEmail: string) => {
    setEmail(selectedEmail);
    setIsOpen(false);
    setHighlightedIndex(-1);
  };

  const handleEmailKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!isOpen) {
      if (e.key === "ArrowDown" && suggestions.length > 0) {
        e.preventDefault();
        setIsOpen(true);
        setHighlightedIndex(0);
      }
      return;
    }
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setHighlightedIndex((i) => Math.min(i + 1, suggestions.length - 1));
        break;
      case "ArrowUp":
        e.preventDefault();
        setHighlightedIndex((i) => Math.max(i - 1, 0));
        break;
      case "Enter": {
        const picked = suggestions[highlightedIndex]?.email;
        if (picked) {
          e.preventDefault();
          handleSelect(picked);
        }
        break;
      }
      case "Escape":
        e.preventDefault();
        setIsOpen(false);
        break;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;
    try {
      const res = await createGrant.mutateAsync({
        email: email.trim(),
        grant_duration_days: durationDays ? Number(durationDays) : null,
        credits: credits ? Number(credits) : null,
        reason: reason.trim() || "tester",
      });
      if (res?.pending) {
        toast.success(
          `${email.trim()} doesn't have an account yet — tester access will activate on their first sign-in.`,
        );
      } else {
        toast.success(`Granted tester access to ${email.trim()}`);
      }
      setEmail("");
      setDurationDays("");
      setCredits("");
      setReason("");
    } catch {
      toast.error("Failed to grant tester access. Try again.");
    }
  };

  return (
    <Card className="mb-4 overflow-hidden p-0">
      <PanelHeader
        title="Grant tester access"
        subtitle="Start typing to match an existing account, or invite an email that hasn't signed up yet"
      />
      <form
        onSubmit={handleSubmit}
        className="grid grid-cols-1 items-end gap-3 p-4 md:grid-cols-[2fr_1fr_1fr_1.4fr_auto]"
      >
        <label className="text-xs font-medium text-muted-foreground">
          Email (required)
          <div ref={containerRef} className="relative mt-1">
            <Input
              type="email"
              required
              placeholder="user@example.com"
              value={email}
              onChange={(e) => {
                setEmail(e.target.value);
                setIsOpen(true);
              }}
              onFocus={() => setIsOpen(true)}
              onKeyDown={handleEmailKeyDown}
              autoComplete="off"
              role="combobox"
              aria-expanded={isOpen && suggestions.length > 0}
              aria-controls="tester-email-suggestions"
              aria-activedescendant={
                highlightedIndex >= 0 ? `tester-suggestion-${highlightedIndex}` : undefined
              }
            />
            {isOpen && showSuggestions && (
              <div
                id="tester-email-suggestions"
                role="listbox"
                ref={listRef}
                className="absolute z-50 mt-1 max-h-60 w-full overflow-auto rounded-md border border-border bg-popover text-popover-foreground shadow-md"
              >
                {suggestionsQuery.isLoading && suggestions.length === 0 && (
                  <div className="px-3 py-2 text-xs text-muted-foreground">Searching…</div>
                )}
                {!suggestionsQuery.isLoading && suggestions.length === 0 && (
                  <div className="px-3 py-2 text-xs text-muted-foreground">No matches</div>
                )}
                {suggestions.map((u, i) =>
                  u.email ? (
                    <button
                      type="button"
                      key={u.id}
                      id={`tester-suggestion-${i}`}
                      role="option"
                      aria-selected={i === highlightedIndex}
                      onMouseDown={(e) => {
                        // mousedown (not click) so we beat the document mousedown
                        // close-on-outside handler — by the time click fires the
                        // dropdown would already be closed.
                        e.preventDefault();
                        handleSelect(u.email!);
                      }}
                      onMouseEnter={() => setHighlightedIndex(i)}
                      className={cn(
                        "flex w-full flex-col gap-0.5 px-3 py-2 text-left text-sm",
                        i === highlightedIndex
                          ? "bg-accent text-accent-foreground"
                          : "hover:bg-accent/60",
                      )}
                    >
                      <span className="font-medium">{u.email}</span>
                      <span className="text-[11px] opacity-80">
                        {tierLabel(u.tier)}
                        {u.is_admin || u.is_env_admin ? " · Admin" : ""}
                        {u.has_override ? " · Override" : ""}
                      </span>
                    </button>
                  ) : null,
                )}
              </div>
            )}
          </div>
        </label>
        <label className="text-xs font-medium text-muted-foreground">
          Expires in (days)
          <Input
            type="number"
            min={1}
            max={3650}
            placeholder="Never"
            value={durationDays}
            onChange={(e) => setDurationDays(e.target.value)}
            className="mt-1"
          />
        </label>
        <label className="text-xs font-medium text-muted-foreground">
          Starting credits
          <Input
            type="number"
            min={1}
            max={1_000_000}
            placeholder="Default"
            value={credits}
            onChange={(e) => setCredits(e.target.value)}
            className="mt-1"
          />
        </label>
        <label className="text-xs font-medium text-muted-foreground">
          Reason
          <Input
            type="text"
            placeholder="tester"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            className="mt-1"
          />
        </label>
        <Button type="submit" disabled={createGrant.isPending}>
          Grant access
        </Button>
      </form>
    </Card>
  );
}
