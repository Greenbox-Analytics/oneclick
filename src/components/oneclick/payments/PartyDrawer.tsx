// src/components/oneclick/payments/PartyDrawer.tsx
import { useState } from "react";
import {
  X,
  Send,
  AlertTriangle,
  Folder,
  ChevronRight,
  CheckCheck,
  Clock,
  Coins,
  Scissors,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "@/components/ui/tooltip";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import {
  useRoyaltyPayee,
  useSetPayeeCurrency,
  useSplitPayee,
} from "@/hooks/useRoyalties";
import type { PayeeStatement, PayeeLine } from "@/hooks/useRoyalties";
import {
  PartyAvatar,
  StatusBadge,
  partyStatusLabel,
  fmtMoney,
  fmtDate,
  CURRENCIES,
} from "./shared";
import { PayeeTrendChart } from "./analytics/PayeeTrendChart";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface PartyDrawerProps {
  payeeId: string;
  base: string;
  onClose: () => void;
  onPayout: (ids: string[]) => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function periodLabel(s: PayeeStatement): string {
  if (s.period_start && s.period_end) {
    const fmt = (iso: string) =>
      new Date(iso + "T00:00:00").toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
      });
    return `${fmt(s.period_start)} – ${fmt(s.period_end)}`;
  }
  return s.royalty_statement_id.slice(0, 8);
}

function stateDotColor(state: string): string {
  if (state === "settled") return "bg-[hsl(150_55%_45%)]";
  if (state === "scheduled") return "bg-[hsl(217_70%_58%)]";
  return "bg-[hsl(0_72%_55%)]"; // unpaid — red
}

// ---------------------------------------------------------------------------
// Section label
// ---------------------------------------------------------------------------

function SectionLabel({
  children,
  right,
}: {
  children: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <div className="mb-2.5 flex items-center gap-2 text-[11.5px] font-bold uppercase tracking-wider text-muted-foreground">
      {children}
      <span className="h-px flex-1 bg-border" />
      {right && (
        <span className="font-normal normal-case tracking-normal">{right}</span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Loading skeleton
// ---------------------------------------------------------------------------

function DrawerSkeleton() {
  return (
    <div className="flex flex-1 flex-col gap-[22px] overflow-y-auto p-5 pb-7">
      {/* balances */}
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="rounded-xl border border-border p-3">
            <div className="h-[11px] w-16 animate-pulse rounded bg-muted" />
            <div className="mt-2 h-[18px] w-24 animate-pulse rounded bg-muted" />
          </div>
        ))}
      </div>
      {/* rows */}
      {[0, 1, 2].map((i) => (
        <div key={i} className="h-[44px] animate-pulse rounded-xl border border-border bg-muted" />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SplitPanel — compact checklist + name input
// ---------------------------------------------------------------------------

interface SplitPanelProps {
  payeeId: string;
  statements: PayeeStatement[];
  onDone: () => void;
}

function SplitPanel({ payeeId, statements, onDone }: SplitPanelProps) {
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [newName, setNewName] = useState("");
  const split = useSplitPayee();

  // Flatten all lines across statements, tagging settled ones
  const allLines: Array<{ line: PayeeLine; settled: boolean }> = [];
  statements.forEach((stmt) => {
    const settled = stmt.state === "settled";
    stmt.lines.forEach((ln) => allLines.push({ line: ln, settled }));
  });

  const toggle = (id: string) =>
    setChecked((s) => {
      const n = new Set(s);
      if (n.has(id)) {
        n.delete(id);
      } else {
        n.add(id);
      }
      return n;
    });

  const canSplit = checked.size > 0 && newName.trim().length > 0;

  const handleSplit = () => {
    split.mutate(
      {
        id: payeeId,
        line_ids: Array.from(checked),
        new_display_name: newName.trim(),
      },
      { onSuccess: onDone },
    );
  };

  return (
    <TooltipProvider>
      <div className="flex flex-col gap-3">
        <div className="flex max-h-[220px] flex-col gap-1 overflow-y-auto pr-1">
          {allLines.map(({ line, settled }) =>
            settled ? (
              <Tooltip key={line.line_id}>
                <TooltipTrigger asChild>
                  <label className="flex cursor-not-allowed items-center gap-2.5 rounded-lg border border-dashed border-border px-2.5 py-1.5 opacity-45">
                    <input
                      type="checkbox"
                      className="h-[14px] w-[14px] shrink-0"
                      disabled
                    />
                    <span className="min-w-0 flex-1 truncate text-[12.5px]">
                      {line.song_title}
                    </span>
                    <span className="shrink-0 font-mono text-[11.5px] text-muted-foreground">
                      {fmtMoney(line.amount_owed, line.statement_currency)}
                    </span>
                  </label>
                </TooltipTrigger>
                <TooltipContent>
                  Settled by a paid invoice — can't be split
                </TooltipContent>
              </Tooltip>
            ) : (
              <label
                key={line.line_id}
                className="flex cursor-pointer items-center gap-2.5 rounded-lg border border-border px-2.5 py-1.5 hover:bg-muted/40"
              >
                <input
                  type="checkbox"
                  className="h-[14px] w-[14px] shrink-0"
                  checked={checked.has(line.line_id)}
                  onChange={() => toggle(line.line_id)}
                />
                <span className="min-w-0 flex-1 truncate text-[12.5px]">
                  {line.song_title}
                  {line.role && (
                    <span className="ml-1.5 text-[11px] text-muted-foreground">
                      · {line.role}
                    </span>
                  )}
                </span>
                <span className="shrink-0 font-mono text-[11.5px] text-muted-foreground">
                  {fmtMoney(line.amount_owed, line.statement_currency)}
                </span>
              </label>
            ),
          )}
        </div>

        <div className="flex gap-2">
          <Input
            placeholder="New display name…"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            className="h-8 text-[12.5px]"
          />
          <Button
            size="sm"
            disabled={!canSplit || split.isPending}
            onClick={handleSplit}
            className="shrink-0"
          >
            {split.isPending ? "Splitting…" : "Split"}
          </Button>
        </div>

        {split.isError && (
          <p className="text-[11.5px] text-destructive">
            {split.error instanceof Error
              ? split.error.message
              : "Split failed"}
          </p>
        )}
      </div>
    </TooltipProvider>
  );
}

// ---------------------------------------------------------------------------
// EmailEditor — view/edit the payee's payment email
// ---------------------------------------------------------------------------

function EmailEditor({ payeeId, currentEmail }: { payeeId: string; currentEmail?: string }) {
  const patchPayee = useSetPayeeCurrency();
  const [draft, setDraft] = useState(currentEmail ?? "");

  const trimmed = draft.trim();
  const dirty = trimmed !== (currentEmail ?? "");
  const valid = /^\S+@\S+\.\S+$/.test(trimmed);

  return (
    <div>
      <div className="flex items-center gap-2">
        <Input
          type="email"
          placeholder="payee@example.com"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          className="h-10 w-[210px]"
        />
        {dirty && (
          <Button
            size="sm"
            disabled={!valid || patchPayee.isPending}
            onClick={() => patchPayee.mutate({ id: payeeId, email: trimmed })}
          >
            {patchPayee.isPending ? "Saving…" : "Save"}
          </Button>
        )}
        {!dirty && patchPayee.isSuccess && (
          <span className="text-[11.5px] text-muted-foreground">Saved</span>
        )}
      </div>
      <p className="mt-1.5 text-[11.5px] text-muted-foreground">
        Used for PayPal payments and emailed receipts.
      </p>
      {patchPayee.isError && (
        <p className="mt-1 text-[11.5px] text-destructive">Couldn't save the email. Please try again.</p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main drawer
// ---------------------------------------------------------------------------

export function PartyDrawer({
  payeeId,
  base,
  onClose,
  onPayout,
}: PartyDrawerProps) {
  const { data, isLoading, isError, refetch } = useRoyaltyPayee(payeeId, base);
  const setCurrency = useSetPayeeCurrency();

  const [openProj, setOpenProj] = useState<string | null>(null);
  const [openStmt, setOpenStmt] = useState<string | null>(null);
  const [showSplit, setShowSplit] = useState(false);

  const summary = data?.summary;
  const projects = data?.projects ?? [];
  const payouts = data?.payouts ?? [];

  return (
    <>
      {/* Overlay */}
      <div
        className="fixed inset-0 z-[60] bg-black/40 backdrop-blur-[2px] duration-200 animate-in fade-in"
        onClick={onClose}
      />

      {/* Slide-over */}
      <aside className="fixed inset-y-0 right-0 z-[61] flex w-[480px] max-w-[94vw] flex-col border-l border-border bg-background shadow-lg duration-200 animate-in slide-in-from-right max-[560px]:w-full">
        {/* ── Header ──────────────────────────────────────────────────────── */}
        <div className="flex items-start gap-3.5 border-b border-border p-5">
          <PartyAvatar id={payeeId} name={summary?.display_name} size={46} />
          <div className="min-w-0 flex-1">
            <div className="text-[19px] font-bold tracking-tight">
              {summary?.display_name ?? (
                <span className="h-[22px] w-40 animate-pulse rounded bg-muted inline-block" />
              )}
            </div>
            <div className="mt-0.5 flex flex-wrap items-center gap-2 text-[13px] text-muted-foreground">
              {summary && (
                <>
                  <span className="rounded-[5px] bg-muted px-1.5 py-px text-[10px] font-bold">
                    {summary.payout_currency}
                  </span>
                  <StatusBadge
                    kind={summary.status as "owed" | "scheduled" | "settled"}
                  >
                    {partyStatusLabel(summary.status)}
                  </StatusBadge>
                </>
              )}
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* ── Collision warning ────────────────────────────────────────────── */}
        {summary?.collision && (
          <div className="mx-5 mt-4 flex items-start gap-2 rounded-[10px] border border-[hsl(28_70%_75%)] bg-[hsl(28_85%_95%)] px-3 py-2.5 text-[12px] leading-relaxed text-[hsl(28_55%_32%)] dark:border-[hsl(28_40%_30%)] dark:bg-[hsl(28_40%_18%)] dark:text-[hsl(28_70%_75%)]">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            This name may map to more than one person — split if needed.
          </div>
        )}

        {/* ── Body ─────────────────────────────────────────────────────────── */}
        {isLoading ? (
          <DrawerSkeleton />
        ) : isError || !summary ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 p-6 text-center">
            <AlertTriangle className="h-8 w-8 text-muted-foreground/50" />
            <p className="text-sm font-medium text-foreground">
              Couldn't load this party's details
            </p>
            <p className="max-w-[36ch] text-xs text-muted-foreground">
              Something went wrong fetching their royalty history. Please try again.
            </p>
            <Button variant="outline" size="sm" onClick={() => refetch()}>
              Retry
            </Button>
          </div>
        ) : (
          <div className="flex flex-1 flex-col gap-[22px] overflow-y-auto p-5 pb-7">
            {/* Balances */}
            {summary && (
              <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
                {(
                  [
                    {
                      label: "Earned",
                      base: summary.earned,
                      native: summary.earned_native,
                      hl: false,
                    },
                    {
                      label: "Paid",
                      base: summary.paid,
                      native: summary.paid_native,
                      hl: false,
                    },
                    {
                      label: "Outstanding",
                      base: summary.unpaid,
                      native: summary.unpaid_native,
                      hl: true,
                    },
                  ] as const
                ).map((b) => (
                  <div
                    key={b.label}
                    className={cn(
                      "rounded-xl border p-3",
                      b.hl
                        ? "border-[hsl(28_60%_55%/0.5)] bg-[hsl(28_80%_95%/0.5)] dark:bg-[hsl(28_35%_16%/0.5)]"
                        : "border-border",
                    )}
                  >
                    <div className="text-[11px] text-muted-foreground">
                      {b.label}
                    </div>
                    <div className="mt-1 font-mono text-[18px] font-bold tabular-nums tracking-tight">
                      {fmtMoney(b.base, base)}
                    </div>
                    {summary.payout_currency !== base && (
                      <div className="mt-0.5 font-mono text-[11.5px] text-muted-foreground">
                        {fmtMoney(b.native, summary.payout_currency)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Payout currency */}
            {summary && (
              <div>
                <SectionLabel>Payout currency</SectionLabel>
                <div className="flex items-center gap-3">
                  <Select
                    value={summary.payout_currency}
                    onValueChange={(c) =>
                      setCurrency.mutate({ id: payeeId, payout_currency: c })
                    }
                  >
                    <SelectTrigger className="h-10 w-[210px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="z-[70]">
                      {Object.values(CURRENCIES).map((c) => (
                        <SelectItem key={c.code} value={c.code}>
                          <span className="inline-flex items-center gap-2">
                            <span className="text-base leading-none">{c.flag}</span>
                            <span className="font-semibold">{c.code}</span>
                            <span className="text-muted-foreground">{c.name}</span>
                          </span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {setCurrency.isPending && (
                    <span className="text-[11.5px] text-muted-foreground">
                      Saving…
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Payment email */}
            {summary && (
              <div>
                <SectionLabel>Payment email</SectionLabel>
                <EmailEditor
                  key={summary.email ?? ""}
                  payeeId={payeeId}
                  currentEmail={summary.email}
                />
              </div>
            )}

            {/* Earnings by project */}
            <div>
              <SectionLabel right={String(projects.length)}>
                Earnings by project
              </SectionLabel>
              {projects.map((proj) => {
                const isProjOpen = openProj === proj.project_id;
                return (
                  <div
                    key={proj.project_id}
                    className="mt-2.5 overflow-hidden rounded-xl border border-border first:mt-0"
                  >
                    {/* Project row */}
                    <div
                      className="flex cursor-pointer items-center gap-2.5 p-3 hover:bg-muted/50"
                      onClick={() =>
                        setOpenProj(isProjOpen ? null : proj.project_id)
                      }
                    >
                      <span className="flex h-[30px] w-[30px] shrink-0 items-center justify-center rounded-lg bg-secondary text-primary">
                        <Folder className="h-[15px] w-[15px]" />
                      </span>
                      <span className="min-w-0 flex-1 text-[13.5px] font-semibold">
                        {proj.name}
                        <div className="mt-px text-[11.5px] font-normal text-muted-foreground">
                          {proj.statements.length} statement
                          {proj.statements.length !== 1 ? "s" : ""}
                        </div>
                      </span>
                      <ChevronRight
                        className={cn(
                          "h-4 w-4 text-muted-foreground transition-transform",
                          isProjOpen && "rotate-90",
                        )}
                      />
                    </div>

                    {/* Statements */}
                    {isProjOpen && (
                      <div className="border-t border-dashed border-border py-1 pl-[54px] pr-3">
                        {proj.statements.map((stmt) => {
                          const isStmtOpen =
                            openStmt === stmt.royalty_statement_id;
                          const label = periodLabel(stmt);
                          return (
                            <div
                              key={stmt.royalty_statement_id}
                              className="border-t border-dashed border-border first:border-t-0"
                            >
                              {/* Statement row */}
                              <div
                                className="flex cursor-pointer items-center justify-between gap-2.5 py-[7px] text-[12.5px]"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setOpenStmt(
                                    isStmtOpen
                                      ? null
                                      : stmt.royalty_statement_id,
                                  );
                                }}
                              >
                                <span className="flex min-w-0 items-center gap-1.5">
                                  <ChevronRight
                                    className={cn(
                                      "h-[13px] w-[13px] text-muted-foreground transition-transform",
                                      isStmtOpen && "rotate-90",
                                    )}
                                  />
                                  <span
                                    className={cn(
                                      "h-[7px] w-[7px] rounded-full shrink-0",
                                      stateDotColor(stmt.state),
                                    )}
                                  />
                                  <span>{label}</span>
                                  <span className="rounded-[5px] bg-muted px-1.5 py-px text-[10.5px] font-semibold text-muted-foreground">
                                    {stmt.lines.length} line
                                    {stmt.lines.length !== 1 ? "s" : ""}
                                  </span>
                                </span>
                                <span className="shrink-0 font-mono tabular-nums text-muted-foreground">
                                  {fmtMoney(stmt.unpaid, stmt.statement_currency)}
                                </span>
                              </div>

                              {/* Lines */}
                              {isStmtOpen && (
                                <div className="flex flex-col pb-2 pl-5">
                                  {stmt.lines.map((ln) => (
                                    <div
                                      key={ln.line_id}
                                      className="grid grid-cols-[1fr_auto] items-start gap-2.5 border-t border-dotted border-border py-1.5 text-xs first:border-t-0"
                                    >
                                      <span className="min-w-0">
                                        <div className="font-semibold">
                                          {ln.song_title}
                                        </div>
                                        {(ln.role || ln.royalty_type) && (
                                          <div className="mt-px text-[11px] text-muted-foreground">
                                            {[ln.role, ln.royalty_type]
                                              .filter(Boolean)
                                              .join(" · ")}
                                          </div>
                                        )}
                                      </span>
                                      <span className="font-mono tabular-nums text-muted-foreground whitespace-nowrap">
                                        {fmtMoney(
                                          ln.amount_owed,
                                          ln.statement_currency,
                                        )}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Split payee */}
            <div>
              <SectionLabel
                right={
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 text-[11.5px] font-semibold text-primary hover:underline"
                    onClick={() => setShowSplit((v) => !v)}
                  >
                    <Scissors className="h-[11px] w-[11px]" />
                    {showSplit ? "Cancel" : "Split payee…"}
                  </button>
                }
              >
                Split payee
              </SectionLabel>
              {showSplit && (
                <SplitPanel
                  payeeId={payeeId}
                  statements={projects.flatMap((p) => p.statements)}
                  onDone={() => {
                    setShowSplit(false);
                    onClose();
                  }}
                />
              )}
              {!showSplit && (
                <p className="text-[12px] text-muted-foreground">
                  Use split to separate lines that belong to different people
                  sharing the same name.
                </p>
              )}
            </div>

            {/* Payment history */}
            <div>
              <SectionLabel>Payment history</SectionLabel>
              {payouts.length === 0 ? (
                <p className="text-[12px] text-muted-foreground">
                  No payouts yet.
                </p>
              ) : (
                <div className="flex flex-col">
                  {payouts.map((payout, i) => {
                    const isPaid = payout.status === "paid";
                    const isDraft = payout.status === "draft";
                    const Ic = isPaid ? CheckCheck : isDraft ? Clock : Coins;
                    const dotTone = isPaid
                      ? "bg-[hsl(var(--pay-paid-bg))] text-[hsl(var(--pay-paid-fg))]"
                      : isDraft
                        ? "bg-[hsl(var(--pay-sched-bg))] text-[hsl(var(--pay-sched-fg))]"
                        : "bg-muted text-muted-foreground";
                    const dateStr = payout.paid_at || payout.created_at;
                    return (
                      <div
                        key={payout.id ?? i}
                        className="relative grid grid-cols-[26px_1fr_auto] gap-2.5 py-2.5 [&:not(:last-child)]:before:absolute [&:not(:last-child)]:before:left-3 [&:not(:last-child)]:before:top-[30px] [&:not(:last-child)]:before:bottom-[-10px] [&:not(:last-child)]:before:w-0.5 [&:not(:last-child)]:before:bg-border"
                      >
                        <span
                          className={cn(
                            "z-[1] flex h-[26px] w-[26px] items-center justify-center rounded-full",
                            dotTone,
                          )}
                        >
                          <Ic className="h-[13px] w-[13px]" />
                        </span>
                        <span className="min-w-0">
                          <div className="text-[13px] font-semibold">
                            {isPaid ? "Paid" : "Draft invoice"}
                          </div>
                          <div className="mt-0.5 text-[11.5px] text-muted-foreground">
                            {fmtDate(dateStr)}
                            {payout.note && ` · ${payout.note}`}
                          </div>
                        </span>
                        <span
                          className={cn(
                            "whitespace-nowrap text-right font-mono text-[13px] font-semibold tabular-nums",
                            isPaid
                              ? "text-[hsl(150_55%_38%)]"
                              : "text-[hsl(217_65%_50%)]",
                          )}
                        >
                          {fmtMoney(payout.total_amount, payout.pay_currency)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Earned vs Paid trend chart */}
            <div>
              <SectionLabel>Earnings trend</SectionLabel>
              <PayeeTrendChart payeeId={payeeId} base={base} />
            </div>
          </div>
        )}

        {/* ── Footer ───────────────────────────────────────────────────────── */}
        <div className="flex gap-2.5 border-t border-border bg-card p-3.5">
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
          <Button
            className="flex-1"
            disabled={!summary || summary.owed <= 0}
            onClick={() => onPayout([payeeId])}
          >
            <Send className="mr-1.5 h-4 w-4" />
            {summary && summary.owed > 0
              ? `Send ${fmtMoney(summary.owed_native, summary.payout_currency)} owed`
              : "Nothing to pay"}
          </Button>
        </div>
      </aside>
    </>
  );
}
