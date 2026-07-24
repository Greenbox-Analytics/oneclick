import { RefObject, useState } from "react";
import { useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Lock } from "lucide-react";
import type { Message, AssistantQuickAction } from "@/hooks/useStreamingChat";
import { ContractSlideOver } from "@/components/zoe/ContractSlideOver";
import { PaywallCard } from "@/components/paywall/PaywallCard";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useEntitlements } from "@/hooks/useEntitlements";

// File icon SVG (inline, matches mockup)
const FileIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <path d="M14 2v6h6" />
  </svg>
);

// Copy icon
const CopyIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <rect width="14" height="14" x="8" y="8" rx="2" />
    <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
  </svg>
);

// Regenerate icon
const RegenerateIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
    <path d="M21 3v5h-5" />
  </svg>
);

// Music note brand icon (matches topbar mockup)
const MusicIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M9 18V5l12-2v13" />
    <circle cx="6" cy="18" r="3" />
    <circle cx="18" cy="16" r="3" />
  </svg>
);

interface ContractInfo {
  id: string;
  file_name: string;
}

interface ZoeChatMessagesProps {
  messages: Message[];
  isStreaming: boolean;
  selectedArtist: string;
  selectedProject: string;
  selectedContracts: string[];
  contracts: ContractInfo[];
  contractMarkdowns: Record<string, string>;
  copiedMessageId: string | null;
  messagesEndRef: RefObject<HTMLDivElement>;
  onQuickAction: (question: string) => void;
  onAssistantQuickAction: (action: AssistantQuickAction) => void;
  onRetry: () => void;
  onCopyMessage: (content: string, messageId: string) => void;
  onOpenContextPopover?: () => void;
  isMobile?: boolean;
}

export function ZoeChatMessages({
  messages,
  isStreaming,
  selectedArtist,
  selectedProject,
  selectedContracts,
  contracts,
  contractMarkdowns,
  copiedMessageId,
  messagesEndRef,
  onQuickAction,
  onAssistantQuickAction,
  onRetry,
  onCopyMessage,
  isMobile = false,
}: ZoeChatMessagesProps) {
  // Slide-over state
  const [slideOverOpen, setSlideOverOpen] = useState(false);
  const [slideOverId, setSlideOverId] = useState<string | null>(null);
  const [slideOverName, setSlideOverName] = useState<string | null>(null);
  const [slideOverPage, setSlideOverPage] = useState<number | null>(null);

  // Credit-wall org identity is two-source (licensing follow-ups Task 1): a
  // 402's own `detail.managedByOrg` covers the enriched shape, and this
  // billingContext fallback covers any 402 that arrives without it (legacy /
  // degraded shapes) — see CreditWallCard below.
  const { data: entitlements } = useEntitlements();
  const billingContextIsOrg = entitlements?.billingContext?.type === "org";

  const openSlideOver = (id: string, name: string, page?: number | null) => {
    setSlideOverId(id);
    setSlideOverName(name);
    setSlideOverPage(page ?? null);
    setSlideOverOpen(true);
  };

  // Build a map from file_name -> contract id for source chip lookup
  const contractByName: Record<string, ContractInfo> = {};
  for (const c of contracts) {
    contractByName[c.file_name] = c;
  }

  return (
    <>
      <div className="scroll">
        <div className="thread">
          {messages.length === 0 ? (
            <EmptyState
              selectedArtist={selectedArtist}
              selectedProject={selectedProject}
              selectedContracts={selectedContracts}
              isStreaming={isStreaming}
              isMobile={isMobile}
              onQuickAction={onQuickAction}
            />
          ) : (
            messages.map((message, idx) => {
              if (message.role === "system") {
                return (
                  <div key={message.id} className="turn-system">
                    <div className="turn-system-line" />
                    <span className="turn-system-text">
                      {message.content.replace(/^---\s*|\s*---$/g, "")}
                    </span>
                    <div className="turn-system-line" />
                  </div>
                );
              }

              if (message.role === "user") {
                return (
                  <div key={message.id} className="turn-user">
                    <div className="user-msg">{message.content}</div>
                  </div>
                );
              }

              // assistant
              const isLast = idx === messages.length - 1;
              const isThinking = message.isStreaming && !message.content;
              const isCreditWall = message.confidence === "credit_wall";

              // Which contracts to show as sources for THIS message.
              // Prefer backend-cited sources (with page jumps); otherwise fall back to the
              // contracts pinned to this turn (message.contractIds) — NOT the live selection,
              // so older messages keep their own contracts after the user switches selection.
              const sourcesToShow: { id: string; file_name: string; page?: number }[] = [];
              if (message.sources && message.sources.length > 0) {
                for (const src of message.sources) {
                  // Prefer the authoritative contract id from the backend; the page-jump endpoint
                  // is by-id, so this avoids ever passing a filename as the id. Fall back to name
                  // resolution only for older messages that predate contract_id in the payload.
                  const id = src.contract_id || contractByName[src.contract_file]?.id;
                  if (id && !sourcesToShow.find((s) => s.id === id)) {
                    sourcesToShow.push({ id, file_name: src.contract_file, page: src.page_number });
                  }
                }
              } else if (!message.isStreaming && message.content && message.contractIds?.length) {
                for (const id of message.contractIds) {
                  const c = contracts.find((x) => x.id === id);
                  if (c) sourcesToShow.push({ id: c.id, file_name: c.file_name });
                }
              }

              return (
                <div key={message.id} className="turn-zoe">
                  <div className="zoe-label">Zoe</div>
                  <div className="answer">
                    {isCreditWall ? (
                      <CreditWallCard
                        reason={message.content}
                        detail={message.detail}
                        billingContextIsOrg={billingContextIsOrg}
                      />
                    ) : isThinking ? (
                      <div className="thinking">
                        <div className="thinking-dots">
                          <span />
                          <span />
                          <span />
                        </div>
                        Thinking…
                      </div>
                    ) : (
                      <>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {message.content}
                        </ReactMarkdown>
                        {message.isStreaming && <span className="cursor" />}
                      </>
                    )}

                    {/* Quick actions embedded in the answer */}
                    {!isCreditWall &&
                      !message.isStreaming &&
                      message.quickActions &&
                      message.quickActions.length > 0 && (
                        <div className="quick-actions" style={{ marginTop: 12 }}>
                          {message.quickActions.map((action) => (
                            <button
                              key={action.id}
                              className="qa-btn"
                              onClick={() => onAssistantQuickAction(action)}
                              disabled={isStreaming}
                            >
                              {action.label}
                            </button>
                          ))}
                        </div>
                      )}

                    {!isCreditWall &&
                      !message.isStreaming &&
                      message.showQuickActions &&
                      (!message.quickActions || message.quickActions.length === 0) && (
                        <div className="quick-actions" style={{ marginTop: 12 }}>
                          <button
                            className="qa-btn"
                            onClick={() =>
                              onQuickAction("What are the royalty splits in this contract?")
                            }
                            disabled={isStreaming}
                          >
                            Royalty Splits
                          </button>
                          <button
                            className="qa-btn"
                            onClick={() =>
                              onQuickAction("Who are the parties involved in this contract?")
                            }
                            disabled={isStreaming}
                          >
                            Involved Parties
                          </button>
                          <button
                            className="qa-btn"
                            onClick={() =>
                              onQuickAction("What are the payment terms in this contract?")
                            }
                            disabled={isStreaming}
                          >
                            Payment Terms
                          </button>
                        </div>
                      )}

                    {/* Sources */}
                    {!isCreditWall && !message.isStreaming && sourcesToShow.length > 0 && (
                      <div className="sources">
                        <div className="sources-label">Sources</div>
                        <div className="source-chips">
                          {sourcesToShow.map((c) => (
                            <span
                              key={c.id}
                              className="chip"
                              onClick={() => openSlideOver(c.id, c.file_name, c.page)}
                              title={c.file_name}
                            >
                              <FileIcon />
                              {c.file_name}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Answer actions — hover-revealed. Not shown on a credit-wall
                      card: there's nothing to copy, and regenerating just re-hits
                      the same wall. */}
                  {!isCreditWall && !message.isStreaming && message.content && (
                    <div className="answer-actions">
                      <button
                        className="act"
                        onClick={() => onCopyMessage(message.content, message.id)}
                        title="Copy"
                      >
                        <CopyIcon />
                        {copiedMessageId === message.id ? "Copied" : "Copy"}
                      </button>
                      {isLast && (
                        <button className="act" onClick={onRetry} title="Regenerate">
                          <RegenerateIcon />
                          Regenerate
                        </button>
                      )}
                    </div>
                  )}
                </div>
              );
            })
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <ContractSlideOver
        open={slideOverOpen}
        onOpenChange={setSlideOverOpen}
        contractId={slideOverId}
        contractName={slideOverName}
        page={slideOverPage}
      />
    </>
  );
}

// ── Credit wall card ──────────────────────────────────────────────────────
// Renders in place of the answer when a Zoe turn was denied by a credit-402
// (licensing follow-ups Task 1). Reuses PaywallCard for the two branches its
// props already fit (managedByOrg's "Request credits" branch, and the
// default upgrade branch) rather than duplicating that CTA/analytics logic;
// the pay-per-use and plain-reason branches have no PaywallCard equivalent,
// so they're small local cards in the same visual language (Lock badge,
// centered copy, Card shell).

function formatResetDate(iso?: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleDateString(undefined, { month: "long", day: "numeric" });
}

function CreditWallCard({
  reason,
  detail,
  billingContextIsOrg,
}: {
  /** Always a non-empty human-readable reason — the hook guarantees a
   * fallback string even for legacy plain-string 402s where `detail` is
   * undefined, so this card is never blank. */
  reason: string;
  detail?: Message["detail"];
  billingContextIsOrg: boolean;
}) {
  const navigate = useNavigate();

  // Two-source org identity (review round 2, finding 2): the enriched detail
  // flag covers the primary path; the billingContext fallback covers any 402
  // that arrives without it (legacy / degraded shapes).
  const managedByOrg = detail?.managedByOrg === true || billingContextIsOrg;
  const upgradeRequired = !managedByOrg && detail?.upgradeRequired === true;
  const overageAvailable = !managedByOrg && !upgradeRequired && detail?.overageAvailable === true;
  const resetDate = formatResetDate(detail?.resetDate);
  const resetNote = resetDate ? (
    <p className="text-xs text-muted-foreground mt-2">Your credits reset on {resetDate}.</p>
  ) : null;

  if (managedByOrg || upgradeRequired) {
    return (
      <div>
        <PaywallCard
          variant="inline"
          reason={reason}
          managedByOrg={managedByOrg}
          requestUrl={detail?.requestUrl}
          ownerCanUnlink={detail?.ownerCanUnlink}
          projectId={detail?.projectId}
          projectName={detail?.projectName}
        />
        {resetNote}
      </div>
    );
  }

  return (
    <Card className="p-6 border-0 shadow-none">
      <div className="flex flex-col items-center text-center gap-3">
        <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
          <Lock className="w-5 h-5 text-muted-foreground" />
        </div>
        <h2 className="text-xl font-semibold">
          {overageAvailable ? "You're out of included credits" : "Out of credits"}
        </h2>
        <p className="text-muted-foreground text-sm max-w-sm">{reason}</p>
        {overageAvailable && (
          <Button onClick={() => navigate("/profile")} className="mt-2">
            Enable pay-per-use
          </Button>
        )}
      </div>
      {resetNote}
    </Card>
  );
}

// ── Empty state ────────────────────────────────────────────────────────────

function EmptyState({
  selectedArtist,
  selectedProject,
  selectedContracts,
  isStreaming,
  isMobile,
  onQuickAction,
}: {
  selectedArtist: string;
  selectedProject: string;
  selectedContracts: string[];
  isStreaming: boolean;
  isMobile: boolean;
  onQuickAction: (q: string) => void;
}) {
  const sub = !selectedArtist
    ? isMobile
      ? "Ask me anything about the music business — or tap the attach button to pick an artist and contract for specific help."
      : "Ask me anything about the music business — or select an artist and contract for specific help."
    : !selectedProject
    ? "Select a project and choose contracts to start asking questions."
    : selectedContracts.length === 0
    ? "Select one or more contracts to get started. I'll answer questions based on your selected contracts."
    : "I'm ready to answer questions about your selected contracts. Ask me about royalty splits, payment terms, parties involved, and more.";

  const hint =
    selectedContracts.length > 0
      ? "Zoe's answers are based on your selected contracts."
      : "Zoe draws on general music-business knowledge. Select contracts for deal-specific answers.";

  return (
    <div className="empty-state">
      <div className="empty-badge">
        <MusicIcon />
      </div>
      <p className="empty-title">Hi, I'm Zoe!</p>
      <p className="empty-sub">{sub}</p>
      <p className="empty-hint">{hint}</p>

      {selectedArtist && selectedProject && selectedContracts.length > 0 && (
        <div className="quick-actions">
          <button
            className="qa-btn"
            onClick={() =>
              onQuickAction("What are the streaming royalty splits in this contract?")
            }
            disabled={isStreaming}
          >
            Royalty Splits
          </button>
          <button
            className="qa-btn"
            onClick={() => onQuickAction("What are the payment terms in this contract?")}
            disabled={isStreaming}
          >
            Payment Terms
          </button>
          <button
            className="qa-btn"
            onClick={() =>
              onQuickAction("Who are the parties involved in this contract?")
            }
            disabled={isStreaming}
          >
            Involved Parties
          </button>
        </div>
      )}
    </div>
  );
}
