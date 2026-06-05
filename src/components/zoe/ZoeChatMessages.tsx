import { RefObject, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message, AssistantQuickAction } from "@/hooks/useStreamingChat";
import { ContractSlideOver } from "@/components/zoe/ContractSlideOver";

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

              // Determine which selected contracts to show as sources for this message.
              // Use message.sources if available, else show all selected contracts.
              const sourcesToShow: { id: string; file_name: string; page?: number }[] = [];
              if (message.sources && message.sources.length > 0) {
                for (const src of message.sources) {
                  const c = contractByName[src.contract_file];
                  if (c && !sourcesToShow.find((s) => s.id === c.id)) {
                    sourcesToShow.push({ id: c.id, file_name: c.file_name, page: src.page_number });
                  }
                }
              } else if (!message.isStreaming && message.content && selectedContracts.length > 0) {
                for (const id of selectedContracts) {
                  const c = contracts.find((x) => x.id === id);
                  if (c) sourcesToShow.push({ id: c.id, file_name: c.file_name });
                }
              }

              return (
                <div key={message.id} className="turn-zoe">
                  <div className="zoe-label">Zoe</div>
                  <div className="answer">
                    {isThinking ? (
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
                    {!message.isStreaming &&
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

                    {!message.isStreaming &&
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
                    {!message.isStreaming && sourcesToShow.length > 0 && (
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

                  {/* Answer actions — hover-revealed */}
                  {!message.isStreaming && message.content && (
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
