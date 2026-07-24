import { useRef, useEffect, useState } from "react";
import { ZoeContextPopover } from "@/components/zoe/ZoeContextPopover";
import { ContractSlideOver } from "@/components/zoe/ContractSlideOver";

// Inline SVGs — exact mockup shapes

const AttachIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48" />
  </svg>
);

const SendIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2.2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M5 12h14M13 6l6 6-6 6" />
  </svg>
);

const StopIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <rect x="3" y="3" width="18" height="18" rx="2" />
  </svg>
);

interface ZoeInputBarProps {
  inputMessage: string;
  onInputChange: (value: string) => void;
  // Currently unrendered. If this ever becomes a visible banner, it MUST NOT
  // render for credit-wall errors (hook sets `error` on 402s too) — the wall
  // already renders inline in the transcript, and a banner would double-show it.
  error: string;
  isStreaming: boolean;
  isAtLimit: boolean;
  selectedContracts: string[];
  /** Session-wide {id, file_name} so selected contracts from other artists still render as chips. */
  knownContracts: { id: string; file_name: string }[];
  contractMarkdowns: Record<string, string>;
  onDeselectContract: (id: string) => void;
  onSend: () => void;
  onStop: () => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  // Comparison-context popover props
  contextTree: {
    artists: { id: string; name: string; project_count: number }[];
    projects: { id: string; name: string; artist_id: string; doc_count: number }[];
  };
  checkedArtistIds: string[];
  setCheckedArtistIds: (updater: string[] | ((prev: string[]) => string[])) => void;
  checkedProjectIds: string[];
  setCheckedProjectIds: (updater: string[] | ((prev: string[]) => string[])) => void;
  projectDocuments: Record<
    string,
    { id: string; file_name: string; project_id: string; folder_category?: string; page_count?: number | null }[]
  >;
  onSelectedContractsChange: (contracts: string[]) => void;
  uploadModalOpen: boolean;
  onUploadModalOpenChange: (open: boolean) => void;
  onUploadComplete: () => void;
}

export function ZoeInputBar({
  inputMessage,
  onInputChange,
  isStreaming,
  isAtLimit,
  selectedContracts,
  knownContracts,
  onDeselectContract,
  onSend,
  onStop,
  onKeyDown,
  // comparison-context popover
  contextTree,
  checkedArtistIds,
  setCheckedArtistIds,
  checkedProjectIds,
  setCheckedProjectIds,
  projectDocuments,
  onSelectedContractsChange,
  uploadModalOpen,
  onUploadModalOpenChange,
  onUploadComplete,
}: ZoeInputBarProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Slide-over for clicking attachment chips
  const [slideOverOpen, setSlideOverOpen] = useState(false);
  const [slideOverId, setSlideOverId] = useState<string | null>(null);
  const [slideOverName, setSlideOverName] = useState<string | null>(null);

  const openSlideOver = (id: string, name: string) => {
    setSlideOverId(id);
    setSlideOverName(name);
    setSlideOverOpen(true);
  };

  // Auto-grow textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
  }, [inputMessage]);

  // Resolve from the session-wide map so selected contracts from OTHER artists still render as chips.
  const selectedContractInfos = selectedContracts
    .map((id) => knownContracts.find((c) => c.id === id))
    .filter((c): c is { id: string; file_name: string } => c !== undefined);

  const placeholder = isAtLimit
    ? "Conversation limit reached. Please refresh the page."
    : selectedContracts.length > 0
    ? "Ask about your selected contracts…"
    : "Ask a music-business question, or select an artist/contract for specific help…";

  const noteText = isAtLimit
    ? "Conversation limit reached. Please refresh the page to continue."
    : selectedContracts.length > 0
    ? `Answering from ${selectedContracts.length} selected contract${selectedContracts.length > 1 ? "s" : ""} · Zoe can be wrong — verify against the cited clauses.`
    : "Answering from music-business knowledge · select contracts for advice specific to your deals";

  return (
    <>
      <div className="composer-wrap">
        <div className="composer">
          {/* Attachment chips */}
          {selectedContractInfos.length > 0 && (
            <div className="attachments">
              {selectedContractInfos.map((c) => (
                <span
                  key={c.id}
                  className="attach"
                  onClick={() => openSlideOver(c.id, c.file_name)}
                  title={`View ${c.file_name}`}
                >
                  <AttachIcon />
                  <span
                    style={{
                      maxWidth: 180,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {c.file_name}
                  </span>
                  <span
                    className="x"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeselectContract(c.id);
                    }}
                    title={`Remove ${c.file_name}`}
                  >
                    ✕
                  </span>
                </span>
              ))}
            </div>
          )}

          {/* Input box */}
          <div className="input-box">
            {/* Attach / context button */}
            <ZoeContextPopover
              contextTree={contextTree}
              checkedArtistIds={checkedArtistIds}
              setCheckedArtistIds={setCheckedArtistIds}
              checkedProjectIds={checkedProjectIds}
              setCheckedProjectIds={setCheckedProjectIds}
              projectDocuments={projectDocuments}
              knownContracts={knownContracts}
              selectedContracts={selectedContracts}
              onSelectedContractsChange={onSelectedContractsChange}
              uploadModalOpen={uploadModalOpen}
              onUploadModalOpenChange={onUploadModalOpenChange}
              onUploadComplete={onUploadComplete}
            >
              <button
                className="icon-btn"
                title="Select contracts"
                type="button"
                disabled={isAtLimit}
              >
                <AttachIcon />
              </button>
            </ZoeContextPopover>

            <textarea
              ref={textareaRef}
              className="zoe-textarea"
              rows={1}
              placeholder={placeholder}
              value={inputMessage}
              onChange={(e) => onInputChange(e.target.value)}
              onKeyDown={onKeyDown}
              disabled={isAtLimit}
            />

            {isStreaming ? (
              <button
                className="send stop"
                onClick={onStop}
                title="Stop generating"
                type="button"
              >
                <StopIcon />
              </button>
            ) : (
              <button
                className="send"
                onClick={onSend}
                disabled={!inputMessage.trim() || isAtLimit}
                title="Send"
                type="button"
              >
                <SendIcon />
              </button>
            )}
          </div>

          <p className="composer-note">{noteText}</p>
        </div>
      </div>

      {/* Slide-over for attachment chip clicks */}
      <ContractSlideOver
        open={slideOverOpen}
        onOpenChange={setSlideOverOpen}
        contractId={slideOverId}
        contractName={slideOverName}
      />
    </>
  );
}
