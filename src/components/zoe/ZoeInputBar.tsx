import { useRef, useEffect, useState } from "react";
import { ZoeContextPopover } from "@/components/zoe/ZoeContextPopover";
import { ContractSlideOver } from "@/components/zoe/ContractSlideOver";
import type { Artist, Project, Contract } from "@/components/zoe/types";

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
  error: string;
  isStreaming: boolean;
  isAtLimit: boolean;
  selectedArtist: string;
  selectedProject: string;
  selectedContracts: string[];
  contracts: Contract[];
  contractMarkdowns: Record<string, string>;
  onDeselectContract: (id: string) => void;
  onSend: () => void;
  onStop: () => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  // Context popover props
  artists: Artist[];
  projects: Project[];
  onArtistChange: (value: string) => void;
  onProjectChange: (value: string) => void;
  onSelectedContractsChange: (contracts: string[]) => void;
  uploadModalOpen: boolean;
  onUploadModalOpenChange: (open: boolean) => void;
  onUploadComplete: () => void;
  isCreateProjectOpen: boolean;
  onCreateProjectOpenChange: (open: boolean) => void;
  newProjectNameInput: string;
  onNewProjectNameInputChange: (value: string) => void;
  isCreatingProject: boolean;
  onCreateProject: () => void;
}

export function ZoeInputBar({
  inputMessage,
  onInputChange,
  isStreaming,
  isAtLimit,
  selectedProject,
  selectedContracts,
  contracts,
  contractMarkdowns,
  onDeselectContract,
  onSend,
  onStop,
  onKeyDown,
  // context popover
  artists,
  selectedArtist,
  projects,
  onArtistChange,
  onProjectChange,
  onSelectedContractsChange,
  uploadModalOpen,
  onUploadModalOpenChange,
  onUploadComplete,
  isCreateProjectOpen,
  onCreateProjectOpenChange,
  newProjectNameInput,
  onNewProjectNameInputChange,
  isCreatingProject,
  onCreateProject,
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

  const selectedContractInfos = selectedContracts
    .map((id) => contracts.find((c) => c.id === id))
    .filter((c): c is Contract => c !== undefined);

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
              artists={artists}
              selectedArtist={selectedArtist}
              onArtistChange={onArtistChange}
              projects={projects}
              selectedProject={selectedProject}
              onProjectChange={onProjectChange}
              contracts={contracts}
              selectedContracts={selectedContracts}
              onSelectedContractsChange={onSelectedContractsChange}
              uploadModalOpen={uploadModalOpen}
              onUploadModalOpenChange={onUploadModalOpenChange}
              onUploadComplete={onUploadComplete}
              isCreateProjectOpen={isCreateProjectOpen}
              onCreateProjectOpenChange={onCreateProjectOpenChange}
              newProjectNameInput={newProjectNameInput}
              onNewProjectNameInputChange={onNewProjectNameInputChange}
              isCreatingProject={isCreatingProject}
              onCreateProject={onCreateProject}
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
