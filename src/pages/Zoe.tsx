import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { RequireFeature } from "@/components/paywall/RequireFeature";
import { useAnalytics } from "@/hooks/useAnalytics";
import { ZoeChatMessages } from "@/components/zoe/ZoeChatMessages";
import { ZoeInputBar } from "@/components/zoe/ZoeInputBar";
import { ZoeContextPopover } from "@/components/zoe/ZoeContextPopover";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";
import { useZoeData } from "@/hooks/useZoeData";
import "@/components/zoe/zoe-chat.css";

// ── Inline SVGs matching the mockup ──────────────────────────────────────

const MusicBadgeIcon = () => (
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

const NewChatIcon = () => (
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

const Zoe = () => {
  const {
    artists,
    projects,
    contracts,
    knownContracts,
    selectedArtist,
    setSelectedArtist,
    selectedProject,
    setSelectedProject,
    selectedContracts,
    setSelectedContracts,
    selectedArtistName,
    selectedProjectName,
    messagesEndRef,
    contractMarkdowns,
    contextTree,
    checkedArtistIds,
    setCheckedArtistIds,
    checkedProjectIds,
    setCheckedProjectIds,
    projectDocuments,
    messages,
    isStreaming,
    isAtLimit,
    inputMessage,
    setInputMessage,
    copiedMessageId,
    uploadModalOpen,
    setUploadModalOpen,
    isCreateProjectOpen,
    setIsCreateProjectOpen,
    newProjectNameInput,
    setNewProjectNameInput,
    isCreatingProject,
    showReloadDialog,
    setShowReloadDialog,
    stopGeneration,
    handleNewConversation,
    handleUploadComplete,
    handleCreateProject,
    handleSendMessage,
    handleQuickAction,
    handleAssistantQuickAction,
    handleRetry,
    handleCopyMessage,
    handleKeyDown,
    error,
  } = useZoeData();

  const navigate = useNavigate();
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(TOOL_CONFIGS.zoe, {
    onComplete: () => markToolCompleted("zoe"),
  });

  const { captureToolOpened } = useAnalytics();
  useEffect(() => {
    captureToolOpened("zoe");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!onboardingLoading && !statuses.zoe && walkthrough.phase === "idle") {
      const timer = setTimeout(() => walkthrough.startModal(), 500);
      return () => clearTimeout(timer);
    }
  }, [onboardingLoading, statuses.zoe]); // eslint-disable-line react-hooks/exhaustive-deps

  // Build the context pill label
  const hasContext = !!selectedArtist;
  const contractCount = selectedContracts.length;

  return (
    <RequireFeature feature="zoe">
      <div className="zoe-chat app">
        {/* ── Topbar ─────────────────────────────────────────────────── */}
        <header className="topbar">
          <div className="topbar-inner">
            {/* Brand + back */}
            <div className="brand">
              <button
                className="topbar-back"
                onClick={() => navigate("/dashboard")}
                title="Back to dashboard"
                aria-label="Back to dashboard"
              >
                <ArrowLeft />
              </button>
              <button className="brand-link" onClick={() => navigate("/dashboard")} title="Back to dashboard">
                <div className="brand-badge">
                  <MusicBadgeIcon />
                </div>
                <div className="brand-name">
                  Zoe <span>· Contract Analyst</span>
                </div>
              </button>
            </div>

            {/* Right controls */}
            <div className="topbar-right">
              {/* Context pill — opens the context popover */}
              <ZoeContextPopover
                contextTree={contextTree}
                checkedArtistIds={checkedArtistIds}
                setCheckedArtistIds={setCheckedArtistIds}
                checkedProjectIds={checkedProjectIds}
                setCheckedProjectIds={setCheckedProjectIds}
                projectDocuments={projectDocuments}
                knownContracts={knownContracts}
                selectedContracts={selectedContracts}
                onSelectedContractsChange={setSelectedContracts}
                uploadModalOpen={uploadModalOpen}
                onUploadModalOpenChange={setUploadModalOpen}
                onUploadComplete={handleUploadComplete}
              >
                <button className="ctx-pill" title="Select context" data-walkthrough="zoe-context">
                  {contractCount > 0 ? (
                    <>
                      <span className="live" />
                      <span>
                        <b>{contractCount}</b> contract{contractCount > 1 ? "s" : ""}
                      </span>
                      {checkedArtistIds.length > 1 && (
                        <>
                          <span className="ctx-dot" />
                          <span>
                            <b>{checkedArtistIds.length}</b> artists
                          </span>
                        </>
                      )}
                    </>
                  ) : (
                    <>
                      <span className="ctx-dot" />
                      <span>Select contracts to compare</span>
                    </>
                  )}
                </button>
              </ZoeContextPopover>

              {/* New chat — always visible, icon-only on mobile */}
              <button
                className="btn-ghost"
                data-walkthrough="zoe-newchat"
                onClick={handleNewConversation}
                title="New chat"
              >
                <NewChatIcon />
                <span className="label-hide">New chat</span>
              </button>

              {/* Help button */}
              <ToolHelpButton onClick={walkthrough.replay} />
            </div>
          </div>
        </header>

        {/* ── Thread ─────────────────────────────────────────────────── */}
        <ZoeChatMessages
          messages={messages}
          isStreaming={isStreaming}
          selectedArtist={selectedArtist}
          selectedProject={selectedProject}
          selectedContracts={selectedContracts}
          contracts={knownContracts}
          contractMarkdowns={contractMarkdowns}
          copiedMessageId={copiedMessageId}
          messagesEndRef={messagesEndRef}
          onQuickAction={handleQuickAction}
          onAssistantQuickAction={handleAssistantQuickAction}
          onRetry={handleRetry}
          onCopyMessage={handleCopyMessage}
        />

        {/* ── Composer ───────────────────────────────────────────────── */}
        <div data-walkthrough="zoe-chat">
          <ZoeInputBar
            inputMessage={inputMessage}
            onInputChange={setInputMessage}
            error={error}
            isStreaming={isStreaming}
            isAtLimit={isAtLimit}
            selectedContracts={selectedContracts}
            knownContracts={knownContracts}
            contractMarkdowns={contractMarkdowns}
            onDeselectContract={(id) =>
              setSelectedContracts((prev) => prev.filter((c) => c !== id))
            }
            onSend={handleSendMessage}
            onStop={stopGeneration}
            onKeyDown={handleKeyDown}
            contextTree={contextTree}
            checkedArtistIds={checkedArtistIds}
            setCheckedArtistIds={setCheckedArtistIds}
            checkedProjectIds={checkedProjectIds}
            setCheckedProjectIds={setCheckedProjectIds}
            projectDocuments={projectDocuments}
            onSelectedContractsChange={setSelectedContracts}
            uploadModalOpen={uploadModalOpen}
            onUploadModalOpenChange={setUploadModalOpen}
            onUploadComplete={handleUploadComplete}
          />
        </div>

        {/* ── Dialogs ────────────────────────────────────────────────── */}
        <AlertDialog open={showReloadDialog} onOpenChange={setShowReloadDialog}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Session Refresh Required</AlertDialogTitle>
              <AlertDialogDescription>
                {isAtLimit
                  ? "You've reached the conversation limit. Please refresh the page to start a fresh session with Zoe."
                  : "The conversation context was reset. Please reload the page to start a fresh session with Zoe."}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogAction onClick={() => window.location.reload()}>
                Refresh Page
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        <ToolIntroModal
          config={TOOL_CONFIGS.zoe}
          isOpen={walkthrough.phase === "modal"}
          onStartTour={walkthrough.startSpotlight}
          onSkip={walkthrough.skip}
        />
        <WalkthroughProvider
          isActive={walkthrough.phase === "spotlight"}
          currentStep={walkthrough.currentStep}
          currentStepIndex={walkthrough.visibleStepIndex}
          totalSteps={walkthrough.totalSteps}
          onNext={walkthrough.next}
          onSkip={walkthrough.skip}
        />
      </div>
    </RequireFeature>
  );
};

export default Zoe;
