import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Music, PanelLeftClose, PanelLeft, RefreshCw, BookOpen } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import { ZoeChatMessages } from "@/components/zoe/ZoeChatMessages";
import { ZoeInputBar } from "@/components/zoe/ZoeInputBar";
import { ZoeDocumentPanel } from "@/components/zoe/ZoeDocumentPanel";
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

const Zoe = () => {
  const navigate = useNavigate();

  const {
    artists,
    projects,
    contracts,
    selectedArtist,
    setSelectedArtist,
    selectedProject,
    setSelectedProject,
    selectedContracts,
    setSelectedContracts,
    selectedArtistName,
    selectedProjectName,
    sidebarOpen,
    setSidebarOpen,
    sidebarWidth,
    isResizing,
    sidebarRef,
    messagesEndRef,
    contractsOpen,
    setContractsOpen,
    messages,
    isStreaming,
    error,
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
    deleteDialogOpen,
    setDeleteDialogOpen,
    contractToDelete,
    deleting,
    showReloadDialog,
    setShowReloadDialog,
    sharedWorks,
    isLoadingSharedWorks,
    sharedWorksOpen,
    setSharedWorksOpen,
    selectedSharedWork,
    setSelectedSharedWork,
    sharedWorkFiles,
    setSharedWorkFiles,
    loadingWorkFiles,
    stopGeneration,
    handleNewConversation,
    handleUploadComplete,
    handleCreateProject,
    handleDeleteClick,
    handleDeleteConfirm,
    handleSendMessage,
    handleQuickAction,
    handleAssistantQuickAction,
    handleRetry,
    handleCopyMessage,
    handleKeyDown,
    handleMouseDown,
  } = useZoeData();

  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(TOOL_CONFIGS.zoe, {
    onComplete: () => markToolCompleted("zoe"),
  });

  useEffect(() => {
    if (!onboardingLoading && !statuses.zoe && walkthrough.phase === "idle") {
      const timer = setTimeout(() => walkthrough.startModal(), 500);
      return () => clearTimeout(timer);
    }
  }, [onboardingLoading, statuses.zoe]);

  return (
    <div className="h-screen flex flex-col bg-background overflow-hidden">
      {/* Header */}
      <header className="border-b border-border bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/60 flex-shrink-0 z-10">
        <div className="px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="h-9 w-9"
            >
              {sidebarOpen ? <PanelLeftClose className="h-5 w-5" /> : <PanelLeft className="h-5 w-5" />}
            </Button>
            <div className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity" onClick={() => navigate("/dashboard")}>
              <div className="w-9 h-9 rounded-lg bg-primary flex items-center justify-center">
                <Music className="w-5 h-5 text-primary-foreground" />
              </div>
              <div className="hidden sm:block">
                <h1 className="text-lg font-bold text-foreground leading-none">Zoe AI</h1>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {selectedArtist && (
              <Badge variant="secondary" className="hidden sm:flex items-center gap-1.5 text-xs">
                <span className="max-w-[100px] truncate">{selectedArtistName}</span>
                {selectedProject && (
                  <>
                    <span className="text-muted-foreground">•</span>
                    <span className="max-w-[100px] truncate">{selectedProjectName}</span>
                  </>
                )}
                {selectedContracts.length > 0 && (
                  <>
                    <span className="text-muted-foreground">•</span>
                    <span>{selectedContracts.length} docs</span>
                  </>
                )}
              </Badge>
            )}
            {messages.length > 0 && (
              <Button data-walkthrough="zoe-newchat" variant="outline" onClick={handleNewConversation} size="sm" className="gap-2">
                <RefreshCw className="w-4 h-4" />
                <span className="hidden sm:inline">New Chat</span>
              </Button>
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate("/docs")}
              title="Documentation"
              className="text-muted-foreground hover:text-foreground"
            >
              <BookOpen className="w-4 h-4" />
            </Button>
            <ToolHelpButton onClick={walkthrough.replay} />
            <Button variant="outline" onClick={() => navigate("/tools")} size="sm" className="gap-2">
              <ArrowLeft className="w-4 h-4" />
              <span className="hidden sm:inline">Back to Tools</span>
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        <ZoeDocumentPanel
          sidebarRef={sidebarRef}
          sidebarOpen={sidebarOpen}
          sidebarWidth={sidebarWidth}
          isResizing={isResizing}
          onMouseDown={handleMouseDown}
          artists={artists}
          selectedArtist={selectedArtist}
          onArtistChange={(v) => { setSelectedArtist(v); setSelectedProject(''); setSelectedContracts([]); }}
          projects={projects}
          selectedProject={selectedProject}
          onProjectChange={(v) => { setSelectedProject(v); setSelectedContracts([]); }}
          contracts={contracts}
          selectedContracts={selectedContracts}
          onSelectedContractsChange={setSelectedContracts}
          contractsOpen={contractsOpen}
          onContractsOpenChange={setContractsOpen}
          sharedWorks={sharedWorks}
          isLoadingSharedWorks={isLoadingSharedWorks}
          sharedWorksOpen={sharedWorksOpen}
          onSharedWorksOpenChange={setSharedWorksOpen}
          selectedSharedWork={selectedSharedWork}
          onSelectedSharedWorkChange={setSelectedSharedWork}
          sharedWorkFiles={sharedWorkFiles}
          loadingWorkFiles={loadingWorkFiles}
          onSharedWorkFilesReset={() => setSharedWorkFiles([])}
          uploadModalOpen={uploadModalOpen}
          onUploadModalOpenChange={setUploadModalOpen}
          onUploadComplete={handleUploadComplete}
          isCreateProjectOpen={isCreateProjectOpen}
          onCreateProjectOpenChange={setIsCreateProjectOpen}
          newProjectNameInput={newProjectNameInput}
          onNewProjectNameInputChange={setNewProjectNameInput}
          isCreatingProject={isCreatingProject}
          onCreateProject={handleCreateProject}
          deleteDialogOpen={deleteDialogOpen}
          onDeleteDialogOpenChange={setDeleteDialogOpen}
          contractToDelete={contractToDelete}
          deleting={deleting}
          onDeleteClick={handleDeleteClick}
          onDeleteConfirm={handleDeleteConfirm}
        />

        {/* Chat Area */}
        <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
          <ZoeChatMessages
            messages={messages}
            isStreaming={isStreaming}
            selectedArtist={selectedArtist}
            selectedProject={selectedProject}
            selectedContracts={selectedContracts}
            copiedMessageId={copiedMessageId}
            messagesEndRef={messagesEndRef}
            onQuickAction={handleQuickAction}
            onAssistantQuickAction={handleAssistantQuickAction}
            onRetry={handleRetry}
            onCopyMessage={handleCopyMessage}
          />
          <div data-walkthrough="zoe-chat">
            <ZoeInputBar
              inputMessage={inputMessage}
              onInputChange={setInputMessage}
              error={error}
              isStreaming={isStreaming}
              isAtLimit={isAtLimit}
              selectedArtist={selectedArtist}
              selectedProject={selectedProject}
              selectedContracts={selectedContracts}
              contracts={contracts}
              onDeselectContract={(id) => setSelectedContracts(prev => prev.filter(c => c !== id))}
              onSend={handleSendMessage}
              onStop={stopGeneration}
              onKeyDown={handleKeyDown}
              onUploadClick={() => setUploadModalOpen(true)}
            />
          </div>
        </main>
      </div>

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
  );
};

export default Zoe;
