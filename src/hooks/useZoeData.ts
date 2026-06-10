import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useToast } from "@/hooks/use-toast";
import { useStreamingChat } from "@/hooks/useStreamingChat";
import type { Message, AssistantQuickAction } from "@/hooks/useStreamingChat";
import { useConversationPersistence, restoreLatestSession } from "@/hooks/useConversationPersistence";
import { useMyCollaborations } from "@/hooks/useRegistry";
import { type WorkFileLink } from "@/hooks/useWorkFiles";
import type {
  Artist, Project, Contract,
  ArtistDataExtracted, ExtractedContractData,
  ConversationContext,
} from "@/components/zoe/types";
import { apiFetch, getAuthHeaders, API_URL, ApiError } from "@/lib/apiFetch";

const WORKING_SET_BUDGET_CHARS = 250_000; // well under the backend's 370k hard cap

export function pickWorkingSet(opts: {
  selectedIds: string[];
  recencyOrder: string[]; // most-recently-selected first
  markdowns: Record<string, string>;
  budgetChars?: number;
}): Record<string, string> {
  const budget = opts.budgetChars ?? WORKING_SET_BUDGET_CHARS;
  const out: Record<string, string> = {};
  let used = 0;
  const add = (id: string, enforceBudget: boolean) => {
    if (out[id]) return;
    const md = opts.markdowns[id];
    if (!md) return; // only contracts whose text we actually have
    // Pinned (selected) contracts are added first with enforceBudget=false, so `used` can only
    // exceed the budget via the user's own selection — carried contracts never cause overflow.
    if (enforceBudget && used + md.length > budget) return;
    out[id] = md;
    used += md.length;
  };
  for (const id of opts.selectedIds) add(id, false); // pinned
  for (const id of opts.recencyOrder) if (!opts.selectedIds.includes(id)) add(id, true);
  return out;
}

export function useZoeData() {
  const { user } = useAuth();
  const { toast } = useToast();

  const restoredSession = useState(() => restoreLatestSession())[0];

  const [artists, setArtists] = useState<Artist[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [contracts, setContracts] = useState<Contract[]>([]);
  const [selectedArtist, setSelectedArtist] = useState<string>(restoredSession?.selectedArtist || "");
  const [selectedProject, setSelectedProject] = useState<string>(restoredSession?.selectedProject || "");
  const [selectedContracts, setSelectedContracts] = useState<string[]>(restoredSession?.selectedContracts || []);

  const [contractMarkdowns, setContractMarkdowns] = useState<Record<string, string>>({});

  // ── Comparison-context tree: multi-select across artists/projects ──
  const [contextTree, setContextTree] = useState<{
    artists: { id: string; name: string; project_count: number }[];
    projects: { id: string; name: string; artist_id: string; doc_count: number }[];
  }>({ artists: [], projects: [] });
  const [checkedArtistIds, setCheckedArtistIds] = useState<string[]>([]);
  const [checkedProjectIds, setCheckedProjectIds] = useState<string[]>([]);
  const [projectDocuments, setProjectDocuments] = useState<
    Record<
      string,
      { id: string; file_name: string; project_id: string; folder_category?: string; page_count?: number | null }[]
    >
  >({});

  const [isCreateProjectOpen, setIsCreateProjectOpen] = useState(false);
  const [newProjectNameInput, setNewProjectNameInput] = useState("");
  const [isCreatingProject, setIsCreatingProject] = useState(false);

  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const [sidebarWidth, setSidebarWidth] = useState<number>(288);
  const [isResizing, setIsResizing] = useState<boolean>(false);
  const [contractsOpen, setContractsOpen] = useState<boolean>(false);
  const prevContractsCountRef = useRef<number>(0);
  const prevSelectedContractsRef = useRef<string[]>([]);
  const recencyRef = useRef<string[]>([]);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const {
    messages, setMessages, isStreaming, error, setError,
    sendMessage, stopGeneration, retryLastMessage,
    clearMessages, isAtLimit,
  } = useStreamingChat();
  const [inputMessage, setInputMessage] = useState("");
  const [sessionId, setSessionId] = useState<string>(
    () => restoredSession?.sessionId || crypto.randomUUID()
  );
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);

  const [conversationContext, setConversationContext] = useState<ConversationContext>(() => {
    if (restoredSession?.conversationContext) {
      return restoredSession.conversationContext as ConversationContext;
    }
    return {
      session_id: sessionId,
      artist: null,
      artists_discussed: [],
      project: null,
      contracts_discussed: [],
      context_switches: [],
    };
  });

  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [contractToDelete, setContractToDelete] = useState<Contract | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [showReloadDialog, setShowReloadDialog] = useState(false);

  const { data: sharedWorks = [], isLoading: isLoadingSharedWorks } = useMyCollaborations();
  const [sharedWorksOpen, setSharedWorksOpen] = useState(false);
  const [selectedSharedWork, setSelectedSharedWork] = useState<string | null>(null);
  const [sharedWorkFiles, setSharedWorkFiles] = useState<WorkFileLink[]>([]);
  const [loadingWorkFiles, setLoadingWorkFiles] = useState(false);

  const { clearSession } = useConversationPersistence({
    sessionId,
    messages,
    selectedArtist,
    selectedProject,
    selectedContracts,
    conversationContext,
  });

  const selectedArtistName = artists.find((a) => a.id === selectedArtist)?.name;
  const selectedProjectName = projects.find((p) => p.id === selectedProject)?.name;

  useEffect(() => {
    if (restoredSession?.messages && restoredSession.messages.length > 0) {
      setMessages(restoredSession.messages);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleNewConversation = useCallback(() => {
    clearSession();
    clearMessages();
    recencyRef.current = []; // fresh working set per conversation
    setCheckedArtistIds([]);
    setCheckedProjectIds([]);
    setSessionId(crypto.randomUUID());
    setConversationContext({
      session_id: crypto.randomUUID(),
      artist: selectedArtist && selectedArtistName
        ? { id: selectedArtist, name: selectedArtistName }
        : null,
      artists_discussed: [],
      project: selectedProject && selectedProjectName
        ? { id: selectedProject, name: selectedProjectName }
        : null,
      contracts_discussed: [],
      context_switches: [],
    });
    setInputMessage("");
  }, [clearSession, clearMessages, selectedArtist, selectedArtistName, selectedProject, selectedProjectName]);

  useEffect(() => {
    if (!user) return;
    apiFetch<Artist[]>(`${API_URL}/artists`)
      .then((data) => setArtists(data))
      .catch((err) => {
        console.error("Error fetching artists:", err);
        setError("Failed to load artists");
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id]);

  useEffect(() => {
    if (selectedArtist) {
      apiFetch<Project[]>(`${API_URL}/artists/${selectedArtist}/projects`)
        .then((data) => setProjects(data))
        .catch((err) => {
          console.error("Error fetching projects:", err);
          setProjects([]);
        });
    } else {
      setProjects([]);
      setSelectedProject("");
    }
  }, [selectedArtist]);

  const fetchContracts = () => {
    if (selectedProject) {
      apiFetch<Contract[]>(`${API_URL}/projects/${selectedProject}/documents`)
        .then((data) => setContracts(data))
        .catch((err) => {
          console.error("Error fetching documents:", err);
          setContracts([]);
        });
    } else {
      // No project in view → clear the BROWSE list only. The selection persists (it can span
      // artists/projects for cross-contract comparison); use "Clear all" to reset it.
      setContracts([]);
    }
  };

  useEffect(() => {
    fetchContracts();
    if (selectedProject) {
      setContractsOpen(true);
    } else {
      setContractsOpen(false);
    }
  }, [selectedProject]);

  useEffect(() => {
    if (selectedArtist && selectedArtistName) {
      setConversationContext(prev => {
        const isArtistSwitch = prev.artist && prev.artist.id !== selectedArtist;
        if (isArtistSwitch) {
          const dividerMessage: Message = {
            id: `divider-artist-${Date.now()}`,
            role: "system",
            content: `--- Switched to Artist: ${selectedArtistName} ---`,
            timestamp: new Date().toISOString(),
          };
          setMessages(prevMessages => [...prevMessages, dividerMessage]);
        }
        const contextSwitches = isArtistSwitch
          ? [...prev.context_switches, {
              timestamp: new Date().toISOString(),
              type: 'artist' as const,
              from: prev.artist.name,
              to: selectedArtistName
            }]
          : prev.context_switches;
        const artistInDiscussed = prev.artists_discussed.find(a => a.id === selectedArtist);
        const updatedArtistsDiscussed = artistInDiscussed
          ? prev.artists_discussed
          : [...prev.artists_discussed, {
              id: selectedArtist,
              name: selectedArtistName,
              data_extracted: {}
            }];
        return {
          ...prev,
          session_id: sessionId,
          artist: { id: selectedArtist, name: selectedArtistName },
          artists_discussed: updatedArtistsDiscussed,
          context_switches: contextSwitches
        };
      });
    }
  }, [selectedArtist, selectedArtistName, sessionId]);

  useEffect(() => {
    if (selectedProject && selectedProjectName) {
      setConversationContext(prev => {
        const isProjectSwitch = prev.project && prev.project.id !== selectedProject;
        if (isProjectSwitch) {
          const dividerMessage: Message = {
            id: `divider-project-${Date.now()}`,
            role: "system",
            content: `--- Switched to Project: ${selectedProjectName} ---`,
            timestamp: new Date().toISOString(),
          };
          setMessages(prevMessages => [...prevMessages, dividerMessage]);
        }
        const contextSwitches = isProjectSwitch
          ? [...prev.context_switches, {
              timestamp: new Date().toISOString(),
              type: 'project' as const,
              from: prev.project.name,
              to: selectedProjectName
            }]
          : prev.context_switches;
        return {
          ...prev,
          session_id: sessionId,
          project: { id: selectedProject, name: selectedProjectName },
          context_switches: contextSwitches
        };
      });
    } else if (!selectedProject) {
      setConversationContext(prev => ({
        ...prev,
        project: null
      }));
    }
  }, [selectedProject, selectedProjectName, sessionId]);

  useEffect(() => {
    // Resolve names across the current artist's contracts AND popover-selected
    // project documents — falling back to the raw id leaks UUIDs into the UI.
    const nameById = new Map(contracts.map((c) => [c.id, c.file_name]));
    for (const docs of Object.values(projectDocuments)) {
      for (const d of docs) if (!nameById.has(d.id)) nameById.set(d.id, d.file_name);
    }
    const prevIds = prevSelectedContractsRef.current;
    const currentIds = selectedContracts;
    const isContractSwitch = prevIds.length > 0
      && (prevIds.length !== currentIds.length
          || !prevIds.every(id => currentIds.includes(id)));
    if (isContractSwitch) {
      const fromNames = prevIds.map(id => nameById.get(id) || id);
      const toNames = currentIds.map(id => nameById.get(id) || id);
      const singleName = currentIds.length === 1 ? nameById.get(currentIds[0]) : undefined;
      const dividerContent = singleName
        ? `--- Switched to Contract: ${singleName} ---`
        : currentIds.length >= 1
          ? `--- Switched Contracts ---`
          : `--- Deselected Contracts ---`;
      const dividerMessage: Message = {
        id: `divider-contract-${Date.now()}`,
        role: "system",
        content: dividerContent,
        timestamp: new Date().toISOString(),
      };
      setMessages(prevMessages => [...prevMessages, dividerMessage]);
      setConversationContext(prev => ({
        ...prev,
        context_switches: [...prev.context_switches, {
          timestamp: new Date().toISOString(),
          type: 'contract' as const,
          from: fromNames.join(', '),
          to: toNames.join(', ')
        }]
      }));
    }
    if (selectedContracts.length > 0) {
      setConversationContext(prev => {
        const existingIds = prev.contracts_discussed.map(c => c.id);
        const newContracts = selectedContracts
          .filter(id => !existingIds.includes(id))
          .map(id => ({
            id,
            name: nameById.get(id) || id,
            data_extracted: {}
          }));
        if (newContracts.length === 0) return prev;
        return {
          ...prev,
          contracts_discussed: [...prev.contracts_discussed, ...newContracts]
        };
      });
    }
    recencyRef.current = [
      ...selectedContracts,
      ...recencyRef.current.filter((id) => !selectedContracts.includes(id)),
    ];
    prevSelectedContractsRef.current = selectedContracts;
  }, [selectedContracts, contracts, projectDocuments]);

  // NOTE: the working set is intentionally NOT reset on artist/project switch — carrying
  // recently-discussed contracts across artists is what enables "compare to the previous
  // contract" after switching. It IS reset on New Chat (see handleNewConversation).

  useEffect(() => {
    if (!user || selectedContracts.length === 0) return;
    const fetchMissing = async () => {
      const missing = selectedContracts.filter(id => !contractMarkdowns[id]);
      if (missing.length === 0) return;
      const results: Record<string, string> = {};
      const authHeaders = await getAuthHeaders();
      await Promise.all(
        missing.map(async (cid) => {
          try {
            const res = await fetch(`${API_URL}/contracts/${cid}/markdown`, {
              headers: authHeaders,
            });
            if (res.ok) {
              const data = await res.json();
              if (data.markdown) {
                results[cid] = data.markdown;
              }
            }
          } catch (err) {
            console.warn(`Failed to fetch markdown for contract ${cid}:`, err);
          }
        })
      );
      if (Object.keys(results).length > 0) {
        setContractMarkdowns(prev => ({ ...prev, ...results }));
      }
    };
    fetchMissing();
  }, [selectedContracts, user]);

  // Comparison-context tree (artists + projects with counts).
  const fetchContextTree = useCallback(async () => {
    if (!user) return;
    try {
      const headers = await getAuthHeaders();
      const res = await fetch(`${API_URL}/zoe/context-tree`, { headers });
      if (res.ok) setContextTree(await res.json());
    } catch (err) {
      console.warn("Failed to fetch Zoe context tree:", err);
    }
  }, [user]);

  useEffect(() => {
    fetchContextTree();
  }, [fetchContextTree]);

  // Documents (with page counts) for newly-checked projects.
  useEffect(() => {
    if (!user) return;
    const missing = checkedProjectIds.filter((pid) => !projectDocuments[pid]);
    if (missing.length === 0) return;
    (async () => {
      const headers = await getAuthHeaders();
      const results: Record<
        string,
        { id: string; file_name: string; project_id: string; folder_category?: string; page_count?: number | null }[]
      > = {};
      await Promise.all(
        missing.map(async (pid) => {
          try {
            const res = await fetch(`${API_URL}/projects/${pid}/documents`, { headers });
            if (res.ok) results[pid] = await res.json();
          } catch (err) {
            console.warn(`Failed to fetch documents for project ${pid}:`, err);
          }
        })
      );
      if (Object.keys(results).length > 0) setProjectDocuments((prev) => ({ ...prev, ...results }));
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [checkedProjectIds, projectDocuments, user]);

  useEffect(() => {
    if (!selectedSharedWork || !user) {
      setSharedWorkFiles([]);
      return;
    }
    setLoadingWorkFiles(true);
    apiFetch<{ files: WorkFileLink[] }>(`${API_URL}/registry/works/${selectedSharedWork}/files`)
      .then((data) => {
        setSharedWorkFiles(data.files || []);
        setLoadingWorkFiles(false);
      })
      .catch((err) => {
        console.error("Error fetching work files:", err);
        setSharedWorkFiles([]);
        setLoadingWorkFiles(false);
      });
  }, [selectedSharedWork, user]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    const prev = prevContractsCountRef.current;
    if (prev === 0 && selectedContracts.length > 0) {
      // placeholder for future auto-collapse behavior
    }
    prevContractsCountRef.current = selectedContracts.length;
  }, [selectedContracts]);

  const handleUploadComplete = () => {
    fetchContracts();
    // Refresh the comparison-context panel: re-pull counts and drop cached docs for the
    // upload target so the new file appears in its project's document list.
    fetchContextTree();
    setProjectDocuments((prev) => {
      const next = { ...prev };
      for (const pid of checkedProjectIds) delete next[pid];
      return next;
    });
  };

  const handleCreateProject = async () => {
    if (!selectedArtist || !newProjectNameInput.trim()) return;
    setIsCreatingProject(true);
    try {
        const newProject = await apiFetch<Project>(`${API_URL}/projects`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                artist_id: selectedArtist,
                name: newProjectNameInput,
            })
        });
        setProjects([newProject, ...projects]);
        setSelectedProject(newProject.id);
        setNewProjectNameInput("");
        setIsCreateProjectOpen(false);
        toast({
          title: "Success",
          description: "Project created successfully",
        });
    } catch (err) {
        console.error("Error creating project:", err);
        toast({
          title: "Error",
          description: err instanceof ApiError ? err.message : "Failed to create project",
          variant: "destructive",
        });
    } finally {
        setIsCreatingProject(false);
    }
  };

  const handleDeleteClick = (contract: Contract) => {
    setContractToDelete(contract);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!contractToDelete || !user) return;
    setDeleting(true);
    try {
      const authHeaders = await getAuthHeaders();
      const response = await fetch(`${API_URL}/contracts/${contractToDelete.id}`, {
        method: "DELETE",
        headers: authHeaders,
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Delete failed");
      }
      setSelectedContracts((prev) => prev.filter((id) => id !== contractToDelete.id));
      fetchContracts();
      setDeleteDialogOpen(false);
      setContractToDelete(null);
    } catch (err) {
      console.error("Error deleting contract:", err);
      setError(err instanceof Error ? err.message : "Failed to delete contract");
    } finally {
      setDeleting(false);
    }
  };

  const getChatParams = useCallback(() => {
    const markdowns = pickWorkingSet({
      selectedIds: selectedContracts,
      recencyOrder: recencyRef.current,
      markdowns: contractMarkdowns,
    });
    return {
      userId: user!.id,
      artistId: selectedArtist || undefined,
      projectId: selectedProject || undefined,
      contractIds: selectedContracts.length > 0 ? selectedContracts : undefined,
      sessionId,
      context: conversationContext,
      contractMarkdowns: Object.keys(markdowns).length > 0 ? markdowns : undefined,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id, selectedArtist, selectedProject, selectedContracts, sessionId, conversationContext, contractMarkdowns]);

  const handleSendResult = useCallback((result: {
    sessionId: string;
    contextCleared?: boolean;
    extractedData?: Record<string, unknown> | null;
    answeredFrom?: string;
    sources?: Array<{ contract_file: string; score: number }>;
  }) => {
    if (result.sessionId !== sessionId) {
      setSessionId(result.sessionId);
    }
    if (result.contextCleared) {
      setShowReloadDialog(true);
    }
    if (result.extractedData && Object.keys(result.extractedData).length > 0) {
      const isArtistData = 'bio' in result.extractedData || 'social_media' in result.extractedData || 'streaming_links' in result.extractedData;
      if (isArtistData && selectedArtist && selectedArtistName &&
          (result.answeredFrom === 'artist_data' || result.answeredFrom === 'artist_comparison')) {
        setConversationContext(prev => {
          const idx = prev.artists_discussed.findIndex(a => a.id === selectedArtist);
          const updatedArtists = [...prev.artists_discussed];
          if (idx >= 0) {
            updatedArtists[idx] = {
              ...updatedArtists[idx],
              data_extracted: { ...updatedArtists[idx].data_extracted, ...result.extractedData }
            };
          } else {
            updatedArtists.push({ id: selectedArtist, name: selectedArtistName!, data_extracted: result.extractedData as ArtistDataExtracted });
          }
          return { ...prev, artists_discussed: updatedArtists };
        });
      }
      if (!isArtistData && result.sources && result.sources.length > 0) {
        const contractFiles = [...new Set(result.sources.map(s => s.contract_file))];
        setConversationContext(prev => {
          const updatedContracts = [...prev.contracts_discussed];
          contractFiles.forEach(fileName => {
            const idx = updatedContracts.findIndex(c => c.name === fileName);
            if (idx >= 0) {
              updatedContracts[idx] = {
                ...updatedContracts[idx],
                data_extracted: { ...updatedContracts[idx].data_extracted, ...result.extractedData as ExtractedContractData }
              };
            } else {
              updatedContracts.push({ id: fileName, name: fileName, data_extracted: result.extractedData as ExtractedContractData });
            }
          });
          return { ...prev, contracts_discussed: updatedContracts };
        });
      }
    }
  }, [sessionId, selectedArtist, selectedArtistName]);

  const handleSendMessage = useCallback(async () => {
    if (!inputMessage.trim() || !user) {
      setError("Please enter a message");
      return;
    }
    if (isAtLimit) {
      setShowReloadDialog(true);
      return;
    }
    const query = inputMessage;
    setInputMessage("");
    const result = await sendMessage(query, getChatParams());
    handleSendResult(result);
  }, [inputMessage, user, isAtLimit, sendMessage, getChatParams, handleSendResult]);

  const handleQuickAction = useCallback(async (question: string) => {
    if (!user) return;
    if (isAtLimit) { setShowReloadDialog(true); return; }
    const result = await sendMessage(question, getChatParams());
    handleSendResult(result);
  }, [user, isAtLimit, sendMessage, getChatParams, handleSendResult]);

  const handleAssistantQuickAction = useCallback(async (action: AssistantQuickAction) => {
    const queryToSend = action.query || inputMessage;
    if (!queryToSend.trim() || !user) return;
    const result = await sendMessage(queryToSend, getChatParams(), {
      sourcePreference: action.source_preference,
      userDisplayMessage: action.label,
      silent: !!action.source_preference,
    });
    handleSendResult(result);
  }, [inputMessage, user, sendMessage, getChatParams, handleSendResult]);

  const handleRetry = useCallback(async () => {
    if (!user) return;
    const result = await retryLastMessage(getChatParams());
    if (result) handleSendResult(result);
  }, [user, retryLastMessage, getChatParams, handleSendResult]);

  const handleCopyMessage = useCallback((content: string, messageId: string) => {
    navigator.clipboard.writeText(content);
    setCopiedMessageId(messageId);
    setTimeout(() => setCopiedMessageId(null), 2000);
  }, []);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isStreaming) {
        handleSendMessage();
      }
    }
  }, [isStreaming, handleSendMessage]);

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      const newWidth = e.clientX;
      if (newWidth >= 220 && newWidth <= 480) {
        setSidebarWidth(newWidth);
      }
    };
    const handleMouseUp = () => {
      setIsResizing(false);
    };
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizing]);

  // Session-wide {id, file_name} for every contract discussed this conversation (across artists),
  // merged with the current artist's list. Used to resolve source chips on older messages so they
  // still render and page-jump after the user has switched to a different artist/project.
  const knownContracts = useMemo(() => {
    const byId = new Map(contracts.map((c) => [c.id, { id: c.id, file_name: c.file_name }]));
    for (const docs of Object.values(projectDocuments)) {
      for (const d of docs) if (!byId.has(d.id)) byId.set(d.id, { id: d.id, file_name: d.file_name });
    }
    for (const d of conversationContext.contracts_discussed) {
      if (!byId.has(d.id)) byId.set(d.id, { id: d.id, file_name: d.name });
    }
    return Array.from(byId.values());
  }, [contracts, projectDocuments, conversationContext.contracts_discussed]);

  return {
    user,
    artists,
    projects,
    contracts,
    knownContracts,
    contractMarkdowns,
    // Comparison-context tree (multi-select)
    contextTree,
    checkedArtistIds,
    setCheckedArtistIds,
    checkedProjectIds,
    setCheckedProjectIds,
    projectDocuments,
    refreshContextTree: fetchContextTree,
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
  };
}
