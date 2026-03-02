import { useState, useEffect, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { ArrowLeft, Upload, Trash2, ChevronDown, Music, Plus, Loader2, PanelLeftClose, PanelLeft, FileText, FolderOpen, Users, GripVertical, RefreshCw } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { ZoeChatMessages } from "@/components/zoe/ZoeChatMessages";
import { ZoeInputBar } from "@/components/zoe/ZoeInputBar";
import { ContractUploadModal } from "@/components/ContractUploadModal";
import { useAuth } from "@/contexts/AuthContext";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { useToast } from "@/hooks/use-toast";
import { useStreamingChat } from "@/hooks/useStreamingChat";
import type { Message, AssistantQuickAction } from "@/hooks/useStreamingChat";
import { useConversationPersistence, restoreLatestSession } from "@/hooks/useConversationPersistence";
import { cn } from "@/lib/utils";

// Backend API URL
const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

interface Artist {
  id: string;
  name: string;
}

interface Project {
  id: string;
  name: string;
  artist_id: string;
}

interface Contract {
  id: string;
  file_name: string;
  project_id: string;
}

// Message, SourcePreference, AssistantQuickAction types imported from useStreamingChat

// Conversation context for structured tracking
interface RoyaltySplitData {
  party: string;
  percentage: number;
}

interface RoyaltySplitsByType {
  streaming?: RoyaltySplitData[];
  publishing?: RoyaltySplitData[];
  mechanical?: RoyaltySplitData[];
  sync?: RoyaltySplitData[];
  master?: RoyaltySplitData[];
  performance?: RoyaltySplitData[];
  general?: RoyaltySplitData[];
}

interface ExtractedContractData {
  royalty_splits?: RoyaltySplitsByType;
  payment_terms?: string;
  parties?: string[];
  advances?: string;
  term_length?: string;
  [key: string]: unknown;
}

interface ArtistDataExtracted {
  bio?: string;
  social_media?: Record<string, string>;
  streaming_links?: Record<string, string>;
  genres?: string[];
  email?: string;
}

interface ArtistDiscussed {
  id: string;
  name: string;
  data_extracted: ArtistDataExtracted;
}

interface ContractDiscussed {
  id: string;
  name: string;
  data_extracted: ExtractedContractData;
}

interface ContextSwitch {
  timestamp: string;
  type: 'artist' | 'project' | 'contract';
  from: string;
  to: string;
}

interface ConversationContext {
  session_id: string;
  artist: { id: string; name: string } | null;
  artists_discussed: ArtistDiscussed[];
  project: { id: string; name: string } | null;
  contracts_discussed: ContractDiscussed[];
  context_switches: ContextSwitch[];
}

const Zoe = () => {
  const navigate = useNavigate();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { user } = useAuth();
  const { toast } = useToast();

  // Restore previous session (runs once, before other state)
  const [restoredSession] = useState(() => restoreLatestSession());

  // State for artists, projects and contracts
  const [artists, setArtists] = useState<Artist[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [contracts, setContracts] = useState<Contract[]>([]);
  const [selectedArtist, setSelectedArtist] = useState<string>(restoredSession?.selectedArtist || "");
  const [selectedProject, setSelectedProject] = useState<string>(restoredSession?.selectedProject || "");
  const [selectedContracts, setSelectedContracts] = useState<string[]>(restoredSession?.selectedContracts || []);

  // Create Project State
  const [isCreateProjectOpen, setIsCreateProjectOpen] = useState(false);
  const [newProjectNameInput, setNewProjectNameInput] = useState("");
  const [isCreatingProject, setIsCreatingProject] = useState(false);

  // Collapsible UI state
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const [sidebarWidth, setSidebarWidth] = useState<number>(288); // 288px = w-72
  const [isResizing, setIsResizing] = useState<boolean>(false);
  const [contractsOpen, setContractsOpen] = useState<boolean>(false);
  const prevContractsCountRef = useRef<number>(0);
  const sidebarRef = useRef<HTMLDivElement>(null);

  // Chat state (via streaming hook)
  const {
    messages, setMessages, isStreaming, error, setError,
    sendMessage, stopGeneration, retryLastMessage,
    addSystemMessage, clearMessages, isAtLimit,
  } = useStreamingChat();
  const [inputMessage, setInputMessage] = useState("");
  const [sessionId, setSessionId] = useState<string>(
    () => restoredSession?.sessionId || crypto.randomUUID()
  );
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);

  // Conversation context for structured tracking
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

  // Upload/Delete state
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [contractToDelete, setContractToDelete] = useState<Contract | null>(null);
  const [deleting, setDeleting] = useState(false);
  
  // Context cleared reload dialog
  const [showReloadDialog, setShowReloadDialog] = useState(false);

  // Conversation persistence — auto-saves to localStorage
  const { clearSession } = useConversationPersistence({
    sessionId,
    messages,
    selectedArtist,
    selectedProject,
    selectedContracts,
    conversationContext,
  });

  // Restore messages from saved session on mount
  useEffect(() => {
    if (restoredSession?.messages && restoredSession.messages.length > 0) {
      setMessages(restoredSession.messages);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const selectedArtistName = artists.find((a) => a.id === selectedArtist)?.name;
  const selectedProjectName = projects.find((p) => p.id === selectedProject)?.name;

  // New conversation — clears chat and starts fresh
  const handleNewConversation = useCallback(() => {
    clearSession();
    clearMessages();
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

  // Fetch artists on mount - filter by logged in user
  useEffect(() => {
    if (!user) return;
    
    fetch(`${API_URL}/artists?user_id=${user.id}`)
      .then((res) => res.json())
      .then((data) => setArtists(data))
      .catch((err) => {
        console.error("Error fetching artists:", err);
        setError("Failed to load artists");
      });
  }, [user]);

  // Fetch projects when artist is selected
  useEffect(() => {
    if (selectedArtist) {
      fetch(`${API_URL}/artists/${selectedArtist}/projects`)
        .then((res) => res.json())
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

  // Fetch contracts when project is selected
  const fetchContracts = () => {
    if (selectedProject) {
      fetch(`${API_URL}/projects/${selectedProject}/contracts`)
        .then((res) => res.json())
        .then((data) => setContracts(data))
        .catch((err) => {
          console.error("Error fetching contracts:", err);
          setContracts([]);
        });
    } else {
      setContracts([]);
      setSelectedContracts([]);
    }
  };

  useEffect(() => {
    fetchContracts();
    // Update UI when switching projects (but keep conversation history)
    if (selectedProject) {
      setContractsOpen(true);
    } else {
      setContractsOpen(false);
    }
  }, [selectedProject]);

  // Track artist context changes - insert divider message when switching artists
  useEffect(() => {
    if (selectedArtist && selectedArtistName) {
      setConversationContext(prev => {
        // Check if this is a switch to a different artist
        const isArtistSwitch = prev.artist && prev.artist.id !== selectedArtist;
        
        if (isArtistSwitch) {
          // Insert a divider message instead of clearing conversation
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
        
        // Ensure current artist is in artists_discussed
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

  // Track project context changes - insert divider message when switching projects
  useEffect(() => {
    if (selectedProject && selectedProjectName) {
      setConversationContext(prev => {
        const isProjectSwitch = prev.project && prev.project.id !== selectedProject;
        
        if (isProjectSwitch) {
          // Insert a divider message for project switch
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
          // Keep contracts_discussed to allow cross-project comparisons
          context_switches: contextSwitches
        };
      });
    } else if (!selectedProject) {
      setConversationContext(prev => ({
        ...prev,
        project: null
        // Note: contracts_discussed is preserved to allow cross-artist/project comparisons
      }));
    }
  }, [selectedProject, selectedProjectName, sessionId]);

  // Track contract selection changes for context
  useEffect(() => {
    if (selectedContracts.length > 0) {
      const selectedContractNames = selectedContracts.map(id => {
        const contract = contracts.find(c => c.id === id);
        return contract?.file_name || id;
      });
      
      // Add newly selected contracts to contracts_discussed if not already present
      setConversationContext(prev => {
        const existingIds = prev.contracts_discussed.map(c => c.id);
        const newContracts = selectedContracts
          .filter(id => !existingIds.includes(id))
          .map(id => ({
            id,
            name: contracts.find(c => c.id === id)?.file_name || id,
            data_extracted: {}
          }));
        
        if (newContracts.length === 0) return prev;
        
        return {
          ...prev,
          contracts_discussed: [...prev.contracts_discussed, ...newContracts]
        };
      });
    }
  }, [selectedContracts, contracts]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Collapse contracts dropdown once at least one contract is selected (first time)
  useEffect(() => {
    const prev = prevContractsCountRef.current;
    if (prev === 0 && selectedContracts.length > 0) {
      // setContractsOpen(false);
      // Also collapse the whole context to maximize chat
      // setContextOpen(false);
    }
    prevContractsCountRef.current = selectedContracts.length;
  }, [selectedContracts]);

  const handleUploadComplete = () => {
    // Refresh contracts list after upload
    fetchContracts();
  };

  const handleCreateProject = async () => {
    if (!selectedArtist || !newProjectNameInput.trim()) return;
    
    setIsCreatingProject(true);
    try {
        const response = await fetch(`${API_URL}/projects`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                artist_id: selectedArtist,
                name: newProjectNameInput,
                description: "Created via Zoe"
            })
        });
        
        if (!response.ok) throw new Error("Failed to create project");
        
        const newProject = await response.json();
        setProjects([newProject, ...projects]);
        setSelectedProject(newProject.id); // Auto-select
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
          description: "Failed to create project",
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
      const formData = new FormData();
      formData.append("user_id", user.id);

      const response = await fetch(`${API_URL}/contracts/${contractToDelete.id}`, {
        method: "DELETE",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Delete failed");
      }

      // Remove from selected contracts if it was selected
      setSelectedContracts((prev) => prev.filter((id) => id !== contractToDelete.id));
      
      // Refresh contracts list
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

  // ── Chat params helper (shared by all send functions) ──
  const getChatParams = useCallback(() => ({
    userId: user!.id,
    artistId: selectedArtist,
    projectId: selectedProject || undefined,
    contractIds: selectedContracts.length > 0 ? selectedContracts : undefined,
    sessionId,
    context: conversationContext,
  }), [user, selectedArtist, selectedProject, selectedContracts, sessionId, conversationContext]);

  // Process the result returned from sendMessage (update session, context, etc.)
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
    // Update conversation context with extracted data from backend
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
    if (!selectedArtist) {
      setError("Please select an artist first");
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
  }, [inputMessage, user, selectedArtist, isAtLimit, sendMessage, getChatParams, handleSendResult]);

  // Handle static quick action button clicks
  const handleQuickAction = useCallback(async (question: string) => {
    if (!selectedArtist || !user) return;
    if (isAtLimit) { setShowReloadDialog(true); return; }

    const result = await sendMessage(question, getChatParams());
    handleSendResult(result);
  }, [selectedArtist, user, isAtLimit, sendMessage, getChatParams, handleSendResult]);

  const handleAssistantQuickAction = useCallback(async (action: AssistantQuickAction) => {
    const queryToSend = action.query || inputMessage;
    if (!queryToSend.trim() || !selectedArtist || !user) return;

    const result = await sendMessage(queryToSend, getChatParams(), {
      sourcePreference: action.source_preference,
      userDisplayMessage: action.label,
      silent: !!action.source_preference,
    });
    handleSendResult(result);
  }, [inputMessage, selectedArtist, user, sendMessage, getChatParams, handleSendResult]);

  const handleRetry = useCallback(async () => {
    if (!selectedArtist || !user) return;
    const result = await retryLastMessage(getChatParams());
    if (result) handleSendResult(result);
  }, [selectedArtist, user, retryLastMessage, getChatParams, handleSendResult]);

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

  // Sidebar resize handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      const newWidth = e.clientX;
      // Constrain width between 220px and 480px
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
            <div className="flex items-center gap-2 cursor-pointer" onClick={() => navigate("/dashboard")}>
              <div className="w-9 h-9 rounded-lg bg-primary flex items-center justify-center">
                <Music className="w-5 h-5 text-primary-foreground" />
              </div>
              <div className="hidden sm:block">
                <h1 className="text-lg font-bold text-foreground leading-none">Zoe AI</h1>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {/* Context summary badge - shows current context */}
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
                    <span>{selectedContracts.length} contracts</span>
                  </>
                )}
              </Badge>
            )}
            {messages.length > 0 && (
              <Button variant="outline" onClick={handleNewConversation} size="sm" className="gap-2">
                <RefreshCw className="w-4 h-4" />
                <span className="hidden sm:inline">New Chat</span>
              </Button>
            )}
            <Button variant="outline" onClick={() => navigate("/tools")} size="sm" className="gap-2">
              <ArrowLeft className="w-4 h-4" />
              <span className="hidden sm:inline">Back to Tools</span>
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Collapsible & Resizable Sidebar */}
        <aside
          ref={sidebarRef}
          className={cn(
            "border-r border-border bg-muted/30 flex-shrink-0 overflow-hidden relative",
            !isResizing && "transition-all duration-300 ease-in-out",
            !sidebarOpen && "!w-0"
          )}
          style={{ width: sidebarOpen ? sidebarWidth : 0 }}
        >
          <div className="h-full flex flex-col" style={{ width: sidebarWidth }}>
            {/* Resize Handle */}
            <div
              className={cn(
                "absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-primary/20 active:bg-primary/30 z-10 group",
                isResizing && "bg-primary/30"
              )}
              onMouseDown={handleMouseDown}
            >
              <div className="absolute right-0 top-1/2 -translate-y-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
                <GripVertical className="w-3 h-3 text-muted-foreground" />
              </div>
            </div>
            <ScrollArea className="flex-1">
              <div className="p-4 space-y-6">
                {/* Artist Selection */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    <Users className="w-3.5 h-3.5" />
                    Artist
                  </div>
                  <Select value={selectedArtist} onValueChange={(v) => { setSelectedArtist(v); setSelectedProject(''); setSelectedContracts([]); }}>
                    <SelectTrigger className="w-full bg-background">
                      <SelectValue placeholder="Select artist..." />
                    </SelectTrigger>
                    <SelectContent>
                      {artists.map((artist) => (
                        <SelectItem key={artist.id} value={artist.id}>
                          {artist.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Project Selection */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    <FolderOpen className="w-3.5 h-3.5" />
                    Project
                  </div>
                  <div className="flex items-center gap-2">
                    <Select value={selectedProject} onValueChange={(v) => { setSelectedProject(v); setSelectedContracts([]); }} disabled={!selectedArtist}>
                      <SelectTrigger className="flex-1 bg-background">
                        <SelectValue placeholder="Select project..." />
                      </SelectTrigger>
                      <SelectContent>
                        {projects.map((project) => (
                          <SelectItem key={project.id} value={project.id}>
                            {project.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-9 w-9 flex-shrink-0"
                      onClick={() => setIsCreateProjectOpen(true)}
                      disabled={!selectedArtist}
                      title="Create New Project"
                    >
                      <Plus className="h-4 w-4" />
                    </Button>
                  </div>
                </div>

                {/* Contracts Section */}
                {selectedProject && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                        <FileText className="w-3.5 h-3.5" />
                        Contracts {contracts.length > 0 && `(${contracts.length})`}
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setUploadModalOpen(true)}
                        className="h-7 text-xs px-2"
                      >
                        <Upload className="w-3 h-3 mr-1" />
                        Upload
                      </Button>
                    </div>

                    {/* Collapsible contracts list */}
                    <Collapsible open={contractsOpen} onOpenChange={setContractsOpen}>
                      <CollapsibleTrigger asChild>
                        <Button variant="secondary" size="sm" className="w-full justify-between h-9">
                          {selectedContracts.length > 0
                            ? `${selectedContracts.length} selected`
                            : "All contracts"}
                          <ChevronDown className={cn("ml-2 h-4 w-4 transition-transform", contractsOpen && "rotate-180")} />
                        </Button>
                      </CollapsibleTrigger>
                      <CollapsibleContent>
                        <div className="bg-background rounded-lg border mt-2 max-h-48 overflow-y-auto">
                          {contracts.length > 0 ? (
                            <div className="p-2 space-y-1">
                              {selectedContracts.length > 0 && (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="w-full h-7 text-xs justify-start text-muted-foreground hover:text-foreground"
                                  onClick={() => setSelectedContracts([])}
                                >
                                  Clear selection
                                </Button>
                              )}
                              {contracts.map((contract) => (
                                <div key={contract.id} className="flex items-center justify-between gap-2 group px-2 py-1.5 rounded hover:bg-muted/50">
                                  <div className="flex items-center space-x-2 flex-1 min-w-0">
                                    <Checkbox
                                      id={contract.id}
                                      checked={selectedContracts.includes(contract.id)}
                                      onCheckedChange={(checked) => {
                                        if (checked) {
                                          setSelectedContracts([...selectedContracts, contract.id]);
                                        } else {
                                          setSelectedContracts(selectedContracts.filter(id => id !== contract.id));
                                        }
                                      }}
                                    />
                                    <label
                                      htmlFor={contract.id}
                                      className="text-xs font-medium leading-none cursor-pointer truncate"
                                    >
                                      {contract.file_name}
                                    </label>
                                  </div>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => handleDeleteClick(contract)}
                                    className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                                  >
                                    <Trash2 className="w-3 h-3 text-destructive" />
                                  </Button>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="p-4 text-center">
                              <p className="text-xs text-muted-foreground mb-2">No contracts yet</p>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setUploadModalOpen(true)}
                                className="h-7 text-xs"
                              >
                                <Upload className="w-3 h-3 mr-1" />
                                Upload First
                              </Button>
                            </div>
                          )}
                        </div>
                      </CollapsibleContent>
                    </Collapsible>

                    {/* Search scope indicator */}
                    <Badge variant="outline" className="w-full justify-center text-xs py-1.5">
                      {selectedContracts.length > 0 
                        ? `🎯 Searching ${selectedContracts.length} contract${selectedContracts.length > 1 ? 's' : ''}`
                        : `📁 All project contracts`}
                    </Badge>
                  </div>
                )}

                {/* Empty state when no project */}
                {!selectedProject && selectedArtist && (
                  <div className="rounded-lg border border-dashed border-muted-foreground/25 p-4 text-center">
                    <FolderOpen className="w-8 h-8 mx-auto mb-2 text-muted-foreground/50" />
                    <p className="text-xs text-muted-foreground">Select a project to view contracts</p>
                  </div>
                )}

                {!selectedArtist && (
                  <div className="rounded-lg border border-dashed border-muted-foreground/25 p-4 text-center">
                    <Users className="w-8 h-8 mx-auto mb-2 text-muted-foreground/50" />
                    <p className="text-xs text-muted-foreground">Select an artist to get started</p>
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>
        </aside>

        {/* Chat Area */}
        <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
          <ZoeChatMessages
            messages={messages}
            isStreaming={isStreaming}
            selectedArtist={selectedArtist}
            selectedArtistName={selectedArtistName}
            selectedProject={selectedProject}
            copiedMessageId={copiedMessageId}
            messagesEndRef={messagesEndRef}
            onQuickAction={handleQuickAction}
            onAssistantQuickAction={handleAssistantQuickAction}
            onRetry={handleRetry}
            onCopyMessage={handleCopyMessage}
          />
          <ZoeInputBar
            inputMessage={inputMessage}
            onInputChange={setInputMessage}
            error={error}
            isStreaming={isStreaming}
            isAtLimit={isAtLimit}
            selectedArtist={selectedArtist}
            selectedProject={selectedProject}
            onSend={handleSendMessage}
            onStop={stopGeneration}
            onKeyDown={handleKeyDown}
            onUploadClick={() => setUploadModalOpen(true)}
          />
        </main>
      </div>

      {/* Upload Modal */}
      {selectedProject && (
        <ContractUploadModal
          open={uploadModalOpen}
          onOpenChange={setUploadModalOpen}
          projectId={selectedProject}
          onUploadComplete={handleUploadComplete}
        />
      )}

      {/* Create Project Dialog */}
      <Dialog open={isCreateProjectOpen} onOpenChange={setIsCreateProjectOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Create New Project</DialogTitle>
            <DialogDescription>
              Create a new project to organize your files.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="name" className="text-right">
                Name
              </Label>
              <Input
                id="name"
                value={newProjectNameInput}
                onChange={(e) => setNewProjectNameInput(e.target.value)}
                className="col-span-3"
                placeholder="e.g. Summer 2024 Release"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsCreateProjectOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateProject} disabled={isCreatingProject || !newProjectNameInput.trim()}>
              {isCreatingProject ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Creating...
                </>
              ) : (
                "Create Project"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Contract</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{contractToDelete?.file_name}"? This will remove the
              contract file and all its indexed data from the AI search. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteConfirm}
              disabled={deleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleting ? "Deleting..." : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Context Cleared / Conversation Limit Reload Dialog */}
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
    </div>
  );
};

export default Zoe;
