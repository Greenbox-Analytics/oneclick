import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { ArrowLeft, Send, Bot, User, AlertCircle, Upload, Trash2, ChevronDown, Music, Plus, Loader2, PanelLeftClose, PanelLeft, FileText, FolderOpen, Users, GripVertical, Paperclip } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
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

interface Message {
  role: "user" | "assistant";
  content: string;
  confidence?: string;
  sources?: Array<{
    contract_file: string;
    page_number: number;
    score: number;
  }>;
  timestamp: string;
  showQuickActions?: boolean;
}

// Conversation context for structured tracking
interface ExtractedContractData {
  royalty_splits?: Array<{ party: string; percentage: number }>;
  payment_terms?: string;
  parties?: string[];
  advances?: string;
  term_length?: string;
  [key: string]: unknown;
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
  project: { id: string; name: string } | null;
  contracts_discussed: ContractDiscussed[];
  context_switches: ContextSwitch[];
}

const Zoe = () => {
  const navigate = useNavigate();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { user } = useAuth();
  const { toast } = useToast();

  // State for artists, projects and contracts
  const [artists, setArtists] = useState<Artist[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [contracts, setContracts] = useState<Contract[]>([]);
  const [selectedArtist, setSelectedArtist] = useState<string>("");
  const [selectedProject, setSelectedProject] = useState<string>("");
  const [selectedContracts, setSelectedContracts] = useState<string[]>([]);

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
  
  // Chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [sessionId, setSessionId] = useState<string>(() => crypto.randomUUID());

  // Conversation context for structured tracking
  const [conversationContext, setConversationContext] = useState<ConversationContext>({
    session_id: sessionId,
    artist: null,
    project: null,
    contracts_discussed: [],
    context_switches: []
  });

  // Upload/Delete state
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [contractToDelete, setContractToDelete] = useState<Contract | null>(null);
  const [deleting, setDeleting] = useState(false);

  const selectedArtistName = artists.find((a) => a.id === selectedArtist)?.name;
  const selectedProjectName = projects.find((p) => p.id === selectedProject)?.name;

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
    // Reset UI and session when switching projects
    if (selectedProject) {
      setContractsOpen(true);
      // Start a new session for the new project context
      setSessionId(crypto.randomUUID());
      setMessages([]); // Clear previous conversation
    } else {
      setContractsOpen(false);
    }
  }, [selectedProject]);

  // Track artist context changes
  useEffect(() => {
    if (selectedArtist && selectedArtistName) {
      setConversationContext(prev => {
        const contextSwitches = prev.artist && prev.artist.id !== selectedArtist
          ? [...prev.context_switches, {
              timestamp: new Date().toISOString(),
              type: 'artist' as const,
              from: prev.artist.name,
              to: selectedArtistName
            }]
          : prev.context_switches;
        
        return {
          ...prev,
          session_id: sessionId,
          artist: { id: selectedArtist, name: selectedArtistName },
          context_switches: contextSwitches
        };
      });
    }
  }, [selectedArtist, selectedArtistName, sessionId]);

  // Track project context changes
  useEffect(() => {
    if (selectedProject && selectedProjectName) {
      setConversationContext(prev => {
        const contextSwitches = prev.project && prev.project.id !== selectedProject
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
          contracts_discussed: [], // Reset on project change
          context_switches: contextSwitches
        };
      });
    } else if (!selectedProject) {
      setConversationContext(prev => ({
        ...prev,
        project: null,
        contracts_discussed: []
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

  // Extract structured data from assistant responses to update context
  const extractDataFromAnswer = (answer: string, query: string, sources?: Message['sources']): ExtractedContractData => {
    const extracted: ExtractedContractData = {};
    const queryLower = query.toLowerCase();
    const answerLower = answer.toLowerCase();
    
    // Extract royalty splits if discussing royalties
    if (queryLower.includes('royalty') || queryLower.includes('split') || queryLower.includes('percentage') || 
        queryLower.includes('streaming') || queryLower.includes('revenue')) {
      // Match patterns like "Name: 35%" or "Name: 35% of net revenue"
      const splitMatches = answer.match(/([A-Za-z\s]+):\s*(\d+(?:\.\d+)?)\s*%/g);
      console.log('[Context] Attempting to extract splits from answer, matches:', splitMatches);
      if (splitMatches) {
        extracted.royalty_splits = splitMatches.map(match => {
          const parts = match.match(/([A-Za-z\s]+):\s*(\d+(?:\.\d+)?)\s*%/);
          if (parts) {
            return { party: parts[1].trim(), percentage: parseFloat(parts[2]) };
          }
          return null;
        }).filter((s): s is { party: string; percentage: number } => s !== null);
        console.log('[Context] Extracted royalty_splits:', extracted.royalty_splits);
      }
    }
    
    // Extract parties/signatories
    if (queryLower.includes('parties') || queryLower.includes('who signed') || queryLower.includes('signatories')) {
      // Simple extraction - look for quoted names or bullet points
      const partyMatches = answer.match(/"([^"]+)"|•\s*([^\n]+)/g);
      if (partyMatches) {
        extracted.parties = partyMatches.map(m => m.replace(/["•]/g, '').trim());
      }
    }
    
    // Extract payment terms
    if (queryLower.includes('payment') || queryLower.includes('terms') || queryLower.includes('advance')) {
      // Store a summary of payment-related answer
      if (answer.length > 0 && answer.length < 500) {
        extracted.payment_terms = answer;
      }
    }
    
    // Extract term length
    if (queryLower.includes('term') || queryLower.includes('duration') || queryLower.includes('how long')) {
      const termMatch = answer.match(/(\d+)\s*(year|month|day)s?/i);
      if (termMatch) {
        extracted.term_length = `${termMatch[1]} ${termMatch[2]}${parseInt(termMatch[1]) > 1 ? 's' : ''}`;
      }
    }
    
    return extracted;
  };

  // Update conversation context with extracted data from response
  const updateContextWithExtractedData = (
    answer: string, 
    query: string, 
    sources?: Message['sources'],
    answeredFrom?: string
  ) => {
    console.log('[Context] updateContextWithExtractedData called with sources:', sources);
    if (!sources || sources.length === 0) {
      console.log('[Context] No sources provided, skipping context update');
      return;
    }
    
    const extracted = extractDataFromAnswer(answer, query, sources);
    console.log('[Context] Extracted data:', extracted);
    if (Object.keys(extracted).length === 0) {
      console.log('[Context] No data extracted, skipping context update');
      return;
    }
    
    // Get unique contract files from sources
    const contractFiles = [...new Set(sources.map(s => s.contract_file))];
    console.log('[Context] Contract files from sources:', contractFiles);
    
    setConversationContext(prev => {
      const updatedContracts = [...prev.contracts_discussed];
      
      contractFiles.forEach(fileName => {
        const existingIndex = updatedContracts.findIndex(c => c.name === fileName);
        if (existingIndex >= 0) {
          // Merge extracted data with existing
          updatedContracts[existingIndex] = {
            ...updatedContracts[existingIndex],
            data_extracted: {
              ...updatedContracts[existingIndex].data_extracted,
              ...extracted
            }
          };
        } else {
          // Add new contract with extracted data
          updatedContracts.push({
            id: fileName, // Use filename as ID if we don't have the actual ID
            name: fileName,
            data_extracted: extracted
          });
        }
      });
      
      console.log('[Context] Updated contracts_discussed:', updatedContracts);
      
      return {
        ...prev,
        contracts_discussed: updatedContracts
      };
    });
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !user) {
      setError("Please enter a message");
      return;
    }
    
    // Need at least an artist selected
    if (!selectedArtist) {
      setError("Please select an artist first");
      return;
    }

    const userMessage: Message = {
      role: "user",
      content: inputMessage,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsLoading(true);
    setError("");

    try {
      // Call backend chatbot API with session_id for conversation memory
      const response = await fetch(`${API_URL}/zoe/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: inputMessage,
          project_id: selectedProject || null,
          contract_ids: selectedContracts.length > 0 ? selectedContracts : null,
          user_id: user.id,
          session_id: sessionId,
          artist_id: selectedArtist,
          context: conversationContext, // Send conversation context
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response from Zoe");
      }

      const data = await response.json();

      // Update session_id from response if provided (for new sessions)
      if (data.session_id && data.session_id !== sessionId) {
        setSessionId(data.session_id);
      }

      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        confidence: data.confidence,
        sources: data.sources,
        timestamp: new Date().toISOString(),
        showQuickActions: data.show_quick_actions,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      
      // Extract and store data from the response in context
      updateContextWithExtractedData(
        data.answer, 
        inputMessage, 
        data.sources,
        data.answered_from
      );
    } catch (err) {
      console.error("Error sending message:", err);
      setError("Failed to get response from Zoe. Please try again.");
      
      // Add error message to chat
      const errorMessage: Message = {
        role: "assistant",
        content: "I'm sorry, I encountered an error. Please try again.",
        confidence: "error",
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle quick action button clicks
  const handleQuickAction = (question: string) => {
    setInputMessage(question);
    // Use setTimeout to ensure state is updated before sending
    setTimeout(() => {
      const syntheticEvent = { key: "Enter", shiftKey: false, preventDefault: () => {} } as React.KeyboardEvent;
      // Directly trigger the send with the question
      sendMessageWithQuery(question);
    }, 0);
  };

  // Helper to send a specific query (used by quick actions)
  const sendMessageWithQuery = async (query: string) => {
    if (!query.trim() || !selectedArtist || !user) {
      return;
    }

    const userMessage: Message = {
      role: "user",
      content: query,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/zoe/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          project_id: selectedProject,
          contract_ids: selectedContracts.length > 0 ? selectedContracts : null,
          user_id: user.id,
          session_id: sessionId,
          artist_id: selectedArtist,
          context: conversationContext, // Send conversation context
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response from Zoe");
      }

      const data = await response.json();

      if (data.session_id && data.session_id !== sessionId) {
        setSessionId(data.session_id);
      }

      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        confidence: data.confidence,
        sources: data.sources,
        timestamp: new Date().toISOString(),
        showQuickActions: data.show_quick_actions,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      
      // Extract and store data from the response in context
      updateContextWithExtractedData(
        data.answer,
        query,
        data.sources,
        data.answered_from
      );
    } catch (err) {
      console.error("Error sending message:", err);
      setError("Failed to get response from Zoe. Please try again.");
      
      const errorMessage: Message = {
        role: "assistant",
        content: "I'm sorry, I encountered an error. Please try again.",
        confidence: "error",
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

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
            {/* Context summary badge - visible when sidebar is closed */}
            {!sidebarOpen && selectedProject && (
              <Badge variant="secondary" className="hidden sm:flex items-center gap-1.5 text-xs">
                <span className="max-w-[100px] truncate">{selectedArtistName}</span>
                <span className="text-muted-foreground">•</span>
                <span className="max-w-[100px] truncate">{selectedProjectName}</span>
                {selectedContracts.length > 0 && (
                  <>
                    <span className="text-muted-foreground">•</span>
                    <span>{selectedContracts.length} contracts</span>
                  </>
                )}
              </Badge>
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
          {/* Chat Messages */}
          <ScrollArea className="flex-1">
            <div className="max-w-3xl mx-auto px-4 sm:px-6 py-6">
              <div className="space-y-4">
                {messages.length === 0 ? (
                  <div className="text-center py-16">
                    <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-4">
                      <Bot className="w-8 h-8 text-primary" />
                    </div>
                    <h3 className="text-xl font-semibold mb-2">Hi, I'm Zoe!</h3>
                    <p className="text-muted-foreground mb-8 max-w-md mx-auto">
                      {!selectedArtist 
                        ? "Select an artist from the sidebar to start asking questions."
                        : selectedProject 
                          ? "I can help you understand your contracts and artist info. Ask me about royalty splits, payment terms, or artist details."
                          : `I can tell you about ${selectedArtistName || 'the artist'}. Select a project to also ask about contracts.`}
                    </p>
                    
                    {/* Quick Action Buttons */}
                    {selectedArtist && (
                      <div className="flex flex-wrap justify-center gap-2 max-w-lg mx-auto">
                        {/* Artist quick actions - always available when artist selected */}
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleQuickAction(`What are ${selectedArtistName || 'the artist'}'s social media links?`)}
                          disabled={isLoading}
                          className="text-sm"
                        >
                          📱 Social Media
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleQuickAction(`Tell me about ${selectedArtistName || 'the artist'}`)}
                          disabled={isLoading}
                          className="text-sm"
                        >
                          🎤 Artist Overview
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleQuickAction(`What is ${selectedArtistName || 'the artist'}'s bio`)}
                          disabled={isLoading}
                          className="text-sm"
                        >
                          📄 Artist Bio
                        </Button>
                        {/* Contract quick actions - only when project selected */}
                        {selectedProject && (
                          <>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleQuickAction("What are the streaming royalty splits in this contract?")}
                              disabled={isLoading}
                              className="text-sm"
                            >
                              💰 Royalty Splits
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleQuickAction("What are the payment terms in this contract?")}
                              disabled={isLoading}
                              className="text-sm"
                            >
                              📅 Payment Terms
                            </Button>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  messages.map((message, index) => (
                    <div
                      key={index}
                      className={cn(
                        "flex gap-3",
                        message.role === "user" ? "justify-end" : "justify-start"
                      )}
                    >
                      {message.role === "assistant" && (
                        <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-1">
                          <Bot className="w-4 h-4 text-primary" />
                        </div>
                      )}
                      
                      <div
                        className={cn(
                          "max-w-[85%] rounded-2xl px-4 py-3",
                          message.role === "user"
                            ? "bg-primary text-primary-foreground"
                            : "bg-muted"
                        )}
                      >
                        <div className="text-sm leading-relaxed prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-headings:my-2 prose-ul:my-1 prose-ol:my-1">
                          <ReactMarkdown>{message.content}</ReactMarkdown>
                        </div>
                        
                        {/* Quick Action Buttons in greeting responses */}
                        {message.role === "assistant" && message.showQuickActions && (
                          <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t border-border/50">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleQuickAction("What are the royalty splits in this contract?")}
                              disabled={isLoading}
                              className="text-xs h-7"
                            >
                              💰 Royalty Splits
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleQuickAction("Who are the parties involved in this contract?")}
                              disabled={isLoading}
                              className="text-xs h-7"
                            >
                              👥 Involved Parties
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleQuickAction("What are the payment terms in this contract?")}
                              disabled={isLoading}
                              className="text-xs h-7"
                            >
                              📅 Payment Terms
                            </Button>
                          </div>
                        )}
                      </div>

                      {message.role === "user" && (
                        <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0 mt-1">
                          <User className="w-4 h-4 text-primary-foreground" />
                        </div>
                      )}
                    </div>
                  ))
                )}
                
                {isLoading && (
                  <div className="flex gap-3 justify-start">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-1">
                      <Bot className="w-4 h-4 text-primary animate-pulse" />
                    </div>
                    <div className="bg-muted rounded-2xl px-4 py-3">
                      <div className="flex items-center gap-2">
                        <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                        <p className="text-sm text-muted-foreground">Thinking...</p>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>
            </div>
          </ScrollArea>

          {/* Input Area */}
          <div className="border-t border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 flex-shrink-0">
            <div className="max-w-3xl mx-auto p-4">
              {error && (
                <Alert variant="destructive" className="mb-4">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
              
              <div className="flex gap-2 items-center">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setUploadModalOpen(true)}
                  disabled={!selectedProject}
                  className="h-10 w-10 rounded-full flex-shrink-0 text-muted-foreground hover:text-foreground"
                  title="Upload contract"
                >
                  <Paperclip className="w-5 h-5" />
                </Button>
                <Input
                  placeholder={
                    !selectedArtist
                      ? "Select an artist to start chatting..."
                      : selectedProject
                        ? "Ask about contracts or artist info..."
                        : "Ask about the artist (select a project for contract questions)..."
                  }
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  disabled={!selectedArtist || isLoading}
                  className="flex-1 h-11 rounded-full px-4 bg-muted/50 border-muted"
                />
                <Button
                  onClick={handleSendMessage}
                  disabled={!selectedArtist || !inputMessage.trim() || isLoading}
                  size="icon"
                  className="h-11 w-11 rounded-full flex-shrink-0"
                >
                  <Send className="w-4 h-4" />
                </Button>
              </div>
              
              <p className="text-[11px] text-center text-muted-foreground mt-2">
                Zoe answers based on your artist profile and uploaded contracts
              </p>
            </div>
          </div>
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
    </div>
  );
};

export default Zoe;
