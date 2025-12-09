import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Music, ArrowLeft, Send, Bot, User, AlertCircle, X, Upload, Trash2 } from "lucide-react";
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

// Backend API URL
const API_URL = "http://localhost:8000";

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
}

const Zoe = () => {
  const navigate = useNavigate();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { user } = useAuth();

  // State for artists, projects and contracts
  const [artists, setArtists] = useState<Artist[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [contracts, setContracts] = useState<Contract[]>([]);
  const [selectedArtist, setSelectedArtist] = useState<string>("");
  const [selectedProject, setSelectedProject] = useState<string>("");
  const [selectedContracts, setSelectedContracts] = useState<string[]>([]);
  
  // Chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  // Upload/Delete state
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [contractToDelete, setContractToDelete] = useState<Contract | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Fetch artists on mount
  useEffect(() => {
    fetch(`${API_URL}/artists`)
      .then((res) => res.json())
      .then((data) => setArtists(data))
      .catch((err) => {
        console.error("Error fetching artists:", err);
        setError("Failed to load artists");
      });
  }, []);

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
  }, [selectedProject]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleUploadComplete = () => {
    // Refresh contracts list after upload
    fetchContracts();
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

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !selectedProject || !user) {
      setError("Please select a project and enter a message");
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
      // Call backend chatbot API
      const response = await fetch(`${API_URL}/zoe/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: inputMessage,
          project_id: selectedProject,
          contract_ids: selectedContracts.length > 0 ? selectedContracts : null,
          user_id: user.id,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response from Zoe");
      }

      const data = await response.json();

      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        confidence: data.confidence,
        sources: data.sources,
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
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

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      <header className="border-b border-border bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/60 sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary to-primary/80 flex items-center justify-center shadow-lg">
              <Bot className="w-6 h-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">Zoe AI Assistant</h1>
              <p className="text-xs text-muted-foreground">Contract Intelligence</p>
            </div>
          </div>
          <Button variant="outline" onClick={() => navigate("/tools")} className="gap-2">
            <ArrowLeft className="w-4 h-4" />
            Back to Tools
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6 max-w-5xl">
        {/* Chat Interface - Full Height */}
        <Card className="flex flex-col shadow-xl border-2" style={{ height: 'calc(100vh - 180px)' }}>
          {/* Header with Context Selection */}
          <CardHeader className="border-b bg-muted/30 pb-4">
            <div className="flex items-start justify-between gap-4 mb-4">
              <div>
                <CardTitle className="text-2xl flex items-center gap-2">
                  <Bot className="w-6 h-6 text-primary" />
                  Ask Zoe Anything
                </CardTitle>
                <CardDescription className="mt-1">
                  Get instant answers about royalty splits, payment terms, contract duration, and more
                </CardDescription>
              </div>
            </div>
            
            {/* Artist and Project Selection */}
            <div className="space-y-3">
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                  Select Artist *
                </label>
                <Select value={selectedArtist} onValueChange={setSelectedArtist}>
                  <SelectTrigger className="h-10">
                    <SelectValue placeholder="Choose an artist" />
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

              {selectedArtist && (
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                    Select Project *
                  </label>
                  <Select value={selectedProject} onValueChange={setSelectedProject} disabled={!selectedArtist}>
                    <SelectTrigger className="h-10">
                      <SelectValue placeholder="Choose a project" />
                    </SelectTrigger>
                    <SelectContent>
                      {projects.map((project) => (
                        <SelectItem key={project.id} value={project.id}>
                          {project.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}

              {/* Contract Management */}
              {selectedProject && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                      Contracts {contracts.length > 0 && `(${contracts.length})`}
                    </label>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setUploadModalOpen(true)}
                      className="h-7 text-xs"
                    >
                      <Upload className="w-3 h-3 mr-1" />
                      Upload
                    </Button>
                  </div>

                  {contracts.length > 0 ? (
                    <div className="bg-muted/50 rounded-lg p-3 max-h-32 overflow-y-auto">
                      <div className="space-y-2">
                        {contracts.map((contract) => (
                          <div key={contract.id} className="flex items-center justify-between gap-2 group">
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
                                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer truncate"
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
                    </div>
                  ) : (
                    <div className="bg-muted/50 rounded-lg p-4 text-center">
                      <p className="text-sm text-muted-foreground mb-2">No contracts uploaded yet</p>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setUploadModalOpen(true)}
                      >
                        <Upload className="w-3 h-3 mr-1" />
                        Upload First Contract
                      </Button>
                    </div>
                  )}
                  {selectedContracts.length > 0 && (
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary" className="text-xs">
                        {selectedContracts.length} contract{selectedContracts.length > 1 ? 's' : ''} selected
                      </Badge>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 text-xs"
                        onClick={() => setSelectedContracts([])}
                      >
                        Clear
                      </Button>
                    </div>
                  )}
                </div>
              )}

              {selectedProject && (
                <div className="flex items-center gap-2 pt-1">
                  <Badge variant="outline" className="text-xs">
                    {selectedContracts.length > 0 
                      ? `🎯 Searching ${selectedContracts.length} specific contract${selectedContracts.length > 1 ? 's' : ''}`
                      : `📁 Searching all contracts in project`}
                  </Badge>
                </div>
              )}
            </div>
          </CardHeader>

          <CardContent className="flex-1 flex flex-col p-0 overflow-hidden">
              {/* Messages Area */}
              <ScrollArea className="flex-1 px-6">
                <div className="space-y-4 py-4">
                  {messages.length === 0 ? (
                    <div className="text-center py-12 text-muted-foreground">
                      <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
                      <p className="text-lg font-medium mb-2">No messages yet</p>
                      <p className="text-sm">
                        Select a project and start asking questions about your contracts
                      </p>
                    </div>
                  ) : (
                    messages.map((message, index) => (
                      <div
                        key={index}
                        className={`flex gap-3 ${
                          message.role === "user" ? "justify-end" : "justify-start"
                        }`}
                      >
                        {message.role === "assistant" && (
                          <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                            <Bot className="w-5 h-5 text-primary" />
                          </div>
                        )}
                        
                        <div
                          className={`max-w-[80%] rounded-lg p-4 ${
                            message.role === "user"
                              ? "bg-primary text-primary-foreground"
                              : "bg-muted"
                          }`}
                        >
                          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                          
                          {message.confidence && message.role === "assistant" && (
                            <div className="mt-3 pt-3 border-t border-border/50">
                              <Badge
                                variant={
                                  message.confidence === "high"
                                    ? "default"
                                    : message.confidence === "low"
                                    ? "secondary"
                                    : "destructive"
                                }
                                className="text-xs"
                              >
                                {message.confidence} confidence
                              </Badge>
                            </div>
                          )}
                          
                          {message.sources && message.sources.length > 0 && (
                            <div className="mt-3 pt-3 border-t border-border/50">
                              <p className="text-xs font-medium mb-2">Sources:</p>
                              <div className="space-y-1">
                                {message.sources.slice(0, 3).map((source, idx) => (
                                  <p key={idx} className="text-xs text-muted-foreground">
                                    • {source.contract_file} (Page {source.page_number}) - Score:{" "}
                                    {source.score.toFixed(2)}
                                  </p>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>

                        {message.role === "user" && (
                          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                            <User className="w-5 h-5 text-primary-foreground" />
                          </div>
                        )}
                      </div>
                    ))
                  )}
                  
                  {isLoading && (
                    <div className="flex gap-3 justify-start">
                      <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                        <Bot className="w-5 h-5 text-primary animate-pulse" />
                      </div>
                      <div className="bg-muted rounded-lg p-4">
                        <p className="text-sm text-muted-foreground">Thinking...</p>
                      </div>
                    </div>
                  )}
                  
                  <div ref={messagesEndRef} />
                </div>
              </ScrollArea>

              {/* Input Area */}
              <div className="border-t border-border p-4">
                {error && (
                  <Alert variant="destructive" className="mb-4">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}
                
                <div className="flex gap-2">
                  <Input
                    placeholder={
                      selectedProject
                        ? "Ask a question about your contracts..."
                        : "Please select a project first"
                    }
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    disabled={!selectedProject || isLoading}
                    className="flex-1"
                  />
                  <Button
                    onClick={handleSendMessage}
                    disabled={!selectedProject || !inputMessage.trim() || isLoading}
                  >
                    <Send className="w-4 h-4" />
                  </Button>
                </div>
                
                <p className="text-xs text-muted-foreground mt-2">
                  Zoe only answers based on your uploaded contracts, using no outside knowledge.
                </p>
              </div>
            </CardContent>
          </Card>
      </main>

      {/* Upload Modal */}
      {selectedProject && (
        <ContractUploadModal
          open={uploadModalOpen}
          onOpenChange={setUploadModalOpen}
          projectId={selectedProject}
          onUploadComplete={handleUploadComplete}
        />
      )}

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
