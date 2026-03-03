import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, Send, Square, Paperclip, X } from "lucide-react";

interface ContractInfo {
  id: string;
  file_name: string;
}

interface ZoeInputBarProps {
  inputMessage: string;
  onInputChange: (value: string) => void;
  error: string;
  isStreaming: boolean;
  isAtLimit: boolean;
  selectedArtist: string;
  selectedProject: string;
  selectedContracts: string[];
  contracts: ContractInfo[];
  onDeselectContract: (id: string) => void;
  onSend: () => void;
  onStop: () => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  onUploadClick: () => void;
}

export function ZoeInputBar({
  inputMessage,
  onInputChange,
  error,
  isStreaming,
  isAtLimit,
  selectedArtist,
  selectedProject,
  selectedContracts,
  contracts,
  onDeselectContract,
  onSend,
  onStop,
  onKeyDown,
  onUploadClick,
}: ZoeInputBarProps) {
  const selectedContractInfos = selectedContracts
    .map(id => contracts.find(c => c.id === id))
    .filter((c): c is ContractInfo => c !== undefined);

  return (
    <div className="border-t border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 flex-shrink-0">
      <div className="max-w-3xl mx-auto p-4">
        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {selectedContractInfos.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-2">
            {selectedContractInfos.map(contract => (
              <Badge
                key={contract.id}
                variant="secondary"
                className="pl-2 pr-1 py-1 gap-1 text-xs font-normal max-w-[200px]"
              >
                <Paperclip className="w-3 h-3 flex-shrink-0" />
                <span className="truncate">{contract.file_name}</span>
                <button
                  onClick={() => onDeselectContract(contract.id)}
                  className="ml-0.5 rounded-full p-0.5 hover:bg-muted-foreground/20 flex-shrink-0"
                  title={`Remove ${contract.file_name}`}
                >
                  <X className="w-3 h-3" />
                </button>
              </Badge>
            ))}
          </div>
        )}

        <div className="flex gap-2 items-center">
          <Button
            variant="ghost"
            size="icon"
            onClick={onUploadClick}
            disabled={!selectedProject}
            className="h-10 w-10 rounded-full flex-shrink-0 text-muted-foreground hover:text-foreground"
            title="Upload contract"
          >
            <Paperclip className="w-5 h-5" />
          </Button>
          <Input
            placeholder={
              isAtLimit
                ? "Conversation limit reached. Please refresh the page."
                : !selectedArtist
                  ? "Select an artist to start chatting..."
                  : selectedProject
                    ? "Ask about contracts or artist info..."
                    : "Ask about the artist (select a project for contract questions)..."
            }
            value={inputMessage}
            onChange={(e) => onInputChange(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={!selectedArtist || isAtLimit}
            className="flex-1 h-11 rounded-full px-4 bg-muted/50 border-muted"
          />
          {isStreaming ? (
            <Button
              onClick={onStop}
              size="icon"
              variant="destructive"
              className="h-11 w-11 rounded-full flex-shrink-0"
              title="Stop generating"
            >
              <Square className="w-4 h-4" />
            </Button>
          ) : (
            <Button
              onClick={onSend}
              disabled={!selectedArtist || !inputMessage.trim() || isAtLimit}
              size="icon"
              className="h-11 w-11 rounded-full flex-shrink-0"
            >
              <Send className="w-4 h-4" />
            </Button>
          )}
        </div>

        <p className="text-[11px] text-center text-muted-foreground mt-2">
          {isAtLimit
            ? "Conversation limit reached. Please refresh the page to continue."
            : "Zoe answers based on your artist profile and uploaded contracts"}
        </p>
      </div>
    </div>
  );
}
