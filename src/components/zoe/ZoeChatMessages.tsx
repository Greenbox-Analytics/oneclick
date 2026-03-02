import { RefObject } from "react";
import ReactMarkdown from "react-markdown";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Bot, User, Loader2, Copy, Check, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Message, AssistantQuickAction } from "@/hooks/useStreamingChat";

interface ZoeChatMessagesProps {
  messages: Message[];
  isStreaming: boolean;
  selectedArtist: string;
  selectedArtistName?: string;
  selectedProject: string;
  copiedMessageId: string | null;
  messagesEndRef: RefObject<HTMLDivElement>;
  onQuickAction: (question: string) => void;
  onAssistantQuickAction: (action: AssistantQuickAction) => void;
  onRetry: () => void;
  onCopyMessage: (content: string, messageId: string) => void;
}

export function ZoeChatMessages({
  messages,
  isStreaming,
  selectedArtist,
  selectedArtistName,
  selectedProject,
  copiedMessageId,
  messagesEndRef,
  onQuickAction,
  onAssistantQuickAction,
  onRetry,
  onCopyMessage,
}: ZoeChatMessagesProps) {
  return (
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
                    : `I can tell you about ${selectedArtistName || "the artist"}. Select a project to also ask about contracts.`}
              </p>

              {/* Quick Action Buttons */}
              {selectedArtist && (
                <div className="flex flex-wrap justify-center gap-2 max-w-lg mx-auto">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onQuickAction("What are the artist's social media links?")}
                    disabled={isStreaming}
                    className="text-sm"
                  >
                    📱 Social Media
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onQuickAction("Tell me about the artist")}
                    disabled={isStreaming}
                    className="text-sm"
                  >
                    🎤 Artist Overview
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onQuickAction("What is the artist's bio?")}
                    disabled={isStreaming}
                    className="text-sm"
                  >
                    📄 Artist Bio
                  </Button>
                  {selectedProject && (
                    <>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => onQuickAction("What are the streaming royalty splits in this contract?")}
                        disabled={isStreaming}
                        className="text-sm"
                      >
                        💰 Royalty Splits
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => onQuickAction("What are the payment terms in this contract?")}
                        disabled={isStreaming}
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
            messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  "flex gap-3 group/msg",
                  message.role === "user" ? "justify-end" : message.role === "system" ? "justify-center" : "justify-start"
                )}
              >
                {/* System divider message */}
                {message.role === "system" && (
                  <div className="flex items-center gap-2 py-2">
                    <div className="h-px flex-1 bg-border" />
                    <span className="text-xs text-muted-foreground px-2 whitespace-nowrap">
                      {message.content.replace(/^---\s*|\s*---$/g, "")}
                    </span>
                    <div className="h-px flex-1 bg-border" />
                  </div>
                )}

                {message.role === "assistant" && (
                  <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-1">
                    <Bot className={cn("w-4 h-4 text-primary", message.isStreaming && "animate-pulse")} />
                  </div>
                )}

                {message.role !== "system" && (
                  <div
                    className={cn(
                      "max-w-[85%] rounded-2xl px-4 py-3 relative",
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted"
                    )}
                  >
                    {/* Message content */}
                    {message.role === "assistant" && message.isStreaming && !message.content ? (
                      <div className="flex items-center gap-2">
                        <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                        <p className="text-sm text-muted-foreground">Thinking...</p>
                      </div>
                    ) : (
                      <div className="text-sm leading-relaxed prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-headings:my-2 prose-ul:my-1 prose-ol:my-1">
                        <ReactMarkdown>{message.content}</ReactMarkdown>
                        {message.isStreaming && (
                          <span className="inline-block w-1.5 h-4 bg-foreground/70 animate-pulse ml-0.5 align-text-bottom" />
                        )}
                      </div>
                    )}

                    {/* Copy button (assistant messages only, on hover) */}
                    {message.role === "assistant" && !message.isStreaming && message.content && (
                      <div className="absolute -bottom-3 right-2 opacity-0 group-hover/msg:opacity-100 transition-opacity flex gap-1">
                        <Button
                          variant="secondary"
                          size="sm"
                          className="h-6 px-2 text-[10px] shadow-sm"
                          onClick={() => onCopyMessage(message.content, message.id)}
                        >
                          {copiedMessageId === message.id ? (
                            <>
                              <Check className="w-3 h-3 mr-1" /> Copied
                            </>
                          ) : (
                            <>
                              <Copy className="w-3 h-3 mr-1" /> Copy
                            </>
                          )}
                        </Button>
                      </div>
                    )}

                    {/* Quick Action Buttons in greeting responses */}
                    {message.role === "assistant" &&
                      !message.isStreaming &&
                      message.quickActions &&
                      message.quickActions.length > 0 && (
                        <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t border-border/50">
                          {message.quickActions.map((action) => (
                            <Button
                              key={action.id}
                              variant="outline"
                              size="sm"
                              onClick={() => onAssistantQuickAction(action)}
                              disabled={isStreaming}
                              className="text-xs h-7"
                            >
                              {action.label}
                            </Button>
                          ))}
                        </div>
                      )}

                    {message.role === "assistant" &&
                      !message.isStreaming &&
                      message.showQuickActions &&
                      (!message.quickActions || message.quickActions.length === 0) && (
                        <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t border-border/50">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => onQuickAction("What are the royalty splits in this contract?")}
                            disabled={isStreaming}
                            className="text-xs h-7"
                          >
                            💰 Royalty Splits
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => onQuickAction("Who are the parties involved in this contract?")}
                            disabled={isStreaming}
                            className="text-xs h-7"
                          >
                            👥 Involved Parties
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => onQuickAction("What are the payment terms in this contract?")}
                            disabled={isStreaming}
                            className="text-xs h-7"
                          >
                            📅 Payment Terms
                          </Button>
                        </div>
                      )}
                  </div>
                )}

                {message.role === "user" && (
                  <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0 mt-1">
                    <User className="w-4 h-4 text-primary-foreground" />
                  </div>
                )}
              </div>
            ))
          )}

          {/* Retry button beneath last assistant message */}
          {messages.length > 0 &&
            !isStreaming &&
            messages[messages.length - 1]?.role === "assistant" && (
              <div className="flex justify-start pl-11">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onRetry}
                  className="text-xs h-7 text-muted-foreground hover:text-foreground gap-1.5"
                >
                  <RefreshCw className="w-3 h-3" />
                  Regenerate
                </Button>
              </div>
            )}

          <div ref={messagesEndRef} />
        </div>
      </div>
    </ScrollArea>
  );
}
