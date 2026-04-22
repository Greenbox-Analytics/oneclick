import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Send, Loader2 } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { useToast } from "@/hooks/use-toast";
import type { Tables } from "@/integrations/supabase/types";
import { API_URL, apiFetch } from "@/lib/apiFetch";

interface FileToShare {
  file_name: string;
  file_path: string;
  file_source: "project_file" | "audio_file";
  file_id: string;
}

interface FileShareDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  files: FileToShare[];
}

export const FileShareDialog = ({
  open,
  onOpenChange,
  files,
}: FileShareDialogProps) => {
  const { user } = useAuth();
  const { toast } = useToast();
  const [contacts, setContacts] = useState<Tables<"contacts">[]>([]);
  const [selectedContactId, setSelectedContactId] = useState<string>("");
  const [manualEmail, setManualEmail] = useState("");
  const [manualName, setManualName] = useState("");
  const [message, setMessage] = useState("");
  const [sending, setSending] = useState(false);

  useEffect(() => {
    if (open && user) {
      supabase
        .from("contacts")
        .select("*")
        .order("name")
        .then(({ data }) => {
          if (data) setContacts(data);
        });
    }
    if (!open) {
      setSelectedContactId("");
      setManualEmail("");
      setManualName("");
      setMessage("");
    }
  }, [open, user]);

  const selectedContact = contacts.find((c) => c.id === selectedContactId);
  const recipientEmail = selectedContact?.email || manualEmail;
  const recipientName = selectedContact?.name || manualName || undefined;

  const canSend = recipientEmail.trim() && files.length > 0 && !sending;

  const handleSend = async () => {
    if (!user || !canSend) return;
    setSending(true);

    try {
      await apiFetch(`${API_URL}/share/files`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contact_id: selectedContactId || null,
          recipient_email: recipientEmail,
          recipient_name: recipientName || null,
          files: files.map((f) => ({
            file_name: f.file_name,
            file_path: f.file_path,
            file_source: f.file_source,
            file_id: f.file_id,
          })),
          message: message || null,
        }),
      });

      toast({
        title: "Files shared",
        description: `Sent ${files.length} file${files.length > 1 ? "s" : ""} to ${recipientEmail}`,
      });
      onOpenChange(false);
    } catch (err: unknown) {
      toast({
        title: "Error",
        description: err instanceof Error ? err.message : "Failed to share files",
        variant: "destructive",
      });
    } finally {
      setSending(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl w-[calc(100vw-2rem)] rounded-xl border-border bg-card">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
              <Send className="w-4 h-4 text-primary" />
            </div>
            Share Files
          </DialogTitle>
          <DialogDescription>
            Send {files.length} file{files.length > 1 ? "s" : ""} via email. Links expire in 7 days.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Files being shared */}
          <div>
            <Label className="text-xs text-muted-foreground uppercase tracking-wider">Files</Label>
            <div className="mt-1.5 space-y-1">
              {files.map((f, i) => (
                <div
                  key={i}
                  className="text-sm px-3 py-2 bg-muted/60 rounded-lg truncate border border-border/50"
                >
                  {f.file_name}
                </div>
              ))}
            </div>
          </div>

          {/* Contact picker */}
          <div>
            <Label className="text-sm">Send to contact</Label>
            <Select
              value={selectedContactId}
              onValueChange={(val) => {
                setSelectedContactId(val);
                if (val) {
                  setManualEmail("");
                  setManualName("");
                }
              }}
            >
              <SelectTrigger className="mt-1.5 bg-background border-border">
                <SelectValue placeholder="Choose a contact (optional)" />
              </SelectTrigger>
              <SelectContent className="bg-card border-border">
                {contacts
                  .filter((c) => c.email)
                  .map((c) => (
                    <SelectItem key={c.id} value={c.id}>
                      {c.name} ({c.email})
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          </div>

          {/* Manual email for non-contacts */}
          {!selectedContactId && (
            <>
              <div>
                <Label htmlFor="share-email" className="text-sm">
                  Or enter email <span className="text-primary">*</span>
                </Label>
                <Input
                  id="share-email"
                  type="email"
                  value={manualEmail}
                  onChange={(e) => setManualEmail(e.target.value)}
                  placeholder="recipient@example.com"
                  className="mt-1.5 bg-background border-border"
                />
              </div>
              <div>
                <Label htmlFor="share-name" className="text-sm">Recipient name</Label>
                <Input
                  id="share-name"
                  value={manualName}
                  onChange={(e) => setManualName(e.target.value)}
                  placeholder="Optional"
                  className="mt-1.5 bg-background border-border"
                />
              </div>
            </>
          )}

          {/* Optional message */}
          <div>
            <Label htmlFor="share-message" className="text-sm">Message</Label>
            <Textarea
              id="share-message"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Add a note for the recipient..."
              rows={3}
              className="mt-1.5 bg-background border-border resize-none"
            />
          </div>
        </div>

        <DialogFooter className="gap-2 pt-2">
          <Button variant="outline" onClick={() => onOpenChange(false)} className="border-border">
            Cancel
          </Button>
          <Button onClick={handleSend} disabled={!canSend}>
            {sending ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Send className="w-4 h-4 mr-2" />
            )}
            {sending ? "Sending..." : "Send"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
