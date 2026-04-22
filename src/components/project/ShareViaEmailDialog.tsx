import { useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { API_URL, apiFetch } from "@/lib/apiFetch";

interface ShareViaEmailDialogProps {
  open: boolean;
  onClose: () => void;
  projectId: string;
  fileIds?: string[];
  audioFileIds?: string[];
  defaultSubject?: string;
}

interface ShareResponse {
  total_bytes: number;
  attachment_count: number;
}

export default function ShareViaEmailDialog({
  open,
  onClose,
  projectId,
  fileIds = [],
  audioFileIds = [],
  defaultSubject = "",
}: ShareViaEmailDialogProps) {
  const [recipient, setRecipient] = useState("");
  const [subject, setSubject] = useState(defaultSubject);
  const [message, setMessage] = useState("");
  const [sending, setSending] = useState(false);

  useEffect(() => {
    if (open) {
      setRecipient("");
      setSubject(defaultSubject);
      setMessage("");
    }
  }, [open, defaultSubject]);

  const totalItems = fileIds.length + audioFileIds.length;

  const handleSend = async () => {
    if (!recipient.trim()) {
      toast.error("Recipient email is required");
      return;
    }
    if (totalItems === 0) {
      toast.error("No files selected");
      return;
    }
    setSending(true);
    try {
      const res = await apiFetch<ShareResponse>(`${API_URL}/projects/${projectId}/share-email`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          recipient_email: recipient.trim(),
          subject: subject.trim() || "Files from Msanii",
          message: message.trim(),
          file_ids: fileIds,
          audio_file_ids: audioFileIds,
        }),
      });
      toast.success(
        `Sent ${res.attachment_count} file${res.attachment_count === 1 ? "" : "s"} (${(res.total_bytes / (1024 * 1024)).toFixed(1)} MB)`
      );
      onClose();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to send email";
      toast.error(message);
    } finally {
      setSending(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Share via email</DialogTitle>
          <DialogDescription>
            Send {totalItems} file{totalItems === 1 ? "" : "s"} as email attachments. Max 40 MB total.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <div className="space-y-1">
            <Label htmlFor="share-recipient">Recipient email</Label>
            <Input
              id="share-recipient"
              type="email"
              value={recipient}
              onChange={(e) => setRecipient(e.target.value)}
              placeholder="name@example.com"
              disabled={sending}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="share-subject">Subject</Label>
            <Input
              id="share-subject"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              placeholder="Files from Msanii"
              disabled={sending}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="share-message">Message (optional)</Label>
            <Textarea
              id="share-message"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              rows={4}
              disabled={sending}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={sending}>Cancel</Button>
          <Button onClick={handleSend} disabled={sending || !recipient.trim() || totalItems === 0}>
            {sending ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
            Send
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
