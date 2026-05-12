import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";

interface Props {
  open: boolean;
  onClose: () => void;
}

export const UpgradeRequestModal = ({ open, onClose }: Props) => {
  const { user } = useAuth();
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setEmail(user?.email ?? "");
      setMessage("");
      setSubmitted(false);
      setError(null);
    }
  }, [open, user?.email]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!email.trim()) {
      setError("Email is required");
      return;
    }
    setSubmitting(true);
    try {
      // apiFetch automatically attaches auth header if logged in;
      // for logged-out users it just skips it — endpoint is public.
      await apiFetch(`${API_URL}/pro-requests`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: email.trim(),
          message: message.trim() || null,
        }),
      });
      setSubmitted(true);
    } catch (err) {
      setError(
        `Couldn't submit your request — ${(err as Error).message}. You can also email tech@greenboxanalytics.ca directly.`
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="sm:max-w-md">
        {submitted ? (
          <>
            <DialogHeader>
              <DialogTitle>Thanks — we'll be in touch</DialogTitle>
              <DialogDescription>
                We've recorded your request and will reach out within 48 hours.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button onClick={onClose}>Close</Button>
            </DialogFooter>
          </>
        ) : (
          <form onSubmit={handleSubmit}>
            <DialogHeader>
              <DialogTitle>Request Pro access</DialogTitle>
              <DialogDescription>
                Stripe checkout is coming soon. Tell us a bit about what you're
                looking for and we'll be in touch.
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="upgrade-email">Email</Label>
                <Input
                  id="upgrade-email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@band.com"
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="upgrade-message">Message (optional)</Label>
                <Textarea
                  id="upgrade-message"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Anything we should know? (use case, team size, urgency)"
                  rows={3}
                />
              </div>
              {error && (
                <div className="text-sm text-destructive">{error}</div>
              )}
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="ghost"
                onClick={onClose}
                disabled={submitting}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={submitting}>
                {submitting ? "Submitting…" : "Submit request"}
              </Button>
            </DialogFooter>
          </form>
        )}
      </DialogContent>
    </Dialog>
  );
};
