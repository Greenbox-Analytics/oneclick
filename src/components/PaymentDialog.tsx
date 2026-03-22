import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  DollarSign,
  Loader2,
  CheckCircle,
  Copy,
  Building2,
  Mail,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import type { Tables } from "@/integrations/supabase/types";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface PaymentDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  contact: Tables<"contacts"> | null;
  onPaymentComplete: () => void;
}

export const PaymentDialog = ({
  open,
  onOpenChange,
  contact,
  onPaymentComplete,
}: PaymentDialogProps) => {
  const { user } = useAuth();
  const { toast } = useToast();
  const [amount, setAmount] = useState("");
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [recorded, setRecorded] = useState(false);

  if (!contact) return null;

  const currency = contact.bank_currency || "cad";
  const currencySymbol =
    currency === "eur" ? "€" : currency === "gbp" ? "£" : "$";
  const isCanadian = contact.bank_country === "CA";

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    toast({ title: `${label} copied` });
  };

  const handleRecordPayment = async () => {
    if (!user) return;
    const parsedAmount = parseFloat(amount);
    if (isNaN(parsedAmount) || parsedAmount <= 0) return;

    setSaving(true);
    try {
      const res = await fetch(`${API_BASE}/payments/record`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: user.id,
          contact_id: contact.id,
          party_name: contact.name,
          amount: parsedAmount,
          currency,
          metadata: note ? { note } : null,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to record payment");
      }

      setRecorded(true);
      setTimeout(() => {
        setRecorded(false);
        setAmount("");
        setNote("");
        onPaymentComplete();
      }, 1500);
    } catch (err: any) {
      toast({
        title: "Error",
        description: err.message,
        variant: "destructive",
      });
    } finally {
      setSaving(false);
    }
  };

  const handleClose = (open: boolean) => {
    if (!open) {
      setAmount("");
      setNote("");
      setRecorded(false);
    }
    onOpenChange(open);
  };

  if (recorded) {
    return (
      <Dialog open={open} onOpenChange={handleClose}>
        <DialogContent className="max-w-md">
          <div className="flex flex-col items-center gap-4 py-8">
            <CheckCircle className="w-16 h-16 text-green-500" />
            <p className="text-lg font-semibold">Payment Recorded</p>
            <p className="text-sm text-muted-foreground">
              {currencySymbol}
              {parseFloat(amount).toFixed(2)} {currency.toUpperCase()} to{" "}
              {contact.name}
            </p>
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-md max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Pay {contact.name}</DialogTitle>
          <DialogDescription>
            Send payment using the bank details below, then record it here
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Recipient info */}
          <div className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg">
            <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center text-primary font-semibold">
              {contact.name.charAt(0).toUpperCase()}
            </div>
            <div>
              <p className="font-medium">{contact.name}</p>
              <div className="flex items-center gap-1.5">
                {contact.role && (
                  <Badge variant="secondary" className="text-xs">
                    {contact.role}
                  </Badge>
                )}
              </div>
            </div>
          </div>

          {/* Bank details display */}
          <div className="border border-border rounded-lg p-3 space-y-2">
            <div className="flex items-center gap-2 text-sm font-medium mb-2">
              <Building2 className="w-4 h-4 text-muted-foreground" />
              <span>Bank Details</span>
              <Badge variant="outline" className="text-xs ml-auto">
                {isCanadian ? "Canadian" : "International"}
              </Badge>
            </div>

            {contact.bank_account_holder && (
              <DetailRow
                label="Account Holder"
                value={contact.bank_account_holder}
                onCopy={copyToClipboard}
              />
            )}

            {isCanadian ? (
              <>
                {contact.bank_transit_number && (
                  <DetailRow
                    label="Transit #"
                    value={contact.bank_transit_number}
                    onCopy={copyToClipboard}
                  />
                )}
                {contact.bank_institution_number && (
                  <DetailRow
                    label="Institution #"
                    value={contact.bank_institution_number}
                    onCopy={copyToClipboard}
                  />
                )}
                {contact.bank_account_number && (
                  <DetailRow
                    label="Account #"
                    value={contact.bank_account_number}
                    onCopy={copyToClipboard}
                  />
                )}
              </>
            ) : (
              <>
                {contact.bank_iban && (
                  <DetailRow
                    label="IBAN"
                    value={contact.bank_iban}
                    onCopy={copyToClipboard}
                  />
                )}
                {contact.bank_swift_bic && (
                  <DetailRow
                    label="SWIFT / BIC"
                    value={contact.bank_swift_bic}
                    onCopy={copyToClipboard}
                  />
                )}
              </>
            )}

            <DetailRow
              label="Currency"
              value={currency.toUpperCase()}
              onCopy={copyToClipboard}
            />
          </div>

          {/* Interac suggestion for Canadian */}
          {isCanadian && contact.email && (
            <div className="flex items-center gap-2 p-3 bg-blue-50 dark:bg-blue-950 rounded-lg text-sm">
              <Mail className="w-4 h-4 text-blue-600 dark:text-blue-400 flex-shrink-0" />
              <div>
                <span className="text-blue-800 dark:text-blue-300">
                  You can also send via Interac e-Transfer to{" "}
                </span>
                <button
                  onClick={() => copyToClipboard(contact.email!, "Email")}
                  className="font-medium text-blue-600 dark:text-blue-400 hover:underline"
                >
                  {contact.email}
                </button>
              </div>
            </div>
          )}

          {/* Record payment section */}
          <div className="border-t border-border pt-4 space-y-3">
            <p className="text-sm font-medium">Record Payment</p>
            <div>
              <Label htmlFor="pay-amount">
                Amount ({currency.toUpperCase()})
              </Label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">
                  {currencySymbol}
                </span>
                <Input
                  id="pay-amount"
                  type="number"
                  step="0.01"
                  min="0.01"
                  value={amount}
                  onChange={(e) => setAmount(e.target.value)}
                  placeholder="0.00"
                  className="pl-7"
                  disabled={saving}
                />
              </div>
            </div>

            <div>
              <Label htmlFor="pay-note">Note (optional)</Label>
              <Textarea
                id="pay-note"
                value={note}
                onChange={(e) => setNote(e.target.value)}
                placeholder="e.g. Royalty payment for Q1 2026"
                rows={2}
                disabled={saving}
              />
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => handleClose(false)} disabled={saving}>
            Cancel
          </Button>
          <Button
            onClick={handleRecordPayment}
            disabled={!amount || parseFloat(amount) <= 0 || saving}
          >
            {saving ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <DollarSign className="w-4 h-4 mr-1" />
                Record Payment
                {amount && parseFloat(amount) > 0
                  ? ` ${currencySymbol}${parseFloat(amount).toFixed(2)}`
                  : ""}
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

function DetailRow({
  label,
  value,
  onCopy,
}: {
  label: string;
  value: string;
  onCopy: (text: string, label: string) => void;
}) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-muted-foreground">{label}</span>
      <button
        onClick={() => onCopy(value, label)}
        className="flex items-center gap-1 font-mono hover:text-primary transition-colors"
      >
        {value}
        <Copy className="w-3 h-3 text-muted-foreground" />
      </button>
    </div>
  );
}
