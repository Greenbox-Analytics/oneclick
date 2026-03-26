import { useState, useEffect } from "react";
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
import { ChevronDown, ChevronUp, Building2 } from "lucide-react";
import type { Tables } from "@/integrations/supabase/types";

const ROLES = [
  "Producer",
  "Songwriter",
  "Featured Artist",
  "Manager",
  "Engineer",
  "Publisher",
  "Label",
  "Distributor",
  "Lawyer",
  "Other",
];

const CURRENCIES = ["cad", "usd", "eur", "gbp"];

interface ContactFormDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  contact?: Tables<"contacts"> | null;
  onSave: (data: {
    name: string;
    email: string;
    phone: string;
    role: string;
    notes: string;
    bank_country: string;
    bank_account_holder: string;
    bank_transit_number: string;
    bank_institution_number: string;
    bank_account_number: string;
    bank_iban: string;
    bank_swift_bic: string;
    bank_currency: string;
  }) => Promise<void>;
}

export const ContactFormDialog = ({
  open,
  onOpenChange,
  contact,
  onSave,
}: ContactFormDialogProps) => {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [role, setRole] = useState("");
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);
  const [showBankDetails, setShowBankDetails] = useState(false);

  // Bank detail fields
  const [bankCountry, setBankCountry] = useState("CA");
  const [bankAccountHolder, setBankAccountHolder] = useState("");
  const [bankTransitNumber, setBankTransitNumber] = useState("");
  const [bankInstitutionNumber, setBankInstitutionNumber] = useState("");
  const [bankAccountNumber, setBankAccountNumber] = useState("");
  const [bankIban, setBankIban] = useState("");
  const [bankSwiftBic, setBankSwiftBic] = useState("");
  const [bankCurrency, setBankCurrency] = useState("cad");

  const isEdit = !!contact;

  useEffect(() => {
    if (contact) {
      setName(contact.name);
      setEmail(contact.email || "");
      setPhone(contact.phone || "");
      setRole(contact.role || "");
      setNotes(contact.notes || "");
      setBankCountry(contact.bank_country || "CA");
      setBankAccountHolder(contact.bank_account_holder || "");
      setBankTransitNumber(contact.bank_transit_number || "");
      setBankInstitutionNumber(contact.bank_institution_number || "");
      setBankAccountNumber(contact.bank_account_number || "");
      setBankIban(contact.bank_iban || "");
      setBankSwiftBic(contact.bank_swift_bic || "");
      setBankCurrency(contact.bank_currency || "cad");
      // Auto-expand bank section if bank details exist
      const hasBankDetails = contact.bank_account_number || contact.bank_iban;
      setShowBankDetails(!!hasBankDetails);
    } else {
      setName("");
      setEmail("");
      setPhone("");
      setRole("");
      setNotes("");
      setBankCountry("CA");
      setBankAccountHolder("");
      setBankTransitNumber("");
      setBankInstitutionNumber("");
      setBankAccountNumber("");
      setBankIban("");
      setBankSwiftBic("");
      setBankCurrency("cad");
      setShowBankDetails(false);
    }
  }, [contact, open]);

  const handleSubmit = async () => {
    if (!name.trim()) return;
    setSaving(true);
    try {
      await onSave({
        name: name.trim(),
        email,
        phone,
        role,
        notes,
        bank_country: bankCountry,
        bank_account_holder: bankAccountHolder,
        bank_transit_number: bankTransitNumber,
        bank_institution_number: bankInstitutionNumber,
        bank_account_number: bankAccountNumber,
        bank_iban: bankIban,
        bank_swift_bic: bankSwiftBic,
        bank_currency: bankCurrency,
      });
      onOpenChange(false);
    } finally {
      setSaving(false);
    }
  };

  const isCanadian = bankCountry === "CA";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{isEdit ? "Edit Contact" : "Add Contact"}</DialogTitle>
          <DialogDescription>
            {isEdit
              ? "Update contact information"
              : "Add a new collaborator or payee to your registry"}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div>
            <Label htmlFor="contact-name">
              Name <span className="text-green-500">*</span>
            </Label>
            <Input
              id="contact-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Full name"
            />
          </div>

          <div>
            <Label htmlFor="contact-email">Email</Label>
            <Input
              id="contact-email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="email@example.com"
            />
          </div>

          <div>
            <Label htmlFor="contact-phone">Phone</Label>
            <Input
              id="contact-phone"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              placeholder="+1 (555) 000-0000"
            />
          </div>

          <div>
            <Label htmlFor="contact-role">Role</Label>
            <Select value={role} onValueChange={setRole}>
              <SelectTrigger>
                <SelectValue placeholder="Select role" />
              </SelectTrigger>
              <SelectContent>
                {ROLES.map((r) => (
                  <SelectItem key={r} value={r}>
                    {r}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label htmlFor="contact-notes">Notes</Label>
            <Textarea
              id="contact-notes"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Any additional notes..."
              rows={3}
            />
          </div>

          {/* Bank Details Section */}
          <div className="border border-border rounded-lg">
            <button
              type="button"
              onClick={() => setShowBankDetails(!showBankDetails)}
              className="w-full flex items-center justify-between p-3 text-sm font-medium hover:bg-muted/50 transition-colors rounded-lg"
            >
              <div className="flex items-center gap-2">
                <Building2 className="w-4 h-4 text-muted-foreground" />
                <span>Bank Details</span>
                {(bankAccountNumber || bankIban) && (
                  <span className="text-xs text-green-500 font-normal">Added</span>
                )}
              </div>
              {showBankDetails ? (
                <ChevronUp className="w-4 h-4 text-muted-foreground" />
              ) : (
                <ChevronDown className="w-4 h-4 text-muted-foreground" />
              )}
            </button>

            {showBankDetails && (
              <div className="px-3 pb-3 space-y-3">
                <div>
                  <Label>Country</Label>
                  <Select value={bankCountry} onValueChange={setBankCountry}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="CA">Canada</SelectItem>
                      <SelectItem value="INTL">International</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="bank-holder">Account Holder Name</Label>
                  <Input
                    id="bank-holder"
                    value={bankAccountHolder}
                    onChange={(e) => setBankAccountHolder(e.target.value)}
                    placeholder={name || "Account holder name"}
                  />
                </div>

                {isCanadian ? (
                  <>
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <Label htmlFor="bank-transit">Transit Number</Label>
                        <Input
                          id="bank-transit"
                          value={bankTransitNumber}
                          onChange={(e) => setBankTransitNumber(e.target.value)}
                          placeholder="5 digits"
                          maxLength={5}
                        />
                      </div>
                      <div>
                        <Label htmlFor="bank-institution">Institution #</Label>
                        <Input
                          id="bank-institution"
                          value={bankInstitutionNumber}
                          onChange={(e) => setBankInstitutionNumber(e.target.value)}
                          placeholder="3 digits"
                          maxLength={3}
                        />
                      </div>
                    </div>
                    <div>
                      <Label htmlFor="bank-account">Account Number</Label>
                      <Input
                        id="bank-account"
                        value={bankAccountNumber}
                        onChange={(e) => setBankAccountNumber(e.target.value)}
                        placeholder="Account number"
                      />
                    </div>
                  </>
                ) : (
                  <>
                    <div>
                      <Label htmlFor="bank-iban">IBAN</Label>
                      <Input
                        id="bank-iban"
                        value={bankIban}
                        onChange={(e) => setBankIban(e.target.value)}
                        placeholder="e.g. GB29 NWBK 6016 1331 9268 19"
                      />
                    </div>
                    <div>
                      <Label htmlFor="bank-swift">SWIFT / BIC</Label>
                      <Input
                        id="bank-swift"
                        value={bankSwiftBic}
                        onChange={(e) => setBankSwiftBic(e.target.value)}
                        placeholder="e.g. NWBKGB2L"
                      />
                    </div>
                  </>
                )}

                <div>
                  <Label>Currency</Label>
                  <Select value={bankCurrency} onValueChange={setBankCurrency}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {CURRENCIES.map((c) => (
                        <SelectItem key={c} value={c}>
                          {c.toUpperCase()}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            )}
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={!name.trim() || saving}>
            {saving ? "Saving..." : isEdit ? "Save Changes" : "Add Contact"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
