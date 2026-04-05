import { useState } from "react";
import { useCreateAgreement, type Agreement } from "@/hooks/useRegistry";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import { Plus, Loader2, Trash2, FileText } from "lucide-react";

const AGREEMENT_TYPES = [
  "ownership_transfer", "split_agreement", "license_grant", "amendment", "termination",
];

interface Props {
  workId: string;
  agreements: Agreement[];
  isOwner: boolean;
}

export default function AgreementsPanel({ workId, agreements, isOwner }: Props) {
  const [showDialog, setShowDialog] = useState(false);
  const [agreementType, setAgreementType] = useState("");
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [effectiveDate, setEffectiveDate] = useState("");
  const [parties, setParties] = useState<Array<{ name: string; role: string; email: string }>>([
    { name: "", role: "", email: "" },
  ]);

  const createAgreement = useCreateAgreement();

  const resetForm = () => {
    setAgreementType(""); setTitle(""); setDescription(""); setEffectiveDate("");
    setParties([{ name: "", role: "", email: "" }]);
    setShowDialog(false);
  };

  const addParty = () => {
    setParties([...parties, { name: "", role: "", email: "" }]);
  };

  const removeParty = (index: number) => {
    if (parties.length <= 1) return;
    setParties(parties.filter((_, i) => i !== index));
  };

  const updateParty = (index: number, field: "name" | "role" | "email", value: string) => {
    const updated = [...parties];
    updated[index] = { ...updated[index], [field]: value };
    setParties(updated);
  };

  const handleSubmit = async () => {
    if (!agreementType || !title || !effectiveDate) return;
    const validParties = parties
      .filter((p) => p.name.trim() && p.role.trim())
      .map((p) => ({
        name: p.name.trim(),
        role: p.role.trim(),
        email: p.email.trim() || undefined,
      }));
    if (validParties.length === 0) return;
    await createAgreement.mutateAsync({
      work_id: workId,
      agreement_type: agreementType,
      title: title.trim(),
      description: description.trim() || undefined,
      effective_date: effectiveDate,
      parties: validParties,
    });
    resetForm();
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Agreements</CardTitle>
          {isOwner && (
            <Dialog open={showDialog} onOpenChange={(open) => { if (!open) resetForm(); setShowDialog(open); }}>
              <DialogTrigger asChild>
                <Button size="sm" variant="outline"><Plus className="w-3 h-3 mr-1" /> Record Agreement</Button>
              </DialogTrigger>
              <DialogContent className="max-w-lg">
                <DialogHeader>
                  <DialogTitle>Record Agreement</DialogTitle>
                </DialogHeader>
                <div className="space-y-3 pt-2 max-h-[70vh] overflow-y-auto">
                  <div>
                    <label className="text-sm font-medium">Agreement Type *</label>
                    <Select value={agreementType} onValueChange={setAgreementType}>
                      <SelectTrigger><SelectValue placeholder="Select type" /></SelectTrigger>
                      <SelectContent>
                        {AGREEMENT_TYPES.map((t) => (
                          <SelectItem key={t} value={t}>
                            {t.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Title *</label>
                    <Input value={title} onChange={(e) => setTitle(e.target.value)}
                      placeholder="Agreement title" />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Description</label>
                    <Textarea value={description} onChange={(e) => setDescription(e.target.value)}
                      placeholder="Describe the agreement..." rows={3} />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Effective Date *</label>
                    <Input type="date" value={effectiveDate}
                      onChange={(e) => setEffectiveDate(e.target.value)} />
                  </div>
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="text-sm font-medium">Parties *</label>
                      <Button type="button" size="sm" variant="outline" onClick={addParty}>
                        <Plus className="w-3 h-3 mr-1" /> Add Party
                      </Button>
                    </div>
                    <div className="space-y-2">
                      {parties.map((party, idx) => (
                        <div key={idx} className="flex items-start gap-2 p-2 rounded border bg-muted/30">
                          <div className="flex-1 space-y-2">
                            <div className="grid grid-cols-2 gap-2">
                              <Input placeholder="Name *" value={party.name}
                                onChange={(e) => updateParty(idx, "name", e.target.value)} />
                              <Input placeholder="Role *" value={party.role}
                                onChange={(e) => updateParty(idx, "role", e.target.value)} />
                            </div>
                            <Input placeholder="Email (optional)" value={party.email}
                              onChange={(e) => updateParty(idx, "email", e.target.value)} />
                          </div>
                          {parties.length > 1 && (
                            <Button size="icon" variant="ghost" className="h-7 w-7 text-destructive mt-1"
                              onClick={() => removeParty(idx)}>
                              <Trash2 className="w-3 h-3" />
                            </Button>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                  <Button onClick={handleSubmit} disabled={createAgreement.isPending} className="w-full">
                    {createAgreement.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    Record Agreement
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {agreements.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-4">
            No agreements recorded
          </p>
        ) : (
          <div className="relative">
            {/* Timeline line */}
            <div className="absolute left-4 top-0 bottom-0 w-px bg-border" />
            <div className="space-y-4">
              {agreements.map((agreement) => (
                <div key={agreement.id} className="relative pl-10">
                  {/* Timeline dot */}
                  <div className="absolute left-2.5 top-3 w-3 h-3 rounded-full bg-primary border-2 border-background" />
                  <div className="p-3 rounded-lg border">
                    <div className="flex items-center gap-2 mb-1">
                      <FileText className="w-4 h-4 text-muted-foreground" />
                      <span className="font-medium text-sm">{agreement.title}</span>
                      <Badge variant="outline">
                        {agreement.agreement_type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                      </Badge>
                    </div>
                    <div className="text-xs text-muted-foreground mb-1">
                      Effective: {agreement.effective_date}
                      {agreement.document_hash && (
                        <span className="ml-2">Hash: {agreement.document_hash.slice(0, 12)}...</span>
                      )}
                    </div>
                    {agreement.description && (
                      <p className="text-xs text-muted-foreground mb-2">{agreement.description}</p>
                    )}
                    {agreement.parties.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {agreement.parties.map((party, idx) => (
                          <Badge key={idx} variant="secondary" className="text-xs">
                            {party.name} ({party.role})
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
