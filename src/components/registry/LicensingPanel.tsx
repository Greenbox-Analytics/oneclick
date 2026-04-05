import { useState } from "react";
import {
  useCreateLicense, useUpdateLicense, useDeleteLicense,
  type LicensingRight,
} from "@/hooks/useRegistry";
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
import { Plus, Pencil, Trash2, Loader2 } from "lucide-react";

const LICENSE_TYPES = [
  "sync", "mechanical", "performance", "print", "digital",
  "exclusive", "non_exclusive", "other",
];

const STATUS_COLORS: Record<string, string> = {
  active: "bg-green-100 text-green-800",
  expired: "bg-gray-100 text-gray-800",
  pending: "bg-amber-100 text-amber-800",
  revoked: "bg-red-100 text-red-800",
};

interface Props {
  workId: string;
  licenses: LicensingRight[];
  isOwner: boolean;
}

export default function LicensingPanel({ workId, licenses, isOwner }: Props) {
  const [showDialog, setShowDialog] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [licenseType, setLicenseType] = useState("");
  const [licenseeName, setLicenseeName] = useState("");
  const [licenseeEmail, setLicenseeEmail] = useState("");
  const [territory, setTerritory] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [terms, setTerms] = useState("");

  const createLicense = useCreateLicense();
  const updateLicense = useUpdateLicense();
  const deleteLicense = useDeleteLicense();

  const resetForm = () => {
    setLicenseType(""); setLicenseeName(""); setLicenseeEmail("");
    setTerritory(""); setStartDate(""); setEndDate(""); setTerms("");
    setShowDialog(false); setEditingId(null);
  };

  const handleSubmit = async () => {
    if (!licenseType || !licenseeName || !startDate) return;
    if (editingId) {
      await updateLicense.mutateAsync({
        licenseId: editingId,
        license_type: licenseType,
        licensee_name: licenseeName,
        licensee_email: licenseeEmail || undefined,
        territory: territory || undefined,
        start_date: startDate,
        end_date: endDate || undefined,
        terms: terms || undefined,
      });
    } else {
      await createLicense.mutateAsync({
        work_id: workId,
        license_type: licenseType,
        licensee_name: licenseeName,
        licensee_email: licenseeEmail || undefined,
        territory: territory || undefined,
        start_date: startDate,
        end_date: endDate || undefined,
        terms: terms || undefined,
      });
    }
    resetForm();
  };

  const startEdit = (license: LicensingRight) => {
    setEditingId(license.id);
    setLicenseType(license.license_type);
    setLicenseeName(license.licensee_name);
    setLicenseeEmail(license.licensee_email || "");
    setTerritory(license.territory || "");
    setStartDate(license.start_date);
    setEndDate(license.end_date || "");
    setTerms(license.terms || "");
    setShowDialog(true);
  };

  const isPending = createLicense.isPending || updateLicense.isPending;

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Licensing Rights</CardTitle>
          {isOwner && (
            <Dialog open={showDialog} onOpenChange={(open) => { if (!open) resetForm(); setShowDialog(open); }}>
              <DialogTrigger asChild>
                <Button size="sm" variant="outline"><Plus className="w-3 h-3 mr-1" /> Add License</Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>{editingId ? "Edit" : "Add"} License</DialogTitle>
                </DialogHeader>
                <div className="space-y-3 pt-2">
                  <div>
                    <label className="text-sm font-medium">License Type *</label>
                    <Select value={licenseType} onValueChange={setLicenseType}>
                      <SelectTrigger><SelectValue placeholder="Select license type" /></SelectTrigger>
                      <SelectContent>
                        {LICENSE_TYPES.map((t) => (
                          <SelectItem key={t} value={t}>
                            {t.replace("_", " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-sm font-medium">Licensee Name *</label>
                      <Input value={licenseeName} onChange={(e) => setLicenseeName(e.target.value)}
                        placeholder="Licensee name" />
                    </div>
                    <div>
                      <label className="text-sm font-medium">Licensee Email</label>
                      <Input type="email" value={licenseeEmail} onChange={(e) => setLicenseeEmail(e.target.value)}
                        placeholder="licensee@example.com" />
                    </div>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Territory</label>
                    <Input value={territory} onChange={(e) => setTerritory(e.target.value)}
                      placeholder="e.g. Worldwide, US, EU" />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-sm font-medium">Start Date *</label>
                      <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
                    </div>
                    <div>
                      <label className="text-sm font-medium">End Date</label>
                      <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
                    </div>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Terms</label>
                    <Textarea value={terms} onChange={(e) => setTerms(e.target.value)}
                      placeholder="License terms and conditions..." rows={3} />
                  </div>
                  <Button onClick={handleSubmit} disabled={isPending} className="w-full">
                    {isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    {editingId ? "Update" : "Add"} License
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {licenses.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-4">
            No licensing rights recorded
          </p>
        ) : (
          <div className="space-y-2">
            {licenses.map((license) => (
              <div key={license.id} className="flex items-center justify-between p-3 rounded-lg border">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm">{license.licensee_name}</span>
                    <Badge variant="outline">
                      {license.license_type.replace("_", " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                    </Badge>
                    <Badge className={STATUS_COLORS[license.status] || "bg-gray-100 text-gray-800"}>
                      {license.status}
                    </Badge>
                  </div>
                  <div className="text-xs text-muted-foreground mt-0.5">
                    {license.territory && <span>{license.territory} · </span>}
                    <span>{license.start_date}</span>
                    {license.end_date && <span> to {license.end_date}</span>}
                    {license.licensee_email && <span> · {license.licensee_email}</span>}
                  </div>
                  {license.terms && (
                    <p className="text-xs text-muted-foreground mt-1 line-clamp-2">{license.terms}</p>
                  )}
                </div>
                {isOwner && (
                  <div className="flex items-center gap-1 ml-2">
                    <Button size="icon" variant="ghost" className="h-7 w-7"
                      onClick={() => startEdit(license)}>
                      <Pencil className="w-3 h-3" />
                    </Button>
                    <Button size="icon" variant="ghost" className="h-7 w-7 text-destructive"
                      onClick={() => deleteLicense.mutate(license.id)}>
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
