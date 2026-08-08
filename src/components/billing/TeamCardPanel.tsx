// src/components/billing/TeamCardPanel.tsx
// Design's TeamCard summary (preview + Configure). "Configure" opens the
// existing TeamCardSettings in a dialog so the underlying config is unchanged.
import { useState } from "react";
import { Settings } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import TeamCardSettings from "@/components/profile/TeamCardSettings";

export function TeamCardPanel({ name, email }: { name: string; email: string }) {
  const [open, setOpen] = useState(false);

  return (
    <Card className="p-6">
      <div className="flex items-start justify-between gap-3.5">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">TeamCard</h2>
          <div className="text-[13.5px] text-muted-foreground mt-0.5">
            Your collaboration identity — what collaborators see about you
          </div>
        </div>
        <Button variant="secondary" size="sm" className="gap-2" onClick={() => setOpen(true)}>
          <Settings className="w-3.5 h-3.5" />
          Configure
        </Button>
      </div>

      <div className="mt-4 bg-background border border-border rounded-xl px-[18px] py-4 flex items-center justify-between gap-3">
        <div>
          <div className="text-base font-semibold">{name || "—"}</div>
          <div className="text-[13px] text-muted-foreground mt-px">{email}</div>
        </div>
        <Badge variant="outline" className="border-primary/30 text-primary bg-primary/10">
          Verified
        </Badge>
      </div>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Configure TeamCard</DialogTitle>
          </DialogHeader>
          <TeamCardSettings />
        </DialogContent>
      </Dialog>
    </Card>
  );
}
