// src/components/billing/IntegrationsCard.tsx
import { Check, Lock } from "lucide-react";
import { Card } from "@/components/ui/card";
import { useIntegrations } from "@/hooks/useIntegrations";

export function IntegrationsCard() {
  const { connections } = useIntegrations();
  const isConn = (provider: string) =>
    connections.some((c) => c.provider === provider && (c.status ? c.status === "active" : true));

  // Google Drive + Slack are real integrations; Spotify + Apple Music are
  // placeholders (not yet available) shown dashed, matching the mockup.
  const chips: { label: string; connected: boolean; available: boolean }[] = [
    { label: "Google Drive", connected: isConn("google_drive"), available: true },
    { label: "Slack", connected: isConn("slack"), available: true },
    { label: "Spotify", connected: false, available: false },
    { label: "Apple Music", connected: false, available: false },
  ];

  return (
    <Card className="p-6">
      <h2 className="text-lg font-semibold tracking-tight">Integrations</h2>
      <div className="text-[13.5px] text-muted-foreground mt-0.5">
        Connected services · usage counted toward this period
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
        {chips.map((c) => (
          <div
            key={c.label}
            className={`flex items-center gap-2.5 px-3.5 py-2.5 border rounded-[10px] text-sm bg-background ${
              c.available && c.connected
                ? "border-border"
                : "border-border border-dashed opacity-55"
            }`}
          >
            {c.available && c.connected ? (
              <Check className="w-[18px] h-[18px] flex-none text-primary" strokeWidth={2.4} />
            ) : (
              <Lock className="w-[18px] h-[18px] flex-none text-muted-foreground" />
            )}
            {c.label}
          </div>
        ))}
      </div>
    </Card>
  );
}
