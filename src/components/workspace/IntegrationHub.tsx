import { useState } from "react";
import { toast } from "sonner";
import { IntegrationCard } from "./IntegrationCard";
import { useIntegrations } from "@/hooks/useIntegrations";
import { SlackPanel } from "./integrations/SlackPanel";
import type { IntegrationProvider, ConnectionStatus } from "@/types/integrations";
import { useIntegrationAllowed } from "@/hooks/useEntitlements";
import { PaywallModal } from "@/components/paywall/PaywallModal";
import { useAnalytics } from "@/hooks/useAnalytics";

// Map backend provider key to the analytics tool id used in the registry.
const PROVIDER_TO_TOOL: Record<IntegrationProvider, "drive" | "slack"> = {
  google_drive: "drive",
  slack: "slack",
};

type IntegrationItem = {
  provider: IntegrationProvider;
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  comingSoon?: boolean;
};

const INTEGRATIONS: IntegrationItem[] = [
  {
    provider: "google_drive",
    name: "Google Drive",
    description: "Import contracts and royalty statements from Drive into your projects",
    icon: <img src="/drive.webp" alt="Google Drive" className="w-6 h-6 object-contain" />,
    color: "#4285F4",
  },
  {
    provider: "slack",
    name: "Slack",
    description: "Get notifications and sync updates to Slack channels",
    icon: <img src="/slack.png" alt="Slack" className="w-6 h-6 object-contain" />,
    color: "#4A154B",
    comingSoon: true,
  },
];

export function IntegrationHub() {
  const { connections, connect, disconnect, isConnecting } = useIntegrations();
  const { captureIntegrationConnectStarted } = useAnalytics();
  const [slackPanelOpen, setSlackPanelOpen] = useState(false);
  const [paywallOpen, setPaywallOpen] = useState(false);
  const [paywallReason, setPaywallReason] = useState<string | undefined>(undefined);

  const { allowed: slackAllowed } = useIntegrationAllowed("slack");

  const integrationAllowed: Record<string, boolean> = {
    google_drive: true, // Drive is always allowed; no paywall
    slack: slackAllowed,
  };

  const integrationLabel: Record<string, string> = {
    slack: "Slack",
  };

  const getStatus = (provider: IntegrationProvider): ConnectionStatus => {
    const conn = connections.find((c) => c.provider === provider);
    return conn?.status || "disconnected";
  };

  const isConnected = (provider: IntegrationProvider) => getStatus(provider) === "active";

  const handleConnect = (integration: IntegrationItem) => {
    if (integration.comingSoon) {
      toast.info(`${integration.name} integration is coming soon!`, {
        description: "We're working on it — stay tuned.",
      });
      return;
    }
    const provider = integration.provider;
    if (!integrationAllowed[provider]) {
      const label = integrationLabel[provider] ?? provider;
      setPaywallReason(`${label} integration is a Pro feature. Upgrade to Pro to connect it.`);
      setPaywallOpen(true);
      return;
    }
    captureIntegrationConnectStarted(PROVIDER_TO_TOOL[provider]);
    connect(provider);
  };

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold">Connected Services</h3>
        <p className="text-sm text-muted-foreground">
          Connect your favorite tools to sync data and receive notifications
        </p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {INTEGRATIONS.map((integration) => (
          <IntegrationCard
            key={integration.provider}
            provider={integration.provider}
            name={integration.name}
            description={integration.description}
            icon={integration.icon}
            color={integration.color}
            status={getStatus(integration.provider)}
            onConnect={() => handleConnect(integration)}
            onDisconnect={() => {
              disconnect(integration.provider);
              if (integration.provider === "slack") setSlackPanelOpen(false);
            }}
            onConfigure={
              integration.provider === "slack" && isConnected("slack")
                ? () => setSlackPanelOpen(!slackPanelOpen)
                : undefined
            }
            isConnecting={isConnecting}
          />
        ))}
      </div>

      {slackPanelOpen && <SlackPanel onClose={() => setSlackPanelOpen(false)} />}

      <PaywallModal
        open={paywallOpen}
        onClose={() => setPaywallOpen(false)}
        reason={paywallReason}
      />
    </div>
  );
}
