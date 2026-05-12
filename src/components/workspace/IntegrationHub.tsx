import { useState } from "react";
import { toast } from "sonner";
import { IntegrationCard } from "./IntegrationCard";
import { useIntegrations } from "@/hooks/useIntegrations";
import { SlackPanel } from "./integrations/SlackPanel";
import type { IntegrationProvider, ConnectionStatus } from "@/types/integrations";
import { useIntegrationAllowed } from "@/hooks/useEntitlements";
import { PaywallModal } from "@/components/paywall/PaywallModal";

type IntegrationItem = {
  provider: IntegrationProvider | "atlassian";
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
  {
    provider: "notion",
    name: "Notion",
    description: "Sync project boards and tasks with Notion databases",
    icon: <img src="/Notion_app_logo.png" alt="Notion" className="w-6 h-6 object-contain" />,
    color: "#000000",
    comingSoon: true,
  },
  {
    provider: "atlassian",
    name: "Jira & Confluence",
    description: "Sync tasks with Jira and link Confluence pages to your projects",
    icon: <img src="/atlassian.png" alt="Atlassian" className="w-6 h-6 object-contain rounded" />,
    color: "#0052CC",
    comingSoon: true,
  },
];

export function IntegrationHub() {
  const { connections, connect, disconnect, isConnecting } = useIntegrations();
  const [slackPanelOpen, setSlackPanelOpen] = useState(false);
  const [paywallOpen, setPaywallOpen] = useState(false);
  const [paywallReason, setPaywallReason] = useState<string | undefined>(undefined);

  const { allowed: slackAllowed } = useIntegrationAllowed("slack");
  const { allowed: notionAllowed } = useIntegrationAllowed("notion");
  const { allowed: mondayAllowed } = useIntegrationAllowed("monday");

  const integrationAllowed: Record<string, boolean> = {
    google_drive: true, // Drive is always allowed; no paywall
    slack: slackAllowed,
    notion: notionAllowed,
    monday: mondayAllowed,
  };

  const integrationLabel: Record<string, string> = {
    slack: "Slack",
    notion: "Notion",
    monday: "Monday.com",
  };

  const getStatus = (provider: IntegrationItem["provider"]): ConnectionStatus => {
    if (provider === "atlassian") return "disconnected";
    const conn = connections.find((c) => c.provider === provider);
    return conn?.status || "disconnected";
  };

  const isConnected = (provider: IntegrationItem["provider"]) =>
    getStatus(provider) === "active";

  const handleConnect = (integration: IntegrationItem) => {
    if (integration.comingSoon) {
      toast.info(`${integration.name} integration is coming soon!`, {
        description: "We're working on it — stay tuned.",
      });
      return;
    }
    const provider = integration.provider;
    if (provider === "atlassian") return;
    if (!integrationAllowed[provider]) {
      const label = integrationLabel[provider] ?? provider;
      setPaywallReason(`${label} integration is a Pro feature. Upgrade to Pro to connect it.`);
      setPaywallOpen(true);
      return;
    }
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
            provider={integration.provider as IntegrationProvider}
            name={integration.name}
            description={integration.description}
            icon={integration.icon}
            color={integration.color}
            status={getStatus(integration.provider)}
            onConnect={() => handleConnect(integration)}
            onDisconnect={() => {
              if (integration.provider === "atlassian") return;
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

      {slackPanelOpen && (
        <SlackPanel onClose={() => setSlackPanelOpen(false)} />
      )}

      <PaywallModal
        open={paywallOpen}
        onClose={() => setPaywallOpen(false)}
        reason={paywallReason}
      />
    </div>
  );
}
