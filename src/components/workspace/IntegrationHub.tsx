import { IntegrationCard } from "./IntegrationCard";
import { useIntegrations } from "@/hooks/useIntegrations";
import type { IntegrationProvider, ConnectionStatus } from "@/types/integrations";

const INTEGRATIONS = [
  {
    provider: "google_drive" as IntegrationProvider,
    name: "Google Drive",
    description: "Sync contracts and royalty statements with Google Drive",
    icon: <img src="/drive.webp" alt="Google Drive" className="w-6 h-6 object-contain" />,
    color: "#4285F4",
  },
  {
    provider: "slack" as IntegrationProvider,
    name: "Slack",
    description: "Get notifications and sync updates to Slack channels",
    icon: <img src="/slack.png" alt="Slack" className="w-6 h-6 object-contain" />,
    color: "#4A154B",
  },
  {
    provider: "notion" as IntegrationProvider,
    name: "Notion",
    description: "Sync project boards and tasks with Notion databases",
    icon: <img src="/Notion_app_logo.png" alt="Notion" className="w-6 h-6 object-contain" />,
    color: "#000000",
  },
  {
    provider: "monday" as IntegrationProvider,
    name: "Monday.com",
    description: "Sync project boards and tasks with Monday.com boards",
    icon: <img src="/mondaycom.png" alt="Monday.com" className="w-6 h-6 object-contain" />,
    color: "#FF3D57",
  },
];

export function IntegrationHub() {
  const { connections, connect, disconnect, isConnecting } = useIntegrations();

  const getStatus = (provider: IntegrationProvider): ConnectionStatus => {
    const conn = connections.find((c) => c.provider === provider);
    return conn?.status || "disconnected";
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
            {...integration}
            status={getStatus(integration.provider)}
            onConnect={() => connect(integration.provider)}
            onDisconnect={() => disconnect(integration.provider)}
            isConnecting={isConnecting}
          />
        ))}
      </div>
    </div>
  );
}
