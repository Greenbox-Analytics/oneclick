import { IntegrationCard } from "./IntegrationCard";
import { useIntegrations } from "@/hooks/useIntegrations";
import type { IntegrationProvider, ConnectionStatus } from "@/types/integrations";
import { useAnalytics } from "@/hooks/useAnalytics";

// Map backend provider key to the analytics tool id used in the registry.
const PROVIDER_TO_TOOL: Record<IntegrationProvider, "drive" | "dropbox"> = {
  google_drive: "drive",
  dropbox: "dropbox",
};

type IntegrationItem = {
  provider: IntegrationProvider;
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
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
    provider: "dropbox",
    name: "Dropbox",
    description: "Import files from Dropbox and save project files back with shareable links",
    icon: (
      <svg viewBox="0 0 24 24" className="w-6 h-6" fill="#0061FF" aria-label="Dropbox">
        <path d="M6 2 0 5.9l6 3.8 6-3.8L6 2zm12 0-6 3.9 6 3.8 6-3.8L18 2zM0 13.6l6 3.8 6-3.8-6-3.9-6 3.9zm18-3.9-6 3.9 6 3.8 6-3.8-6-3.9zM6.1 18.7l6 3.8 5.9-3.8-5.9-3.8-6 3.8z" />
      </svg>
    ),
    color: "#0061FF",
  },
];

export function IntegrationHub() {
  const { connections, connect, disconnect, isConnecting } = useIntegrations();
  const { captureIntegrationConnectStarted } = useAnalytics();

  const getStatus = (provider: IntegrationProvider): ConnectionStatus => {
    const conn = connections.find((c) => c.provider === provider);
    return conn?.status || "disconnected";
  };

  const handleConnect = (provider: IntegrationProvider) => {
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
            onConnect={() => handleConnect(integration.provider)}
            onDisconnect={() => disconnect(integration.provider)}
            isConnecting={isConnecting}
          />
        ))}
      </div>
    </div>
  );
}
