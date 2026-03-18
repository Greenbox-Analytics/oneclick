import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Check, Loader2, Unplug } from "lucide-react";
import type { IntegrationProvider, ConnectionStatus } from "@/types/integrations";

interface IntegrationCardProps {
  provider: IntegrationProvider;
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  status: ConnectionStatus;
  onConnect: () => void;
  onDisconnect: () => void;
  isConnecting?: boolean;
}

export function IntegrationCard({
  name,
  description,
  icon,
  color,
  status,
  onConnect,
  onDisconnect,
  isConnecting,
}: IntegrationCardProps) {
  const isConnected = status === "active";

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="flex flex-row items-center gap-4 pb-3">
        <div
          className="w-12 h-12 rounded-lg flex items-center justify-center shrink-0"
          style={{ backgroundColor: `${color}15` }}
        >
          <div style={{ color }}>{icon}</div>
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <CardTitle className="text-base">{name}</CardTitle>
            {isConnected && (
              <Badge variant="secondary" className="text-xs bg-green-100 text-green-700">
                <Check className="w-3 h-3 mr-1" />
                Connected
              </Badge>
            )}
            {status === "expired" && (
              <Badge variant="destructive" className="text-xs">
                Expired
              </Badge>
            )}
          </div>
          <p className="text-sm text-muted-foreground mt-1">{description}</p>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        {isConnected ? (
          <Button
            variant="outline"
            size="sm"
            onClick={onDisconnect}
            className="text-destructive hover:text-destructive"
          >
            <Unplug className="w-4 h-4 mr-2" />
            Disconnect
          </Button>
        ) : (
          <Button size="sm" onClick={onConnect} disabled={isConnecting}>
            {isConnecting ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Connecting...
              </>
            ) : (
              "Connect"
            )}
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
