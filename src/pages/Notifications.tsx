import { PageHeader } from "@/components/layout/PageHeader";
import { RegistryNotifications } from "@/components/workspace/RegistryNotifications";

export default function Notifications() {
  return (
    <div className="min-h-screen bg-background">
      {/* showLogo={false} is REQUIRED — desktop PageHeader only renders `title` when
          !showLogo (PageHeader.tsx:126); with the default showLogo=true it shows the
          Msanii logo instead and "Notifications" never appears on desktop. */}
      <PageHeader title="Notifications" showLogo={false} backTo="/dashboard" />
      <div className="container mx-auto px-4 py-6 max-w-3xl">
        <RegistryNotifications />
      </div>
    </div>
  );
}
