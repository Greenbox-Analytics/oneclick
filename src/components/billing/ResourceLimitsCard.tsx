// src/components/billing/ResourceLimitsCard.tsx
import { Database } from "lucide-react";
import { Card } from "@/components/ui/card";
import { useEntitlements } from "@/hooks/useEntitlements";
import { useArtistsList } from "@/hooks/useArtistsList";
import { useProjectsList } from "@/hooks/useProjectsList";
import { useBoards } from "@/hooks/useBoards";

const fmtBytes = (bytes: number): string => {
  if (!bytes || bytes < 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
};

const capLabel = (cap: number | undefined): string =>
  cap === -1 ? "Unlimited" : cap == null ? "—" : cap.toLocaleString();

function Tile({ k, v, cap, icon }: { k: string; v: string; cap: string; icon?: React.ReactNode }) {
  return (
    <div className="bg-background border border-border/60 rounded-xl px-[15px] py-3.5">
      <div className="text-[12.5px] text-muted-foreground flex items-center gap-1.5">
        {icon}
        {k}
      </div>
      <div className="text-2xl font-bold tracking-tight mt-1.5 tabular-nums">{v}</div>
      <div className="text-[11.5px] text-muted-foreground/70 mt-0.5">{cap}</div>
    </div>
  );
}

export function ResourceLimitsCard() {
  const { data: ent } = useEntitlements();
  const { artists } = useArtistsList();
  const { projects } = useProjectsList();
  const { tasks } = useBoards();
  const caps = ent?.caps;

  const storageUsed = ent?.usage?.totalStorageBytes ?? 0;
  const storageCap =
    caps?.includedStorageBytes && caps.includedStorageBytes !== -1
      ? caps.includedStorageBytes
      : caps?.maxStorageBytes;

  const allUnlimited =
    caps?.maxArtists === -1 && caps?.maxProjects === -1 && caps?.maxTasks === -1;

  return (
    <Card className="p-6">
      <h2 className="text-lg font-semibold tracking-tight">Resource limits</h2>
      <div className="text-[13.5px] text-muted-foreground mt-0.5">
        {allUnlimited ? "Unlimited on your plan" : "Usage against your plan"}
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-[18px]">
        <Tile k="Artists" v={(artists?.length ?? 0).toLocaleString()} cap={capLabel(caps?.maxArtists)} />
        <Tile k="Projects" v={(projects?.length ?? 0).toLocaleString()} cap={capLabel(caps?.maxProjects)} />
        <Tile k="Tasks" v={(tasks?.length ?? 0).toLocaleString()} cap={capLabel(caps?.maxTasks)} />
        <Tile
          k="Storage"
          v={fmtBytes(storageUsed)}
          cap={storageCap === -1 ? "Unlimited" : `of ${fmtBytes(storageCap ?? 0)}`}
          icon={<Database className="w-[15px] h-[15px]" />}
        />
      </div>
    </Card>
  );
}
