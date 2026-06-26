import { Disc, Music } from "lucide-react";
import { cn } from "@/lib/utils";

interface ArtworkProps {
  seed: string;
  /** Pixel size for a square swatch. Defaults to 44. */
  size?: number;
  /** True when a real DSP-pulled artwork is available — paints a gradient. */
  hasArtwork?: boolean;
  className?: string;
}

function hueFromString(seed: string): number {
  if (!seed) return 0;
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  return (h * 7) % 360;
}

/**
 * Small square swatch used in dashboard rows and the wizard result list.
 * Falls back to a muted placeholder when no DSP artwork is available.
 */
export function Artwork({ seed, size = 44, hasArtwork, className }: ArtworkProps) {
  const hue = hueFromString(seed);
  const style = hasArtwork
    ? {
        width: size,
        height: size,
        background: `linear-gradient(135deg, hsl(${hue} 45% 55%), hsl(${(hue + 40) % 360} 45% 42%))`,
      }
    : { width: size, height: size };

  return (
    <div
      className={cn(
        "rounded-lg flex items-center justify-center shrink-0",
        hasArtwork
          ? "text-white/90"
          : "bg-muted text-muted-foreground border border-border",
        className
      )}
      style={style}
      title={hasArtwork ? "Artwork pulled from DSP" : undefined}
    >
      {hasArtwork ? (
        <Disc style={{ width: size * 0.45, height: size * 0.45 }} />
      ) : (
        <Music style={{ width: size * 0.4, height: size * 0.4 }} />
      )}
    </div>
  );
}
