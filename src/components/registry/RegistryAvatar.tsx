import { cn } from "@/lib/utils";

function initials(name: string): string {
  if (!name) return "?";
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0])
    .join("")
    .toUpperCase();
}

function hashHue(seed: string): number {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  return h % 360;
}

interface RegistryAvatarProps {
  name: string;
  /** Optional explicit color; otherwise derived from the name. */
  color?: string | null;
  size?: number;
  className?: string;
  title?: string;
}

/**
 * Compact initials avatar used throughout the registry surface. Derives a
 * stable hue from the name when no color is supplied.
 */
export function RegistryAvatar({ name, color, size = 26, className, title }: RegistryAvatarProps) {
  const bg = color || `hsl(${hashHue(name)} 50% 40%)`;
  return (
    <span
      className={cn(
        "inline-flex items-center justify-center rounded-full font-semibold text-white select-none shrink-0",
        className
      )}
      style={{
        width: size,
        height: size,
        background: bg,
        fontSize: size * 0.4,
      }}
      title={title || name}
    >
      {initials(name)}
    </span>
  );
}
