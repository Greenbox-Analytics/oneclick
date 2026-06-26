import { cn } from "@/lib/utils";

interface ReleaseTagProps {
  released: boolean;
  size?: "default" | "lg";
}

/** Pastel "Released" / "Unreleased" pill. */
export function ReleaseTag({ released, size = "default" }: ReleaseTagProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full font-medium border",
        size === "lg" ? "text-xs px-2.5 py-1" : "text-[11px] px-2 py-0.5",
        released
          ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
          : "bg-amber-500/10 text-amber-400 border-amber-500/20"
      )}
    >
      <span
        className={cn(
          "w-1.5 h-1.5 rounded-full",
          released ? "bg-emerald-400" : "bg-amber-400"
        )}
      />
      {released ? "Released" : "Unreleased"}
    </span>
  );
}
