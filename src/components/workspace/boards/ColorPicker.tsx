import { cn } from "@/lib/utils";

const PRESET_COLORS = [
  "#ef4444", // red
  "#f97316", // orange
  "#f59e0b", // amber
  "#22c55e", // green
  "#14b8a6", // teal
  "#3b82f6", // blue
  "#6366f1", // indigo
  "#8b5cf6", // violet
  "#ec4899", // pink
  "#6b7280", // gray
];

interface ColorPickerProps {
  value?: string;
  onChange: (color: string) => void;
}

export function ColorPicker({ value, onChange }: ColorPickerProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {/* Clear color option */}
      <button
        type="button"
        onClick={() => onChange("")}
        className={cn(
          "w-7 h-7 rounded-full border-2 border-dashed border-muted-foreground/30 transition-all hover:scale-110",
          !value && "ring-2 ring-primary ring-offset-2"
        )}
        title="No color"
      />
      {PRESET_COLORS.map((color) => (
        <button
          key={color}
          type="button"
          onClick={() => onChange(color)}
          className={cn(
            "w-7 h-7 rounded-full transition-all hover:scale-110",
            value === color && "ring-2 ring-primary ring-offset-2"
          )}
          style={{ backgroundColor: color }}
          title={color}
        />
      ))}
    </div>
  );
}
