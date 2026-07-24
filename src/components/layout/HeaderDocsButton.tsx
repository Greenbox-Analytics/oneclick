import { BookOpen } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { AutoHideTooltip } from "@/components/layout/AutoHideTooltip";

/**
 * The header's documentation icon. One shared component so every page gets
 * the same auto-fading hover tooltip as the home and notification icons —
 * previously each page inlined this button with a browser-native `title`
 * tooltip, which looked different from the rest of the header set.
 */
export function HeaderDocsButton() {
  const navigate = useNavigate();
  return (
    <AutoHideTooltip label="Documentation">
      <Button
        variant="ghost"
        size="icon"
        onClick={() => navigate("/docs")}
        aria-label="Documentation"
        className="text-muted-foreground hover:text-foreground"
      >
        <BookOpen className="w-4 h-4" />
      </Button>
    </AutoHideTooltip>
  );
}
