import { BookOpen } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";

/**
 * The header's documentation icon. One shared component so every page gets the
 * same button instead of inlining its own.
 */
export function HeaderDocsButton() {
  const navigate = useNavigate();
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => navigate("/docs")}
      aria-label="Documentation"
      title="Documentation"
      className="text-muted-foreground hover:text-foreground"
    >
      <BookOpen className="w-4 h-4" />
    </Button>
  );
}
