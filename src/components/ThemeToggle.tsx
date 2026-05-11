import { useEffect, useState } from "react";
import { useTheme } from "next-themes";
import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const isDark = mounted && resolvedTheme === "dark";

  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label="Toggle theme"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className="fixed bottom-4 right-4 z-50 rounded-full border border-border bg-card text-muted-foreground shadow-md hover:text-foreground hover:bg-card"
    >
      {mounted ? (
        isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />
      ) : (
        <Sun className="w-5 h-5 opacity-0" />
      )}
    </Button>
  );
}
