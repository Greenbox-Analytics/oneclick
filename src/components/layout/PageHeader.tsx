import { ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Music } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useIsMobile } from "@/hooks/use-mobile";
import { MobileNavSheet } from "@/components/layout/MobileNavSheet";
import { cn } from "@/lib/utils";

interface PageHeaderProps {
  title?: string;
  subtitle?: string;
  backTo?: string | (() => void);
  actions?: ReactNode;
  showLogo?: boolean;
  showBack?: boolean;
  className?: string;
}

export function PageHeader({
  title,
  subtitle,
  backTo,
  actions,
  showLogo = true,
  showBack = true,
  className,
}: PageHeaderProps) {
  const navigate = useNavigate();
  const isMobile = useIsMobile();

  const handleBack = () => {
    if (typeof backTo === "function") backTo();
    else if (typeof backTo === "string") navigate(backTo);
    else navigate(-1);
  };

  if (isMobile) {
    return (
      <header className={cn("border-b border-border bg-card sticky top-0 z-40", className)}>
        <div className="px-3 py-2 flex items-center gap-2">
          <MobileNavSheet />
          <div className="flex-1 min-w-0">
            {title ? (
              <>
                <h1 className="text-base font-semibold truncate">{title}</h1>
                {subtitle && (
                  <p className="text-xs text-muted-foreground truncate">{subtitle}</p>
                )}
              </>
            ) : showLogo ? (
              <div
                className="flex items-center gap-2 cursor-pointer"
                onClick={() => navigate("/dashboard")}
              >
                <div className="w-7 h-7 rounded-md bg-primary flex items-center justify-center">
                  <Music className="w-4 h-4 text-primary-foreground" />
                </div>
                <span className="text-base font-bold">Msanii</span>
              </div>
            ) : null}
          </div>
          {actions && <div className="flex items-center gap-1 shrink-0">{actions}</div>}
        </div>
      </header>
    );
  }

  return (
    <header className={cn("border-b border-border bg-card", className)}>
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3 min-w-0">
          {showBack && (
            <>
              <Button
                variant="ghost"
                size="sm"
                className="text-muted-foreground hover:text-foreground"
                onClick={handleBack}
              >
                <ArrowLeft className="w-4 h-4 mr-1" /> Back
              </Button>
              <div className="w-px h-6 bg-border" />
            </>
          )}
          {showLogo && (
            <div
              className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
              onClick={() => navigate("/dashboard")}
            >
              <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
                <Music className="w-6 h-6 text-primary-foreground" />
              </div>
              <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
            </div>
          )}
          {title && !showLogo && (
            <div className="min-w-0">
              <h1 className="text-2xl font-bold text-foreground truncate">{title}</h1>
              {subtitle && <p className="text-sm text-muted-foreground truncate">{subtitle}</p>}
            </div>
          )}
        </div>
        {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
      </div>
    </header>
  );
}
