import { ReactNode } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, Home, Music } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { useIsMobile } from "@/hooks/use-mobile";
import { useSmartBack } from "@/hooks/useSmartBack";
import { MobileNavSheet } from "@/components/layout/MobileNavSheet";
import { NotificationBell } from "@/components/layout/NotificationBell";
import { HeaderDocsButton } from "@/components/layout/HeaderDocsButton";
import { HeaderCreditsTicker } from "@/components/billing/HeaderCreditsTicker";
import { HeaderContextSwitcher } from "@/components/billing/HeaderContextSwitcher";
import { cn } from "@/lib/utils";

interface PageHeaderProps {
  title?: string;
  subtitle?: string;
  backTo?: string | (() => void);
  /** Genuinely page-specific actions (search, tool help, etc.) — rendered
   * first. Docs/home/credits are built in; don't pass them here. */
  actions?: ReactNode;
  /** The user/profile dropdown — rendered LAST. Use this slot (instead of
   * stuffing the profile menu into `actions`) so the order stays consistent:
   * actions → context → credits → home → docs → notifications → profile. */
  userMenu?: ReactNode;
  showLogo?: boolean;
  showBack?: boolean;
  className?: string;
}

export function PageHeader({
  title,
  subtitle,
  backTo,
  actions,
  userMenu,
  showLogo = true,
  showBack = true,
  className,
}: PageHeaderProps) {
  const isMobile = useIsMobile();
  const { user } = useAuth();

  // Signed-in globals, in order: context → credits → home → docs → bell.
  // (The context switcher renders nothing unless the user has more than one
  // context.)
  //
  // "Home" means ONE thing for a signed-in user: the dashboard. It used to
  // point at the marketing landing page while the logo two inches away went
  // to /dashboard — two home affordances, two destinations. Now they agree,
  // and the button only renders where the logo isn't already doing the job,
  // so a header never shows both.
  const renderGlobalActions = (showHome: boolean) =>
    user ? (
      <>
        <HeaderContextSwitcher />
        <HeaderCreditsTicker />
        {showHome && (
          <Button
            asChild
            variant="ghost"
            size="icon"
            aria-label="Dashboard"
            title="Dashboard"
            className="text-muted-foreground hover:text-foreground"
          >
            <Link to="/dashboard">
              <Home className="w-4 h-4" />
            </Link>
          </Button>
        )}
        <HeaderDocsButton />
        <NotificationBell />
      </>
    ) : null;

  // A string `backTo` is treated as a *fallback* route, not a forced
  // destination: Back returns the user to the page they actually came from,
  // and only lands on `backTo` when there's no in-app history to go back to.
  const smartBack = useSmartBack(typeof backTo === "string" ? backTo : "/dashboard");
  const handleBack = () => {
    if (typeof backTo === "function") backTo();
    else smartBack();
  };

  if (isMobile) {
    const logoShown = !title && showLogo;
    return (
      <header className={cn("border-b border-border bg-card sticky top-0 z-40", className)}>
        <div className="px-3 py-2 flex items-center gap-2">
          <MobileNavSheet />
          {/* Mobile got no Back button at all, so phone users were left with
              only the browser/hardware back — the one path that hit every
              history bug. Icon-only to keep the cramped header readable. */}
          {showBack && (
            <Button
              variant="ghost"
              size="icon"
              aria-label="Back"
              className="shrink-0 text-muted-foreground hover:text-foreground"
              onClick={handleBack}
            >
              <ArrowLeft className="w-5 h-5" />
            </Button>
          )}
          <div className="flex-1 min-w-0">
            {title ? (
              <>
                <h1 className="text-base font-semibold truncate">{title}</h1>
                {subtitle && (
                  <p className="text-xs text-muted-foreground truncate">{subtitle}</p>
                )}
              </>
            ) : logoShown ? (
              <Link to="/dashboard" className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-md bg-primary flex items-center justify-center">
                  <Music className="w-4 h-4 text-primary-foreground" />
                </div>
                <span className="text-base font-bold">Msanii</span>
              </Link>
            ) : null}
          </div>
          {(actions || user || userMenu) && (
            <div className="flex items-center gap-1 shrink-0">
              {actions}
              {/* The nav sheet already has a Dashboard item, so on mobile the
                  Home icon is always redundant. */}
              {renderGlobalActions(false)}
              {userMenu}
            </div>
          )}
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
            <Link
              to="/dashboard"
              className="flex items-center gap-3 hover:opacity-80 transition-opacity"
            >
              <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
                <Music className="w-6 h-6 text-primary-foreground" />
              </div>
              <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
            </Link>
          )}
          {title && !showLogo && (
            <div className="min-w-0">
              <h1 className="text-2xl font-bold text-foreground truncate">{title}</h1>
              {subtitle && <p className="text-sm text-muted-foreground truncate">{subtitle}</p>}
            </div>
          )}
        </div>
        {(actions || user || userMenu) && (
          <div className="flex items-center gap-2 shrink-0">
            {actions}
            {renderGlobalActions(!showLogo)}
            {userMenu}
          </div>
        )}
      </div>
    </header>
  );
}
