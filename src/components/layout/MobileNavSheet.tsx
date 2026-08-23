import { useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { useBackToClose } from "@/hooks/useBackToClose";
import { cn } from "@/lib/utils";
import {
  Menu,
  Music,
  Folder,
  LayoutGrid,
  Wrench,
  Users,
  User,
  Settings,
  LogOut,
  BookOpen,
  Calculator,
  MessageSquare,
  PieChart,
  FileText,
  Receipt,
} from "lucide-react";

interface NavItem {
  label: string;
  path: string;
  icon: React.ComponentType<{ className?: string }>;
  disabled?: boolean;
}

const TOP_LEVEL: NavItem[] = [
  { label: "Dashboard", path: "/dashboard", icon: LayoutGrid },
  { label: "Portfolio", path: "/portfolio", icon: Folder },
  { label: "Workspace", path: "/workspace", icon: LayoutGrid },
  { label: "Artists", path: "/artists", icon: Users },
];

const TOOLS: NavItem[] = [
  { label: "Metadata Registry", path: "/tools/registry", icon: FileText },
  { label: "OneClick", path: "/tools/oneclick", icon: Calculator },
  { label: "Zoe", path: "/tools/zoe", icon: MessageSquare },
  { label: "Split Sheet", path: "/tools/split-sheet", icon: PieChart },
  { label: "Expense Tracker", path: "/tools/expense-tracker", icon: Receipt },
];

const FOOTER: NavItem[] = [
  { label: "Profile", path: "/profile", icon: User },
  { label: "Documentation", path: "/docs", icon: BookOpen },
];

export function MobileNavTrigger({ className }: { className?: string }) {
  return (
    <SheetTrigger asChild>
      <Button
        variant="ghost"
        size="icon"
        aria-label="Open navigation menu"
        className={cn("text-muted-foreground hover:text-foreground", className)}
      >
        <Menu className="w-5 h-5" />
      </Button>
    </SheetTrigger>
  );
}

interface MobileNavSheetProps {
  children?: React.ReactNode;
}

export function MobileNavSheet({ children }: MobileNavSheetProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { signOut } = useAuth();
  const [open, setOpen] = useState(false);

  // Hardware/browser Back closes the drawer instead of leaving the page.
  useBackToClose(open, () => setOpen(false));

  const isActive = (path: string) => {
    if (path === "/dashboard") return location.pathname === "/dashboard";
    return location.pathname === path || location.pathname.startsWith(path + "/");
  };

  const isToolsActive = TOOLS.some((t) => isActive(t.path));

  const handleSignOut = async () => {
    setOpen(false);
    await signOut();
    navigate("/auth");
  };

  const renderItem = (item: NavItem) => {
    const Icon = item.icon;
    const active = isActive(item.path);
    if (item.disabled) {
      return (
        <button
          key={item.path}
          disabled
          aria-disabled="true"
          className="w-full flex items-center gap-3 px-3 py-3 rounded-md text-left text-sm text-muted-foreground opacity-50 cursor-not-allowed"
        >
          <Icon className="w-4 h-4 shrink-0" />
          <span className="flex-1">{item.label}</span>
          <span className="text-[10px] uppercase tracking-wide">Coming soon</span>
        </button>
      );
    }
    // Real <Link>, not a button: these are the app's primary nav, so
    // cmd/middle-click and "open in new tab" have to work. `go` survives only
    // for the non-item entries below.
    return (
      <Link
        key={item.path}
        to={item.path}
        onClick={() => setOpen(false)}
        aria-current={active ? "page" : undefined}
        className={cn(
          "w-full flex items-center gap-3 px-3 py-3 rounded-md text-left text-sm transition-colors",
          active
            ? "bg-accent text-accent-foreground font-medium"
            : "text-foreground hover:bg-accent/50",
        )}
      >
        <Icon className="w-4 h-4 shrink-0" />
        <span>{item.label}</span>
      </Link>
    );
  };

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      {children ?? <MobileNavTrigger />}
      <SheetContent side="left" className="w-[85vw] max-w-sm p-0 flex flex-col">
        <SheetHeader className="px-4 py-4 border-b border-border text-left">
          <SheetTitle className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-md bg-primary flex items-center justify-center">
              <Music className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="text-lg font-bold">Msanii</span>
          </SheetTitle>
        </SheetHeader>

        <nav className="flex-1 overflow-y-auto px-2 py-3">
          <div className="space-y-1">{renderItem(TOP_LEVEL[0])}</div>

          <Accordion
            type="single"
            collapsible
            defaultValue={isToolsActive ? "tools" : undefined}
            className="mt-1"
          >
            <AccordionItem value="tools" className="border-none">
              <AccordionTrigger
                className={cn(
                  "px-3 py-3 rounded-md text-sm hover:no-underline hover:bg-accent/50",
                  isToolsActive && "text-accent-foreground font-medium",
                )}
              >
                <span className="flex items-center gap-3">
                  <Wrench className="w-4 h-4 shrink-0" />
                  Tools
                </span>
              </AccordionTrigger>
              <AccordionContent className="pb-1">
                <div className="space-y-1 pl-6">
                  {TOOLS.map(renderItem)}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          <div className="mt-1 space-y-1">{TOP_LEVEL.slice(1).map(renderItem)}</div>

          <div className="mt-4 pt-3 border-t border-border space-y-1">
            {FOOTER.map(renderItem)}
            <Link
              to="/workspace?tab=settings"
              onClick={() => setOpen(false)}
              className="w-full flex items-center gap-3 px-3 py-3 rounded-md text-left text-sm text-foreground hover:bg-accent/50 transition-colors"
            >
              <Settings className="w-4 h-4 shrink-0" />
              <span>Settings</span>
            </Link>
            <button
              onClick={handleSignOut}
              className="w-full flex items-center gap-3 px-3 py-3 rounded-md text-left text-sm text-destructive hover:bg-destructive/10 transition-colors"
            >
              <LogOut className="w-4 h-4 shrink-0" />
              <span>Sign out</span>
            </button>
          </div>
        </nav>
      </SheetContent>
    </Sheet>
  );
}
