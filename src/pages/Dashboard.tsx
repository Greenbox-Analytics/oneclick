import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Music, Calculator, User, Users, Plus, LogOut, LayoutGrid, Folder, Clock, Bot, BookOpen } from "lucide-react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { useAuth } from "@/contexts/AuthContext";
import { useEffect, useState, useMemo, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useWorkspaceSettings } from "@/hooks/useWorkspaceSettings";
import { useOnboardingStatus } from "@/hooks/useOnboardingStatus";
import { useWalkthrough } from "@/hooks/useWalkthrough";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";

// Tool registry for Recently Used
const TOOL_REGISTRY: Record<string, { icon: typeof Calculator; label: string }> = {
  "/tools": { icon: Calculator, label: "Tools" },
  "/artists": { icon: Users, label: "Artist Profiles" },
  "/workspace": { icon: LayoutGrid, label: "Workspace" },
  "/portfolio": { icon: Folder, label: "Portfolio" },
  "/artists/new": { icon: Plus, label: "Add Artist" },
  "/tools/oneclick": { icon: Calculator, label: "OneClick" },
  "/tools/zoe": { icon: Bot, label: "Zoe" },
  "/profile": { icon: User, label: "Profile" },
  "/docs": { icon: BookOpen, label: "Documentation" },
};

interface RecentTool {
  route: string;
  name: string;
  timestamp: number;
}

function getRecentTools(): RecentTool[] {
  try {
    const raw = localStorage.getItem("msanii_recent_tools");
    if (!raw) return [];
    return JSON.parse(raw) as RecentTool[];
  } catch {
    return [];
  }
}

export function trackToolUsage(name: string, route: string) {
  const tools = getRecentTools().filter((t) => t.route !== route);
  tools.unshift({ route, name, timestamp: Date.now() });
  localStorage.setItem("msanii_recent_tools", JSON.stringify(tools.slice(0, 10)));
}

const Dashboard = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [profile, setProfile] = useState<{
    full_name: string | null;
    avatar_url: string | null;
    given_name: string | null;
    first_name: string | null;
  } | null>(null);
  const [now, setNow] = useState(new Date());
  const [recentTools, setRecentTools] = useState<RecentTool[]>(getRecentTools());
  const location = useLocation();
  const { settings } = useWorkspaceSettings();
  const { walkthroughCompleted, loading: onboardingLoading } = useOnboardingStatus();
  const walkthrough = useWalkthrough();

  // Optimistic: if we just came from onboarding, start walkthrough immediately
  // without waiting for the Supabase query to resolve.
  const fromOnboarding = (location.state as any)?.fromOnboarding === true;

  // Auto-start walkthrough for first-time users
  useEffect(() => {
    if (onboardingLoading && !fromOnboarding) return;
    const shouldStart = fromOnboarding || walkthroughCompleted === false;
    if (shouldStart && !walkthrough.isActive) {
      // Small delay to ensure DOM elements are rendered
      const timer = setTimeout(() => walkthrough.start(), 300);
      return () => clearTimeout(timer);
    }
  }, [fromOnboarding, walkthroughCompleted, onboardingLoading]);

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formattedDateTime = useMemo(() => {
    const tz = settings?.timezone || Intl.DateTimeFormat().resolvedOptions().timeZone;
    const use24h = settings?.use_24h_time ?? false;
    const dateStr = now.toLocaleDateString("en-US", {
      timeZone: tz,
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
    const timeStr = now.toLocaleTimeString("en-US", {
      timeZone: tz,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: !use24h,
    });
    const tzAbbr = now.toLocaleTimeString("en-US", {
      timeZone: tz,
      timeZoneName: "short",
    }).split(" ").pop() || "";
    return `${dateStr} · ${timeStr} ${tzAbbr}`;
  }, [now, settings?.timezone, settings?.use_24h_time]);

  useEffect(() => {
    const fetchProfile = async () => {
      if (!user) return;

      const { data, error } = await supabase
        .from("profiles")
        .select("*")
        .eq("id", user.id)
        .single();

      if (error) {
        console.error("Dashboard profile fetch error:", error);
      }
      if (data) {
        console.log("Dashboard profile data:", { given_name: data.given_name, first_name: data.first_name, full_name: data.full_name });
        setProfile(data);
      }
    };

    fetchProfile();
  }, [user]);

  const getInitials = () => {
    if (profile?.full_name) {
      return profile.full_name
        .trim()
        .split(/\s+/)
        .map((n) => n[0])
        .join("")
        .toUpperCase()
        .slice(0, 2);
    }
    return user?.email?.substring(0, 2).toUpperCase() || "U";
  };

  const greeting = useMemo(() => {
    // Priority: preferred name > first name only > first word of full name > email prefix
    const givenName = profile?.given_name?.trim();
    const firstName = profile?.first_name?.trim();
    const fullNameFirst = profile?.full_name?.trim().split(/[\s.]+/)[0];
    const emailPrefix = user?.email?.split("@")[0];
    const name = givenName || firstName || fullNameFirst || emailPrefix || "there";
    return `Welcome back, ${name}!`;
  }, [profile, user]);

  const handleNavigate = useCallback((route: string, label: string) => {
    trackToolUsage(label, route);
    setRecentTools(getRecentTools());
    navigate(route);
  }, [navigate]);

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div
            className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => navigate("/dashboard")}
          >
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate("/docs")}
              title="Documentation"
              className="text-muted-foreground hover:text-foreground"
            >
              <BookOpen className="w-4 h-4" />
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="relative h-10 w-10 rounded-full bg-primary hover:bg-primary/90">
                  <Avatar className="h-10 w-10">
                    <AvatarImage src={profile?.avatar_url || ""} alt={profile?.full_name || ""} />
                    <AvatarFallback className="bg-primary text-primary-foreground">{getInitials()}</AvatarFallback>
                  </Avatar>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56" align="end" forceMount>
                <DropdownMenuLabel className="font-normal">
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-medium leading-none">{profile?.full_name || "User"}</p>
                    <p className="text-xs leading-none text-muted-foreground">
                      {user?.email}
                    </p>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => navigate("/profile")}>
                  <User className="mr-2 h-4 w-4" />
                  <span>Profile</span>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => navigate("/")}>
                  <LogOut className="mr-2 h-4 w-4" />
                  <span>Log out</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-foreground mb-2">Dashboard</h2>
          <p className="text-lg text-foreground/80">{greeting}</p>
          <p className="text-sm text-muted-foreground">{formattedDateTime}</p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4 mb-8">
          <Card data-walkthrough="tools" className="flex flex-col border-primary/40 hover:shadow-lg transition-shadow cursor-pointer" onClick={() => handleNavigate("/tools", "Tools")}>
            <CardHeader>
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-3">
                <Calculator className="w-6 h-6 text-primary" />
              </div>
              <CardTitle>Tools</CardTitle>
              <CardDescription>Access OneClick and other management tools</CardDescription>
            </CardHeader>
            <CardContent className="mt-auto">
              <Button variant="outline" className="w-full">
                Open Tools
              </Button>
            </CardContent>
          </Card>

          <Card data-walkthrough="artists" className="flex flex-col border-primary/40 hover:shadow-lg transition-shadow cursor-pointer" onClick={() => handleNavigate("/artists", "Artist Profiles")}>
            <CardHeader>
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-3">
                <Users className="w-6 h-6 text-primary" />
              </div>
              <CardTitle>Artist Profiles</CardTitle>
              <CardDescription>View and manage your artist roster</CardDescription>
            </CardHeader>
            <CardContent className="mt-auto">
              <Button variant="outline" className="w-full">
                View Artists
              </Button>
            </CardContent>
          </Card>

          <Card data-walkthrough="workspace" className="flex flex-col border-primary/40 hover:shadow-lg transition-shadow cursor-pointer" onClick={() => handleNavigate("/workspace", "Workspace")}>
            <CardHeader>
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-3">
                <LayoutGrid className="w-6 h-6 text-primary" />
              </div>
              <CardTitle>Workspace</CardTitle>
              <CardDescription>Project boards, integrations, and connected services</CardDescription>
            </CardHeader>
            <CardContent className="mt-auto">
              <Button variant="outline" className="w-full">
                Open Workspace
              </Button>
            </CardContent>
          </Card>

          <Card data-walkthrough="portfolio" className="flex flex-col border-primary/40 hover:shadow-lg transition-shadow cursor-pointer" onClick={() => handleNavigate("/portfolio", "Portfolio")}>
            <CardHeader>
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-3">
                <Folder className="w-6 h-6 text-primary" />
              </div>
              <CardTitle>Portfolio</CardTitle>
              <CardDescription>Your profile organized by year, artist, and project</CardDescription>
            </CardHeader>
            <CardContent className="mt-auto">
              <Button variant="outline" className="w-full">
                View Portfolio
              </Button>
            </CardContent>
          </Card>
        </div>

        <Card className="border-primary/40">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Clock className="w-5 h-5 text-primary" />
              <CardTitle>Recently Used</CardTitle>
            </div>
            <CardDescription>Quick access to your recent tools</CardDescription>
          </CardHeader>
          <CardContent>
            {recentTools.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">No recent activity yet</p>
            ) : (
              <div className="flex flex-wrap gap-3">
                {recentTools.map((tool) => {
                  const registry = TOOL_REGISTRY[tool.route];
                  const IconComponent = registry?.icon || Calculator;
                  return (
                    <Button
                      key={tool.route}
                      variant="outline"
                      className="gap-2 border-primary/40 hover:border-primary hover:bg-primary/5 hover:text-foreground transition-colors"
                      onClick={() => handleNavigate(tool.route, tool.name)}
                    >
                      <IconComponent className="w-4 h-4 text-primary" />
                      {registry?.label || tool.name}
                    </Button>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </main>
      <WalkthroughProvider
        isActive={walkthrough.isActive}
        currentStep={walkthrough.currentStep}
        currentStepIndex={walkthrough.currentStepIndex}
        totalSteps={walkthrough.totalSteps}
        onNext={walkthrough.next}
        onSkip={walkthrough.skip}
      />
    </div>
  );
};

export default Dashboard;
