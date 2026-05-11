import { useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Calculator, ArrowRight, ArrowLeft, Bot, FileText, Shield, BookOpen } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { trackToolUsage } from "@/pages/Dashboard";
import { PageHeader } from "@/components/layout/PageHeader";

// Hoisted to module scope (rendering-hoist-jsx, rerender-no-inline-components)
const TOOL_CARDS: { route: string; label: string; icon: LucideIcon; desc: string; comingSoon?: boolean }[] = [
  { route: "/tools/oneclick", label: "OneClick", icon: Calculator, desc: "Calculate royalty splits and manage contracts for your artists in one click." },
  { route: "/tools/zoe", label: "Zoe", icon: Bot, desc: "Ask questions about your contracts and get AI-powered answers with source citations." },
  { route: "/tools/split-sheet", label: "Split Sheet", icon: FileText, desc: "Generate professional split sheet agreements to document royalty ownership for your music." },
  { route: "/tools/registry", label: "Rights Registry", icon: Shield, desc: "Track master ownership, publishing splits, licensing rights, and generate proof-of-ownership documents.", comingSoon: true },
];

const Tools = () => {
  const navigate = useNavigate();

  const handleNavigate = useCallback((route: string, label: string) => {
    trackToolUsage(label, route);
    navigate(route);
  }, [navigate]);

  return (
    <div className="min-h-screen bg-background">
      <PageHeader
        actions={
          <>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate("/docs")}
              title="Documentation"
              className="text-muted-foreground hover:text-foreground"
            >
              <BookOpen className="w-4 h-4" />
            </Button>
            <Button variant="outline" className="hidden md:inline-flex" onClick={() => navigate("/dashboard")}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Dashboard
            </Button>
          </>
        }
      />

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-foreground mb-1">Tools</h2>
          <p className="text-muted-foreground mb-3">Select a tool to manage your music data</p>
          <div className="h-0.5 w-16 bg-gradient-to-r from-primary to-primary/0 rounded-full" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {TOOL_CARDS.map((tool) => (
            <Card
              key={tool.route}
              className={
                tool.comingSoon
                  ? "opacity-60 cursor-not-allowed border border-border"
                  : "hover:border-primary/40 transition-all cursor-pointer group border border-border"
              }
              onClick={tool.comingSoon ? undefined : () => handleNavigate(tool.route, tool.label)}
              aria-disabled={tool.comingSoon || undefined}
            >
              <CardHeader>
                <div className={
                  tool.comingSoon
                    ? "w-11 h-11 rounded-xl bg-primary/10 flex items-center justify-center mb-3"
                    : "w-11 h-11 rounded-xl bg-primary/10 flex items-center justify-center mb-3 group-hover:bg-primary/15 group-hover:scale-105 transition-all"
                }>
                  <tool.icon className="w-5 h-5 text-primary" />
                </div>
                <CardTitle className="flex items-center gap-2 text-lg">
                  {tool.label}
                  {!tool.comingSoon && (
                    <ArrowRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-all translate-x-[-8px] group-hover:translate-x-0 text-primary" />
                  )}
                </CardTitle>
                <CardDescription className="leading-relaxed">
                  {tool.desc}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {tool.comingSoon ? (
                  <span className="text-sm font-medium text-muted-foreground">
                    Coming Soon..
                  </span>
                ) : (
                  <span className="text-sm font-medium text-primary/80 group-hover:text-primary transition-colors">
                    Launch Tool →
                  </span>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      </main>
    </div>
  );
};

export default Tools;
