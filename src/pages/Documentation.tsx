import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Music,
  ArrowLeft,
  BookOpen,
  Calculator,
  Bot,
  FileText,
  Users,
  LayoutGrid,
  Folder,
  Phone,
  Lightbulb,
  Rocket,
} from "lucide-react";
import { useNavigate } from "react-router-dom";

interface Section {
  id: string;
  label: string;
  icon: React.ElementType;
}

const SECTIONS: Section[] = [
  { id: "getting-started", label: "Getting Started", icon: Rocket },
  { id: "oneclick", label: "OneClick", icon: Calculator },
  { id: "zoe", label: "Zoe AI", icon: Bot },
  { id: "split-sheet", label: "Split Sheet", icon: FileText },
  { id: "artist-management", label: "Artist Management", icon: Users },
  { id: "workspace", label: "Workspace", icon: LayoutGrid },
  { id: "portfolio", label: "Portfolio", icon: Folder },
  { id: "contacts-payments", label: "Contacts & Payments", icon: Phone },
  { id: "best-practices", label: "Best Practices", icon: Lightbulb },
];

const Documentation = () => {
  const navigate = useNavigate();
  const [activeSection, setActiveSection] = useState("getting-started");

  useEffect(() => {
    const handleScroll = () => {
      const sectionElements = SECTIONS.map((s) => ({
        id: s.id,
        el: document.getElementById(s.id),
      }));
      for (const { id, el } of sectionElements) {
        if (el) {
          const rect = el.getBoundingClientRect();
          if (rect.top <= 120 && rect.bottom > 120) {
            setActiveSection(id);
            break;
          }
        }
      }
    };
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
      setActiveSection(id);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div
            className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => navigate("/")}
          >
            <Music className="w-8 h-8" />
            <span className="text-xl font-bold text-foreground">Msanii</span>
          </div>
          <div className="flex items-center gap-3">
            <Button variant="outline" onClick={() => navigate("/dashboard")}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Dashboard
            </Button>
            <Button onClick={() => navigate("/auth")}>Sign In</Button>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <BookOpen className="w-8 h-8 text-primary" />
            <h1 className="text-3xl font-bold text-foreground">Documentation</h1>
          </div>
          <p className="text-muted-foreground text-lg">
            Learn how to use Msanii to manage your music business effectively.
          </p>
        </div>

        <div className="flex gap-8">
          {/* Sidebar Navigation - hidden on mobile */}
          <nav className="hidden lg:block w-64 shrink-0">
            <div className="sticky top-24 space-y-1">
              {SECTIONS.map((section) => {
                const Icon = section.icon;
                return (
                  <button
                    key={section.id}
                    onClick={() => scrollToSection(section.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors text-left ${
                      activeSection === section.id
                        ? "bg-primary/10 text-primary font-medium"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted"
                    }`}
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    {section.label}
                  </button>
                );
              })}
            </div>
          </nav>

          {/* Mobile Navigation - visible on small screens */}
          <div className="lg:hidden w-full mb-6 overflow-x-auto">
            <div className="flex gap-2 pb-2">
              {SECTIONS.map((section) => {
                const Icon = section.icon;
                return (
                  <button
                    key={section.id}
                    onClick={() => scrollToSection(section.id)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm whitespace-nowrap transition-colors ${
                      activeSection === section.id
                        ? "bg-primary/10 text-primary font-medium"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted"
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {section.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Content Area */}
          <main className="flex-1 min-w-0 space-y-12">
            <section id="getting-started">
              <h2 className="text-2xl font-bold text-foreground mb-4">Getting Started</h2>
              <p className="text-muted-foreground">Content coming soon.</p>
            </section>

            <Separator />

            <section id="oneclick">
              <h2 className="text-2xl font-bold text-foreground mb-4">OneClick</h2>
              <p className="text-muted-foreground">Content coming soon.</p>
            </section>

            <Separator />

            <section id="zoe">
              <h2 className="text-2xl font-bold text-foreground mb-4">Zoe AI</h2>
              <p className="text-muted-foreground">Content coming soon.</p>
            </section>

            <Separator />

            <section id="split-sheet">
              <h2 className="text-2xl font-bold text-foreground mb-4">Split Sheet</h2>
              <p className="text-muted-foreground">Content coming soon.</p>
            </section>

            <Separator />

            <section id="artist-management">
              <h2 className="text-2xl font-bold text-foreground mb-4">Artist Management</h2>
              <p className="text-muted-foreground">Content coming soon.</p>
            </section>

            <Separator />

            <section id="workspace">
              <h2 className="text-2xl font-bold text-foreground mb-4">Workspace</h2>
              <p className="text-muted-foreground">Content coming soon.</p>
            </section>

            <Separator />

            <section id="portfolio">
              <h2 className="text-2xl font-bold text-foreground mb-4">Portfolio</h2>
              <p className="text-muted-foreground">Content coming soon.</p>
            </section>

            <Separator />

            <section id="contacts-payments">
              <h2 className="text-2xl font-bold text-foreground mb-4">Contacts & Payments</h2>
              <p className="text-muted-foreground">Content coming soon.</p>
            </section>

            <Separator />

            <section id="best-practices">
              <h2 className="text-2xl font-bold text-foreground mb-4">Best Practices</h2>
              <p className="text-muted-foreground">Content coming soon.</p>
            </section>
          </main>
        </div>
      </div>
    </div>
  );
};

export default Documentation;
