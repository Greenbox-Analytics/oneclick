import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Music, FolderOpen, TrendingUp, Shield, FileText, Users } from "lucide-react";
import { useNavigate } from "react-router-dom";

const Index = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: Users,
      title: "Artist Profiles",
      description: "Comprehensive artist management with detailed profiles, contact info, and portfolio tracking.",
    },
    {
      icon: FolderOpen,
      title: "Project Organization",
      description: "Organize contracts, split sheets, and royalty statements in dedicated project folders.",
    },
    {
      icon: TrendingUp,
      title: "Royalty Tracking",
      description: "Monitor and manage royalty statements across multiple platforms and revenue streams.",
    },
    {
      icon: Shield,
      title: "Contract Management",
      description: "Securely store and access artist contracts with version control and expiration tracking.",
    },
    {
      icon: FileText,
      title: "Split Sheet Library",
      description: "Maintain organized split sheets for transparent revenue sharing and collaboration.",
    },
    {
      icon: Music,
      title: "DSP Integration",
      description: "Connect to Spotify, Apple Music, and SoundCloud for streamlined profile management.",
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Header Navigation */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            {/* Logo and Company Name */}
            <div 
              className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity" 
              onClick={() => navigate("/")}
            >
              <Music className="w-8 h-8" />
              <span className="text-xl font-bold text-foreground">Msanii</span>
            </div>
            
            {/* Navigation Buttons */}
            <div className="flex items-center gap-3">
              <Button onClick={() => navigate("/auth")} className="text-base">
                Sign In
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="container mx-auto px-4 py-16 md:py-24">
          <div className="max-w-3xl mx-auto text-center">
            <h1 className="text-5xl md:text-6xl font-bold text-foreground mb-6 tracking-tight">
              Artist Management
              <span className="block text-primary mt-2">Simplified</span>
            </h1>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Streamline your workflow with powerful tools built for music professionals.
            </p>
            <Button size="lg" onClick={() => navigate("/auth")} className="text-base px-8 h-12">
              Get Started
            </Button>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-16 md:py-24 bg-muted/30">
        <div className="container mx-auto px-4">
          <div className="max-w-2xl mx-auto text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-3">
              Everything You Need
            </h2>
            <p className="text-lg text-muted-foreground">
              Powerful features for independent music managers.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <Card key={index} className="p-6 text-center border-border/50 hover:border-primary/20 transition-colors bg-card/50">
                  <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-primary/5 mb-4">
                    <Icon className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="text-lg font-semibold text-foreground mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-muted-foreground text-sm leading-relaxed">
                    {feature.description}
                  </p>
                </Card>
              );
            })}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-sm text-muted-foreground">
              &copy; {new Date().getFullYear()} Greenbox Analytics Inc.
            </p>
            <div className="flex gap-6">
              <a href="/about" className="text-sm text-muted-foreground hover:text-foreground transition-colors">About</a>
              <a href="/contact" className="text-sm text-muted-foreground hover:text-foreground transition-colors">Contact</a>
              <a href="/privacy" className="text-sm text-muted-foreground hover:text-foreground transition-colors">Privacy</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
