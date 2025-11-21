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
      {/* Hero Section */}
      <section className="relative overflow-hidden border-b border-border">
        <div className="container mx-auto px-4 py-24 md:py-32">
          <div className="max-w-4xl mx-auto text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-xl bg-primary/10 border border-primary/20 mb-8">
              <Music className="w-8 h-8 text-primary" />
            </div>
            <h1 className="text-4xl md:text-6xl font-bold text-foreground mb-6 tracking-tight">
              Professional Artist Management
              <span className="block text-primary mt-2">Built for Music Managers</span>
            </h1>
            <p className="text-lg md:text-xl text-muted-foreground mb-10 max-w-2xl mx-auto leading-relaxed">
              Streamline your workflow with enterprise-grade tools for royalty tracking, contract management, 
              and artist portfolio organization. Built by industry professionals, for industry professionals.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" onClick={() => navigate("/auth")} className="text-base px-8">
                Get Started Free
              </Button>
              <Button size="lg" variant="outline" onClick={() => navigate("/auth")} className="text-base px-8">
                Sign In
              </Button>
            </div>
          </div>
        </div>
        
        {/* Subtle background pattern */}
        <div className="absolute inset-0 -z-10 bg-[linear-gradient(to_right,hsl(var(--border))_1px,transparent_1px),linear-gradient(to_bottom,hsl(var(--border))_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-20"></div>
      </section>

      {/* Features Grid */}
      <section className="py-24 md:py-32">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Everything You Need to Manage Artists
            </h2>
            <p className="text-lg text-muted-foreground">
              Powerful features designed specifically for independent music managers and labels.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <Card 
                  key={index} 
                  className="p-6 hover:shadow-lg transition-shadow duration-200 border-border bg-card"
                >
                  <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-primary/10 mb-4">
                    <Icon className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold text-card-foreground mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-muted-foreground leading-relaxed">
                    {feature.description}
                  </p>
                </Card>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 border-t border-border">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Ready to Elevate Your Artist Management?
            </h2>
            <p className="text-lg text-muted-foreground mb-8">
              Join professional music managers who trust Msanii AI to streamline their workflow.
            </p>
            <Button size="lg" onClick={() => navigate("/auth")} className="text-base px-10">
              Start Managing Artists Today
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Index;
