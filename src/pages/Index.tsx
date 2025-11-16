import { Button } from "@/components/ui/button";
import { Music } from "lucide-react";
import { useNavigate } from "react-router-dom";

const Index = () => {
  const navigate = useNavigate();

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-background via-secondary to-background">
      <div className="text-center max-w-2xl px-4">
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-primary mb-6">
          <Music className="w-10 h-10 text-primary-foreground" />
        </div>
        <h1 className="mb-4 text-5xl font-bold text-foreground">Msanii AI</h1>
        <p className="text-xl text-muted-foreground mb-8">
          Professional royalty management and artist profile tools for independent music managers
        </p>
        <div className="flex gap-4 justify-center">
          <Button size="lg" onClick={() => navigate("/auth")}>
            Get Started
          </Button>
          <Button size="lg" variant="outline" onClick={() => navigate("/auth")}>
            Sign In
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Index;
