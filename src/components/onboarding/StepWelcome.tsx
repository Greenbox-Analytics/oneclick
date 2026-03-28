import { Button } from "@/components/ui/button";
import { Music } from "lucide-react";

interface StepWelcomeProps {
  onNext: () => void;
}

const StepWelcome = ({ onNext }: StepWelcomeProps) => {
  return (
    <div className="flex flex-col items-center text-center space-y-8 animate-in fade-in duration-500">
      <div className="w-20 h-20 rounded-2xl bg-primary/10 flex items-center justify-center">
        <Music className="w-10 h-10 text-primary" />
      </div>
      <div className="space-y-3">
        <h1 className="text-4xl font-bold tracking-tight text-foreground">
          Welcome to Msanii
        </h1>
        <p className="text-lg text-muted-foreground max-w-md">
          Your all-in-one management platform for managing artists, royalties, contracts, and more.
        </p>
      </div>
      <p className="text-sm text-muted-foreground">
        Let's get your profile set up — it only takes a minute.
      </p>
      <Button size="lg" onClick={onNext} className="px-8">
        Get Started
      </Button>
    </div>
  );
};

export default StepWelcome;
