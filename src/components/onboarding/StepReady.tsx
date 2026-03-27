import { Button } from "@/components/ui/button";
import { CheckCircle2 } from "lucide-react";

interface StepReadyProps {
  preferredName: string;
  firstName: string;
  onFinish: () => void;
  isLoading: boolean;
}

const StepReady = ({ preferredName, firstName, onFinish, isLoading }: StepReadyProps) => {
  const displayName = preferredName.trim() || firstName.trim() || "there";

  return (
    <div className="flex flex-col items-center text-center space-y-8 animate-in fade-in duration-500">
      <div className="w-16 h-16 rounded-full bg-green-500/10 flex items-center justify-center">
        <CheckCircle2 className="w-8 h-8 text-green-500" />
      </div>
      <div className="space-y-3">
        <h2 className="text-3xl font-bold text-foreground">
          You're all set, {displayName}!
        </h2>
        <p className="text-muted-foreground max-w-md">
          Your profile is ready. We'll give you a quick tour of the tools
          when you reach the dashboard.
        </p>
      </div>
      <Button size="lg" onClick={onFinish} disabled={isLoading} className="px-8">
        {isLoading ? "Setting up..." : "Go to Dashboard"}
      </Button>
    </div>
  );
};

export default StepReady;
