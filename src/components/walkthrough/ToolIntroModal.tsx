import { Button } from "@/components/ui/button";
import type { ToolWalkthroughConfig } from "@/config/toolWalkthroughConfig";

interface ToolIntroModalProps {
  config: ToolWalkthroughConfig;
  isOpen: boolean;
  onStartTour: () => void;
  onSkip: () => void;
}

const ToolIntroModal = ({ config, isOpen, onStartTour, onSkip }: ToolIntroModalProps) => {
  if (!isOpen) return null;

  const Icon = config.intro.icon;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/50" />
      <div className="relative z-10 max-w-md w-full mx-4 bg-card border border-border rounded-2xl shadow-2xl p-8 animate-in fade-in zoom-in-95 duration-300">
        <div className="flex flex-col items-center text-center space-y-6">
          <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center">
            <Icon className="w-8 h-8 text-primary" />
          </div>
          <div className="space-y-2">
            <h2 className="text-2xl font-bold text-foreground">
              {config.intro.title}
            </h2>
            <p className="text-muted-foreground leading-relaxed">
              {config.intro.description}
            </p>
          </div>
          <div className="flex gap-3 w-full">
            <Button variant="ghost" onClick={onSkip} className="flex-1">
              Skip
            </Button>
            <Button onClick={onStartTour} className="flex-1">
              Start Tour
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ToolIntroModal;
