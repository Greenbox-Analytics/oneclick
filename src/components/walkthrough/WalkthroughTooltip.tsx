import { useEffect, useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import type { WalkthroughStep } from "@/hooks/useWalkthrough";

interface WalkthroughTooltipProps {
  step: WalkthroughStep;
  stepIndex: number;
  totalSteps: number;
  onNext: () => void;
  onSkip: () => void;
}

const WalkthroughTooltip = ({
  step,
  stepIndex,
  totalSteps,
  onNext,
  onSkip,
}: WalkthroughTooltipProps) => {
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const tooltipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const target = document.querySelector(step.targetSelector);
    if (!target) return;

    const rect = target.getBoundingClientRect();
    const tooltipWidth = 320;
    const gap = 12;

    let top = 0;
    let left = 0;

    switch (step.placement) {
      case "bottom":
        top = rect.bottom + gap + window.scrollY;
        left = rect.left + rect.width / 2 - tooltipWidth / 2 + window.scrollX;
        break;
      case "top":
        top = rect.top - gap + window.scrollY;
        left = rect.left + rect.width / 2 - tooltipWidth / 2 + window.scrollX;
        break;
      case "right":
        top = rect.top + rect.height / 2 + window.scrollY;
        left = rect.right + gap + window.scrollX;
        break;
      case "left":
        top = rect.top + rect.height / 2 + window.scrollY;
        left = rect.left - tooltipWidth - gap + window.scrollX;
        break;
    }

    // Clamp to viewport
    left = Math.max(16, Math.min(left, window.innerWidth - tooltipWidth - 16));

    setPosition({ top, left });

    // Scroll target into view
    target.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [step]);

  const isLast = stepIndex === totalSteps - 1;

  return (
    <div
      ref={tooltipRef}
      className="fixed z-[60] w-80 bg-card border border-border rounded-xl shadow-2xl p-5 animate-in fade-in slide-in-from-bottom-2 duration-300"
      style={{ top: position.top, left: position.left }}
    >
      <div className="space-y-2 mb-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-foreground">{step.title}</h3>
          <span className="text-xs text-muted-foreground">
            {stepIndex + 1} / {totalSteps}
          </span>
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed">
          {step.description}
        </p>
      </div>
      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={onSkip}>
          Skip tour
        </Button>
        <Button size="sm" onClick={onNext}>
          {isLast ? "Finish" : "Next"}
        </Button>
      </div>
    </div>
  );
};

export default WalkthroughTooltip;
