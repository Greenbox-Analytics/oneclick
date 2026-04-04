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

    // Scroll target into view first, then position after layout settles
    target.scrollIntoView({ behavior: "smooth", block: "center" });

    const positionTooltip = () => {
      const rect = target.getBoundingClientRect();
      const tooltipWidth = 320;
      const tooltipHeight = tooltipRef.current?.offsetHeight || 200;
      const gap = 12;
      const padding = 16;

      let top = 0;
      let left = 0;

      // position: fixed uses viewport coordinates — no scroll offsets needed
      switch (step.placement) {
        case "bottom":
          top = rect.bottom + gap;
          left = rect.left + rect.width / 2 - tooltipWidth / 2;
          break;
        case "top":
          top = rect.top - gap - tooltipHeight;
          left = rect.left + rect.width / 2 - tooltipWidth / 2;
          break;
        case "right":
          top = rect.top + rect.height / 2 - tooltipHeight / 2;
          left = rect.right + gap;
          break;
        case "left":
          top = rect.top + rect.height / 2 - tooltipHeight / 2;
          left = rect.left - tooltipWidth - gap;
          break;
      }

      // Clamp to viewport
      left = Math.max(padding, Math.min(left, window.innerWidth - tooltipWidth - padding));
      top = Math.max(padding, Math.min(top, window.innerHeight - tooltipHeight - padding));

      setPosition({ top, left });
    };

    // Small delay to let scroll complete and measure tooltip height
    const timer = setTimeout(positionTooltip, 100);
    return () => clearTimeout(timer);
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
