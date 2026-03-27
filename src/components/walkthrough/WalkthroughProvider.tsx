import { useEffect, useState } from "react";
import WalkthroughTooltip from "./WalkthroughTooltip";
import type { WalkthroughStep } from "@/hooks/useWalkthrough";

interface WalkthroughProviderProps {
  isActive: boolean;
  currentStep: WalkthroughStep | null;
  currentStepIndex: number;
  totalSteps: number;
  onNext: () => void;
  onSkip: () => void;
}

const WalkthroughProvider = ({
  isActive,
  currentStep,
  currentStepIndex,
  totalSteps,
  onNext,
  onSkip,
}: WalkthroughProviderProps) => {
  const [spotlightRect, setSpotlightRect] = useState<DOMRect | null>(null);

  useEffect(() => {
    if (!isActive || !currentStep) {
      setSpotlightRect(null);
      return;
    }

    const updateSpotlight = () => {
      const target = document.querySelector(currentStep.targetSelector);
      if (target) {
        setSpotlightRect(target.getBoundingClientRect());
      }
    };

    updateSpotlight();

    window.addEventListener("resize", updateSpotlight);
    window.addEventListener("scroll", updateSpotlight);

    return () => {
      window.removeEventListener("resize", updateSpotlight);
      window.removeEventListener("scroll", updateSpotlight);
    };
  }, [isActive, currentStep]);

  if (!isActive || !currentStep) return null;

  const padding = 8;

  return (
    <>
      {/* Backdrop with spotlight cutout */}
      <div className="fixed inset-0 z-50 pointer-events-auto">
        <svg className="absolute inset-0 w-full h-full">
          <defs>
            <mask id="walkthrough-mask">
              <rect width="100%" height="100%" fill="white" />
              {spotlightRect && (
                <rect
                  x={spotlightRect.left - padding}
                  y={spotlightRect.top - padding}
                  width={spotlightRect.width + padding * 2}
                  height={spotlightRect.height + padding * 2}
                  rx="12"
                  fill="black"
                />
              )}
            </mask>
          </defs>
          <rect
            width="100%"
            height="100%"
            fill="rgba(0,0,0,0.5)"
            mask="url(#walkthrough-mask)"
          />
        </svg>
      </div>

      {/* Tooltip */}
      <WalkthroughTooltip
        step={currentStep}
        stepIndex={currentStepIndex}
        totalSteps={totalSteps}
        onNext={onNext}
        onSkip={onSkip}
      />
    </>
  );
};

export default WalkthroughProvider;
