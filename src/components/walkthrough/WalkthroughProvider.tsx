import { useEffect, useState, useRef, useCallback } from "react";
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

interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

function easeOutCubic(t: number) {
  return 1 - Math.pow(1 - t, 3);
}

const WalkthroughProvider = ({
  isActive,
  currentStep,
  currentStepIndex,
  totalSteps,
  onNext,
  onSkip,
}: WalkthroughProviderProps) => {
  const [targetRect, setTargetRect] = useState<Rect | null>(null);
  const [animatedRect, setAnimatedRect] = useState<Rect | null>(null);
  const [overlayOpacity, setOverlayOpacity] = useState(0);
  const animationRef = useRef<number>();
  const prevRectRef = useRef<Rect | null>(null);

  // Smooth interpolation between spotlight positions
  const animateTo = useCallback((target: Rect) => {
    if (animationRef.current) cancelAnimationFrame(animationRef.current);

    const start = prevRectRef.current;
    if (!start) {
      // First step — snap immediately
      setAnimatedRect(target);
      prevRectRef.current = target;
      return;
    }

    const startTime = performance.now();
    const duration = 350;

    const tick = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const t = easeOutCubic(progress);

      const interpolated: Rect = {
        left: start.left + (target.left - start.left) * t,
        top: start.top + (target.top - start.top) * t,
        width: start.width + (target.width - start.width) * t,
        height: start.height + (target.height - start.height) * t,
      };

      setAnimatedRect(interpolated);

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(tick);
      } else {
        prevRectRef.current = target;
      }
    };

    animationRef.current = requestAnimationFrame(tick);
  }, []);

  // Resolve target element rect
  useEffect(() => {
    if (!isActive || !currentStep) {
      setTargetRect(null);
      setAnimatedRect(null);
      prevRectRef.current = null;
      return;
    }

    const updateTarget = () => {
      const target = document.querySelector(currentStep.targetSelector);
      if (target) {
        const r = target.getBoundingClientRect();
        setTargetRect({ left: r.left, top: r.top, width: r.width, height: r.height });
      }
    };

    updateTarget();

    window.addEventListener("resize", updateTarget);
    window.addEventListener("scroll", updateTarget);

    return () => {
      window.removeEventListener("resize", updateTarget);
      window.removeEventListener("scroll", updateTarget);
    };
  }, [isActive, currentStep]);

  // Animate spotlight to new target
  useEffect(() => {
    if (targetRect) animateTo(targetRect);
  }, [targetRect, animateTo]);

  // Fade overlay in / out
  useEffect(() => {
    if (isActive) {
      // Fade in after a microtask so the initial render is at opacity 0
      requestAnimationFrame(() => setOverlayOpacity(1));
    } else {
      setOverlayOpacity(0);
    }
  }, [isActive]);

  // Cleanup animation on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, []);

  if (!isActive || !currentStep) return null;

  const padding = 8;

  return (
    <>
      {/* Backdrop with spotlight cutout */}
      <div
        className="fixed inset-0 z-50 pointer-events-auto"
        style={{
          opacity: overlayOpacity,
          transition: "opacity 0.3s ease",
        }}
      >
        <svg className="absolute inset-0 w-full h-full">
          <defs>
            <mask id="walkthrough-mask">
              <rect width="100%" height="100%" fill="white" />
              {animatedRect && (
                <rect
                  x={animatedRect.left - padding}
                  y={animatedRect.top - padding}
                  width={animatedRect.width + padding * 2}
                  height={animatedRect.height + padding * 2}
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

      {/* Tooltip — keyed by step so animate-in retriggers on each step change */}
      <WalkthroughTooltip
        key={`step-${currentStepIndex}`}
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
