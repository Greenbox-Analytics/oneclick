import { useState, useCallback, useEffect, useRef } from "react";
import type { ToolWalkthroughStep, ToolWalkthroughConfig } from "@/config/toolWalkthroughConfig";

type Phase = "idle" | "modal" | "spotlight" | "done";

interface UseToolWalkthroughOptions {
  onBeforeStep?: (stepIndex: number) => void;
}

interface UseToolWalkthroughReturn {
  phase: Phase;
  currentStepIndex: number;
  currentStep: ToolWalkthroughStep | null;
  totalSteps: number;
  visibleStepIndex: number;
  next: () => void;
  skip: () => void;
  replay: () => void;
  startSpotlight: () => void;
}

export const useToolWalkthrough = (
  config: ToolWalkthroughConfig,
  completed: boolean,
  loading: boolean,
  markCompleted: () => Promise<void>,
  options?: UseToolWalkthroughOptions
): UseToolWalkthroughReturn => {
  const [phase, setPhase] = useState<Phase>("idle");
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [visibleStepCount, setVisibleStepCount] = useState(config.steps.length);
  const isReplayRef = useRef(false);
  const hasAutoStarted = useRef(false);

  // Auto-start when completed is false and loading is done
  useEffect(() => {
    if (loading || completed || hasAutoStarted.current) return;
    hasAutoStarted.current = true;
    const timer = setTimeout(() => setPhase("modal"), 500);
    return () => clearTimeout(timer);
  }, [loading, completed]);

  const findNextVisibleStep = useCallback(
    (fromIndex: number): number | null => {
      for (let i = fromIndex; i < config.steps.length; i++) {
        const step = config.steps[i];
        if (step.skipIfMissing) {
          const target = document.querySelector(step.targetSelector);
          if (!target) continue;
        }
        return i;
      }
      return null;
    },
    [config.steps]
  );

  const goToStep = useCallback(
    (index: number) => {
      const step = config.steps[index];
      if (options?.onBeforeStep) {
        options.onBeforeStep(index);
      }

      // Wait a tick for DOM to update after beforeStep (e.g., tab switch)
      setTimeout(() => {
        if (step.skipIfMissing) {
          const target = document.querySelector(step.targetSelector);
          if (!target) {
            const nextVisible = findNextVisibleStep(index + 1);
            if (nextVisible !== null) {
              goToStep(nextVisible);
            } else {
              setPhase("done");
              if (!isReplayRef.current) markCompleted();
            }
            return;
          }
        }
        setCurrentStepIndex(index);
      }, 50);
    },
    [config.steps, options, findNextVisibleStep, markCompleted]
  );

  const startSpotlight = useCallback(() => {
    // Count visible steps for the counter
    let visible = 0;
    for (let i = 0; i < config.steps.length; i++) {
      const step = config.steps[i];
      if (step.skipIfMissing) {
        const target = document.querySelector(step.targetSelector);
        if (!target) continue;
      }
      visible++;
    }
    setVisibleStepCount(visible);

    const firstVisible = findNextVisibleStep(0);
    if (firstVisible === null) {
      setPhase("done");
      if (!isReplayRef.current) markCompleted();
      return;
    }

    setPhase("spotlight");
    goToStep(firstVisible);
  }, [config.steps, findNextVisibleStep, goToStep, markCompleted]);

  const next = useCallback(() => {
    const nextVisible = findNextVisibleStep(currentStepIndex + 1);
    if (nextVisible !== null) {
      goToStep(nextVisible);
    } else {
      setPhase("done");
      if (!isReplayRef.current) markCompleted();
    }
  }, [currentStepIndex, findNextVisibleStep, goToStep, markCompleted]);

  const skip = useCallback(() => {
    setPhase("done");
    if (!isReplayRef.current) markCompleted();
  }, [markCompleted]);

  const replay = useCallback(() => {
    isReplayRef.current = true;
    setCurrentStepIndex(0);
    setPhase("modal");
  }, []);

  // Calculate visible step index for display ("2 / 4" not "3 / 6")
  let visibleStepIndex = 0;
  for (let i = 0; i < currentStepIndex; i++) {
    const step = config.steps[i];
    if (step.skipIfMissing) {
      const target = document.querySelector(step.targetSelector);
      if (!target) continue;
    }
    visibleStepIndex++;
  }

  return {
    phase,
    currentStepIndex,
    currentStep: phase === "spotlight" ? config.steps[currentStepIndex] : null,
    totalSteps: visibleStepCount,
    visibleStepIndex,
    next,
    skip,
    replay,
    startSpotlight,
  };
};
