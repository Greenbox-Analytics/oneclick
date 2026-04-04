import { useState, useCallback, useRef } from "react";
import type { ToolWalkthroughStep, ToolWalkthroughConfig } from "@/config/toolWalkthroughConfig";

type Phase = "idle" | "modal" | "spotlight" | "done";

interface UseToolWalkthroughOptions {
  onBeforeStep?: (stepIndex: number) => void;
  onComplete?: () => void;
}

interface UseToolWalkthroughReturn {
  phase: Phase;
  currentStepIndex: number;
  currentStep: ToolWalkthroughStep | null;
  totalSteps: number;
  visibleStepIndex: number;
  startModal: () => void;
  startSpotlight: () => void;
  next: () => void;
  skip: () => void;
  replay: () => void;
}

export const useToolWalkthrough = (
  config: ToolWalkthroughConfig,
  options?: UseToolWalkthroughOptions
): UseToolWalkthroughReturn => {
  const [phase, setPhase] = useState<Phase>("idle");
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [visibleStepCount, setVisibleStepCount] = useState(config.steps.length);
  const isReplayRef = useRef(false);
  // Store callbacks in refs to avoid dependency chains
  const onCompleteRef = useRef(options?.onComplete);
  const onBeforeStepRef = useRef(options?.onBeforeStep);
  onCompleteRef.current = options?.onComplete;
  onBeforeStepRef.current = options?.onBeforeStep;

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

  const fireComplete = useCallback(() => {
    if (!isReplayRef.current && onCompleteRef.current) {
      onCompleteRef.current();
    }
  }, []);

  const goToStep = useCallback(
    (index: number) => {
      const step = config.steps[index];
      if (onBeforeStepRef.current) {
        onBeforeStepRef.current(index);
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
              fireComplete();
            }
            return;
          }
        }
        setCurrentStepIndex(index);
      }, 50);
    },
    [config.steps, findNextVisibleStep, fireComplete]
  );

  const startModal = useCallback(() => {
    isReplayRef.current = false;
    setPhase("modal");
  }, []);

  const startSpotlight = useCallback(() => {
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
      fireComplete();
      return;
    }

    setPhase("spotlight");
    goToStep(firstVisible);
  }, [config.steps, findNextVisibleStep, goToStep, fireComplete]);

  const next = useCallback(() => {
    const nextVisible = findNextVisibleStep(currentStepIndex + 1);
    if (nextVisible !== null) {
      goToStep(nextVisible);
    } else {
      setPhase("done");
      fireComplete();
    }
  }, [currentStepIndex, findNextVisibleStep, goToStep, fireComplete]);

  const skip = useCallback(() => {
    setPhase("done");
    fireComplete();
  }, [fireComplete]);

  const replay = useCallback(() => {
    isReplayRef.current = true;
    setCurrentStepIndex(0);
    setPhase("modal");
  }, []);

  // Calculate visible step index for display
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
    startModal,
    startSpotlight,
    next,
    skip,
    replay,
  };
};
