import { useState, useCallback } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";

export interface WalkthroughStep {
  targetSelector: string;
  title: string;
  description: string;
  placement: "top" | "bottom" | "left" | "right";
}

export const WALKTHROUGH_STEPS: WalkthroughStep[] = [
  {
    targetSelector: '[data-walkthrough="tools"]',
    title: "Tools",
    description:
      "OneClick for royalty calculations, Zoe AI for contract analysis, Split Sheets for ownership docs, and the Rights Registry to track and protect your catalog.",
    placement: "bottom",
  },
  {
    targetSelector: '[data-walkthrough="artists"]',
    title: "Artist Profiles",
    description:
      "Create and manage your artist roster. Each profile stores DSP links, contracts, and project files. Artist data feeds into collaboration invitations and TeamCards.",
    placement: "bottom",
  },
  {
    targetSelector: '[data-walkthrough="workspace"]',
    title: "Workspace",
    description:
      "Organize tasks on Kanban boards with monthly iterations. Connect Google Drive, Slack, Notion, and Monday.com to keep everything in sync.",
    placement: "bottom",
  },
  {
    targetSelector: '[data-walkthrough="portfolio"]',
    title: "Portfolio",
    description:
      "All your projects grouped by year and artist. See work counts, team members, and access projects shared with you by collaborators — with your role clearly visible.",
    placement: "bottom",
  },
];

interface UseWalkthroughReturn {
  isActive: boolean;
  currentStepIndex: number;
  currentStep: WalkthroughStep | null;
  totalSteps: number;
  start: () => void;
  next: () => void;
  skip: () => void;
}

export const useWalkthrough = (): UseWalkthroughReturn => {
  const { user } = useAuth();
  const [isActive, setIsActive] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);

  const markCompleted = useCallback(async () => {
    if (!user) return;
    await supabase
      .from("profiles")
      .update({ walkthrough_completed: true, updated_at: new Date().toISOString() })
      .eq("id", user.id);
  }, [user]);

  const start = useCallback(() => {
    setCurrentStepIndex(0);
    setIsActive(true);
  }, []);

  const next = useCallback(() => {
    if (currentStepIndex < WALKTHROUGH_STEPS.length - 1) {
      setCurrentStepIndex((prev) => prev + 1);
    } else {
      setIsActive(false);
      markCompleted();
    }
  }, [currentStepIndex, markCompleted]);

  const skip = useCallback(() => {
    setIsActive(false);
    markCompleted();
  }, [markCompleted]);

  return {
    isActive,
    currentStepIndex,
    currentStep: isActive ? WALKTHROUGH_STEPS[currentStepIndex] : null,
    totalSteps: WALKTHROUGH_STEPS.length,
    start,
    next,
    skip,
  };
};
