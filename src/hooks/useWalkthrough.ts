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
      "Access OneClick for rapid document processing, Zoe AI for contract analysis, and Split Sheets for royalty management.",
    placement: "bottom",
  },
  {
    targetSelector: '[data-walkthrough="artists"]',
    title: "Artist Profiles",
    description:
      "Create and manage your artist roster — store DSP links, social media, contracts, and project files all in one place.",
    placement: "bottom",
  },
  {
    targetSelector: '[data-walkthrough="workspace"]',
    title: "Workspace",
    description:
      "Organize projects with Kanban boards, connect Google Drive, Slack, Notion, and Monday.com integrations.",
    placement: "bottom",
  },
  {
    targetSelector: '[data-walkthrough="portfolio"]',
    title: "Portfolio",
    description:
      "See everything at a glance — your artists, projects, and documents organized by year and category.",
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
