import { useEffect } from "react";
import { RequireFeature } from "@/components/paywall/RequireFeature";
import { useAnalytics } from "@/hooks/useAnalytics";
import { PageHeader } from "@/components/layout/PageHeader";
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";
import { RegistryDashboard } from "@/components/registry/RegistryDashboard";

const Registry = () => {
  const { captureToolOpened } = useAnalytics();
  useEffect(() => {
    captureToolOpened("registry");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Tool walkthrough — same wiring as before, attribute targets unchanged.
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(TOOL_CONFIGS.registry, {
    onComplete: () => markToolCompleted("registry"),
  });

  useEffect(() => {
    if (!onboardingLoading && !statuses.registry && walkthrough.phase === "idle") {
      const timer = setTimeout(() => walkthrough.startModal(), 500);
      return () => clearTimeout(timer);
    }
  }, [onboardingLoading, statuses.registry]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <RequireFeature feature="registry">
      <div className="min-h-screen bg-background">
        {/* One Back, one destination: the header's Back returns to wherever
            the user came from and falls back to /tools. A second "Back to
            Tools" button beside it disagreed with it half the time. */}
        <PageHeader
          backTo="/tools"
          actions={<ToolHelpButton onClick={() => walkthrough.replay()} />}
        />

        <main className="container mx-auto px-4 py-8 max-w-6xl">
          <RegistryDashboard />
        </main>

        <ToolIntroModal
          config={TOOL_CONFIGS.registry}
          isOpen={walkthrough.phase === "modal"}
          onStartTour={walkthrough.startSpotlight}
          onSkip={walkthrough.skip}
        />
        <WalkthroughProvider
          isActive={walkthrough.phase === "spotlight"}
          currentStep={walkthrough.currentStep}
          currentStepIndex={walkthrough.visibleStepIndex}
          totalSteps={walkthrough.totalSteps}
          onNext={walkthrough.next}
          onSkip={walkthrough.skip}
        />
      </div>
    </RequireFeature>
  );
};

export default Registry;
