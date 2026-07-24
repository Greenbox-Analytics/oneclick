import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { RequireFeature } from "@/components/paywall/RequireFeature";
import { useAnalytics } from "@/hooks/useAnalytics";
import { Button } from "@/components/ui/button";
import { HeaderDocsButton } from "@/components/layout/HeaderDocsButton";
import { BookOpen } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";
import { RegistryDashboard } from "@/components/registry/RegistryDashboard";

const Registry = () => {
  const navigate = useNavigate();

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
        <PageHeader
          actions={
            <>
              <ToolHelpButton onClick={() => walkthrough.replay()} />
              <HeaderDocsButton />
              <Button variant="outline" className="hidden md:inline-flex" onClick={() => navigate("/tools")}>
                Back to Tools
              </Button>
            </>
          }
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
