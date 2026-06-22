import { useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Loader2, Shield, ArrowLeft } from "lucide-react";
import { RequireFeature } from "@/components/paywall/RequireFeature";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { useWorkFull } from "@/hooks/useRegistry";
import { WorkEditor } from "@/components/registry/WorkEditor";
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";

const WorkDetail = () => {
  const navigate = useNavigate();
  const { workId } = useParams<{ workId: string }>();
  const { data: work, isLoading, isError } = useWorkFull(workId);

  // Tool walkthrough — same wiring as the other tool pages. The spotlight steps
  // target elements inside WorkEditor, so we hold off auto-starting until the
  // work has loaded and those anchors exist in the DOM.
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(TOOL_CONFIGS.work_detail, {
    onComplete: () => markToolCompleted("work_detail"),
  });

  useEffect(() => {
    if (
      !onboardingLoading &&
      !statuses.work_detail &&
      work &&
      walkthrough.phase === "idle"
    ) {
      const timer = setTimeout(() => walkthrough.startModal(), 500);
      return () => clearTimeout(timer);
    }
  }, [onboardingLoading, statuses.work_detail, work]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <RequireFeature feature="registry">
      <div className="min-h-screen bg-background">
        <PageHeader
          actions={
            <>
              <ToolHelpButton onClick={() => walkthrough.replay()} />
              <Button
                variant="outline"
                className="hidden md:inline-flex"
                onClick={() => navigate("/tools/registry")}
              >
                <ArrowLeft className="w-4 h-4 mr-1" /> Registry
              </Button>
            </>
          }
        />

        <main className="container mx-auto px-4 py-8 max-w-6xl">
          {isLoading ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="w-6 h-6 animate-spin text-primary" />
            </div>
          ) : isError || !work ? (
            <div className="rounded-lg border border-dashed bg-card p-12 text-center">
              <Shield className="w-10 h-10 text-muted-foreground/40 mx-auto mb-3" />
              <p className="text-muted-foreground mb-4">
                We couldn't load this work. It may have been removed.
              </p>
              <Button onClick={() => navigate("/tools/registry")}>
                <ArrowLeft className="w-4 h-4 mr-1" /> Back to Registry
              </Button>
            </div>
          ) : (
            <WorkEditor work={work} />
          )}
        </main>

        <ToolIntroModal
          config={TOOL_CONFIGS.work_detail}
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

export default WorkDetail;
