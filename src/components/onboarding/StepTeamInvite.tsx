import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Building2, Check } from "lucide-react";
import { orgNoun } from "@/lib/tiers";

interface StepTeamInviteProps {
  orgName: string | null;
  kind: "self_serve" | "enterprise" | null;
  /** True once the invite has been accepted on the claim page. */
  accepted: boolean;
  onNext: () => void | Promise<void>;
  onBack: () => void;
  isLoading: boolean;
}

/**
 * Replaces the plan step (StepPlan) for a user who arrived through a team
 * invite. A team member keeps their own free tier and spends from the team's
 * credit pool, so selling them a Basic plan here would buy them nothing.
 */
export default function StepTeamInvite({
  orgName,
  kind,
  accepted,
  onNext,
  onBack,
  isLoading,
}: StepTeamInviteProps) {
  const noun = orgNoun(kind);
  const name = orgName ?? `your ${noun}`;

  return (
    <div className="w-full flex flex-col items-center text-center">
      <div className="mx-auto mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
        <Building2 className="h-6 w-6 text-primary" />
      </div>
      <h2 className="text-2xl font-semibold text-foreground mb-2">
        {accepted ? `You're on ${name}` : `You're joining ${name}`}
      </h2>
      <p className="text-sm text-muted-foreground mb-6 max-w-sm">
        {accepted
          ? `Your work runs on ${name}'s credits, so there's no plan to choose.`
          : `Once you accept the invitation, your work runs on ${name}'s credits — there's no plan to choose.`}
      </p>

      <Card className="p-5 text-left border-2 border-primary/40 bg-primary/[0.03] w-full max-w-sm mb-6">
        <ul className="space-y-1.5 text-sm">
          {[
            `Credits come from the ${noun}'s shared pool`,
            "Your artists, projects, and files stay yours",
            "You can still upgrade your own plan later from Billing",
          ].map((b) => (
            <li key={b} className="flex items-start gap-2">
              <Check className="w-3.5 h-3.5 text-primary mt-0.5 shrink-0" />
              <span>{b}</span>
            </li>
          ))}
        </ul>
      </Card>

      <div className="flex flex-col-reverse sm:flex-row items-center justify-center gap-3 w-full">
        <Button variant="ghost" onClick={onBack} disabled={isLoading}>
          Back
        </Button>
        <Button onClick={onNext} disabled={isLoading} className="min-w-[160px]">
          {isLoading ? "..." : "Continue"}
        </Button>
      </div>
    </div>
  );
}
