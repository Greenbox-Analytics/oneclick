import { capture } from "@/lib/posthog";
import type { ToolId } from "@/lib/analytics-tools";

// Kept for backward-compat with existing call sites
export type ToolName = ToolId;
export type Plan = "monthly" | "annual";
export type PaywallVariant = "page" | "modal" | "inline";
export type PaywallFeature =
  | "zoe"
  | "oneclick"
  | "registry"
  | "uploads"
  | "integrations"
  | "create_cap"
  | "splitsheet_cap";

/**
 * Typed analytics wrappers. Use these instead of `capture()` directly so
 * event names + property shapes stay consistent across the codebase.
 *
 * All wrappers are no-ops when PostHog is disabled (key unset) — the
 * underlying `capture()` short-circuits.
 */
export function useAnalytics() {
  return {
    // Pre-existing wrappers (kept exact)
    captureToolOpened: (tool: ToolId) => capture("tool_opened", { tool }),
    captureToolUsed: (tool: ToolId, props: Record<string, unknown> = {}) =>
      capture("tool_used", { tool, ...props }),
    capturePaywallShown: (
      feature: PaywallFeature,
      variant: PaywallVariant,
      reason?: string,
    ) => capture("paywall_shown", { feature, variant, reason }),
    capturePaywallUpgradeClicked: (feature: PaywallFeature, source: PaywallVariant) =>
      capture("paywall_upgrade_clicked", { feature, source }),
    captureCheckoutStarted: (plan: Plan) => capture("checkout_started", { plan }),
    captureCheckoutCompleted: (plan: Plan) => capture("checkout_completed", { plan }),

    // Admin role grants (DB-backed admin roles)
    captureAdminUserPromoted: (targetUserId: string) =>
      capture("admin_user_promoted", { target_user_id: targetUserId }),
    captureAdminUserDemoted: (targetUserId: string) =>
      capture("admin_user_demoted", { target_user_id: targetUserId }),

    // OneClick (client-side selector dedup handled by callers via useRef)
    captureOneClickContractSelected: (source: "portfolio" | "artist" | "work") =>
      capture("oneclick_contract_selected", { tool: "oneclick", source }),
    captureOneClickStatementSelected: () =>
      capture("oneclick_statement_selected", { tool: "oneclick" }),

    // Zoe
    captureZoeCitationClicked: () =>
      capture("zoe_citation_clicked", { tool: "zoe" }),

    // SplitSheet (form interaction; once per page-instance via useRef)
    captureSplitSheetFormStarted: () =>
      capture("splitsheet_form_started", { tool: "splitsheet" }),
    captureSplitSheetFormCompleted: (collaborator_count: number) =>
      capture("splitsheet_form_completed", {
        tool: "splitsheet",
        collaborator_count,
      }),

    // Onboarding
    captureOnboardingStepCompleted: (step_name: string) =>
      capture("onboarding_step_completed", { step_name }),
    captureOnboardingFinished: () => capture("onboarding_finished", {}),

    // Integrations (client-side "Connect" click)
    captureIntegrationConnectStarted: (tool: "drive" | "slack") =>
      capture("integration_connect_started", { tool }),
  };
}
