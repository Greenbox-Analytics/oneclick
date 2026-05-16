import { capture } from "@/lib/posthog";

export type ToolName = "zoe" | "oneclick" | "registry" | "splitsheet";
export type Plan = "monthly" | "annual";
export type PaywallVariant = "page" | "modal" | "inline";
export type PaywallFeature = "zoe" | "oneclick" | "registry" | "uploads" | "integrations" | "create_cap" | "splitsheet_cap";

/**
 * Typed analytics wrappers. Use these instead of `capture()` directly so
 * event names + property shapes stay consistent across the codebase.
 *
 * All wrappers are no-ops when PostHog is disabled (key unset) — the
 * underlying `capture()` short-circuits.
 */
export function useAnalytics() {
  return {
    captureToolOpened: (tool: ToolName) => capture("tool_opened", { tool }),
    captureToolUsed: (tool: ToolName, props: Record<string, unknown> = {}) =>
      capture("tool_used", { tool, ...props }),
    capturePaywallShown: (feature: PaywallFeature, variant: PaywallVariant, reason?: string) =>
      capture("paywall_shown", { feature, variant, reason }),
    capturePaywallUpgradeClicked: (feature: PaywallFeature, source: PaywallVariant) =>
      capture("paywall_upgrade_clicked", { feature, source }),
    captureCheckoutStarted: (plan: Plan) => capture("checkout_started", { plan }),
    captureCheckoutCompleted: (plan: Plan) => capture("checkout_completed", { plan }),
  };
}
