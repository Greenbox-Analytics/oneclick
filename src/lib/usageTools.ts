// src/lib/usageTools.ts
// The four tools a credit action belongs to. Members spend the product actions
// and API keys spend the partner ones; the Usage card shows both under one
// name per tool, in the same colours the billing card uses (TOOL_META, which
// itself reads --t-* from index.css).
import type { CreditAction } from "@/hooks/useCreditUsage";
import { TOOL_META } from "@/lib/credits";

export type ToolId = "oneclick" | "registry" | "splitsheet" | "zoe";

export type UsageAction = CreditAction | `partner_${CreditAction}`;

// A Record over the CreditAction/partner union is exhaustive: adding a new
// CreditAction without mapping it here is a compile error, not a silent drop.
const ACTION_TOOL: Record<UsageAction, ToolId> = {
  oneclick_run: "oneclick",
  partner_oneclick_run: "oneclick",
  registry_parse: "registry",
  partner_registry_parse: "registry",
  split_sheet: "splitsheet",
  partner_split_sheet: "splitsheet",
  zoe_message: "zoe",
  partner_zoe_message: "zoe",
};

export interface Tool {
  id: ToolId;
  label: string;
  actions: readonly string[];
  color: string;
}

const actionsFor = (id: ToolId): UsageAction[] =>
  (Object.keys(ACTION_TOOL) as UsageAction[]).filter((a) => ACTION_TOOL[a] === id);

export const TOOLS: readonly Tool[] = [
  { id: "oneclick", label: "OneClick", actions: actionsFor("oneclick"), color: TOOL_META.oneclick_run.color },
  { id: "registry", label: "Registry", actions: actionsFor("registry"), color: TOOL_META.registry_parse.color },
  { id: "splitsheet", label: "Split sheet", actions: actionsFor("splitsheet"), color: TOOL_META.split_sheet.color },
  { id: "zoe", label: "Zoe", actions: actionsFor("zoe"), color: TOOL_META.zoe_message.color },
];

export function toolOf(action: string): ToolId | null {
  return Object.hasOwn(ACTION_TOOL, action) ? ACTION_TOOL[action as UsageAction] : null;
}
