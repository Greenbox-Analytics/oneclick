export const TOOL_REGISTRY = [
  { id: "oneclick",   label: "OneClick",         status: "live", category: "tool" },
  { id: "zoe",        label: "Zoe",              status: "live", category: "tool" },
  { id: "splitsheet", label: "Split Sheet",      status: "live", category: "tool" },
  { id: "registry",   label: "Metadata Registry",  status: "live", category: "tool" },
  { id: "expense-tracker", label: "Expense Tracker", status: "live", category: "tool" },
  { id: "boards",     label: "Boards",           status: "live", category: "workspace" },
  { id: "calendar",   label: "Calendar",         status: "live", category: "workspace" },
  { id: "drive",      label: "Google Drive",     status: "live", category: "integration" },
  { id: "slack",      label: "Slack",            status: "live", category: "integration" },
] as const;

export type ToolId = (typeof TOOL_REGISTRY)[number]["id"];
export type ToolCategory = (typeof TOOL_REGISTRY)[number]["category"];

export const TOOL_BY_ID: Record<ToolId, (typeof TOOL_REGISTRY)[number]> =
  Object.fromEntries(TOOL_REGISTRY.map((t) => [t.id, t])) as Record<
    ToolId,
    (typeof TOOL_REGISTRY)[number]
  >;
