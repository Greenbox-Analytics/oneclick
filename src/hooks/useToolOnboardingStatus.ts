import { useEffect, useState, useCallback } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";

const TOOL_KEYS = [
  "oneclick",
  "zoe",
  "splitsheet",
  "artists",
  "workspace",
  "portfolio",
  "registry",
  "project_detail",
  "profile",
  "work_detail",
] as const;

type ToolKey = (typeof TOOL_KEYS)[number];

interface ToolOnboardingStatus {
  statuses: Record<ToolKey, boolean>;
  loading: boolean;
  markToolCompleted: (toolKey: ToolKey) => Promise<void>;
}

const defaultStatuses: Record<ToolKey, boolean> = {
  oneclick: false,
  zoe: false,
  splitsheet: false,
  artists: false,
  workspace: false,
  portfolio: false,
  registry: false,
  project_detail: false,
  profile: false,
  work_detail: false,
};

const STORAGE_KEY = "msanii_tour_completed";

/** Read local fallback flags (used when DB columns don't exist yet) */
function getLocalFlags(): Partial<Record<ToolKey, boolean>> {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

/** Persist a flag locally as immediate fallback */
function setLocalFlag(toolKey: ToolKey) {
  try {
    const flags = getLocalFlags();
    flags[toolKey] = true;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(flags));
  } catch {
    // localStorage unavailable — ignore
  }
}

export const useToolOnboardingStatus = (): ToolOnboardingStatus => {
  const { user } = useAuth();
  const [statuses, setStatuses] = useState<Record<ToolKey, boolean>>(defaultStatuses);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStatus = async () => {
      if (!user) {
        setLoading(false);
        return;
      }

      const localFlags = getLocalFlags();

      try {
        const { data, error } = await supabase
          .from("user_onboarding")
          .select("*")
          .eq("user_id", user.id)
          .single();

        if (error && error.code !== "PGRST116") {
          console.error("Error fetching tool onboarding status:", error);
        }

        if (data) {
          // Use DB value if present, fall back to localStorage flag
          setStatuses({
            oneclick: data.oneclick_completed ?? localFlags.oneclick ?? false,
            zoe: data.zoe_completed ?? localFlags.zoe ?? false,
            splitsheet: data.splitsheet_completed ?? localFlags.splitsheet ?? false,
            artists: data.artists_completed ?? localFlags.artists ?? false,
            workspace: data.workspace_completed ?? localFlags.workspace ?? false,
            portfolio: data.portfolio_completed ?? localFlags.portfolio ?? false,
            registry: data.registry_completed ?? localFlags.registry ?? false,
            project_detail: data.project_detail_completed ?? localFlags.project_detail ?? false,
            profile: data.profile_completed ?? localFlags.profile ?? false,
            work_detail: data.work_detail_completed ?? localFlags.work_detail ?? false,
          });
        } else {
          // No DB row yet — use local flags only
          setStatuses((prev) => ({ ...prev, ...localFlags }));
        }
      } finally {
        setLoading(false);
      }
    };

    fetchStatus();
  }, [user]);

  const markToolCompleted = useCallback(
    async (toolKey: ToolKey) => {
      if (!user) return;

      // Immediately update local state + localStorage so the tour never retriggers
      setStatuses((prev) => ({ ...prev, [toolKey]: true }));
      setLocalFlag(toolKey);

      // Persist to DB in background
      const columnName = `${toolKey}_completed` as const;

      const { error } = await supabase
        .from("user_onboarding")
        .upsert(
          {
            user_id: user.id,
            [columnName]: true,
          },
          { onConflict: "user_id" }
        );

      if (error) {
        console.error("Error marking tool onboarding complete:", error);
      }
    },
    [user]
  );

  return { statuses, loading, markToolCompleted };
};
