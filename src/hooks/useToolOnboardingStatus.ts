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
};

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
          setStatuses({
            oneclick: data.oneclick_completed,
            zoe: data.zoe_completed,
            splitsheet: data.splitsheet_completed,
            artists: data.artists_completed,
            workspace: data.workspace_completed,
            portfolio: data.portfolio_completed,
          });
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
        return;
      }

      setStatuses((prev) => ({ ...prev, [toolKey]: true }));
    },
    [user]
  );

  return { statuses, loading, markToolCompleted };
};
