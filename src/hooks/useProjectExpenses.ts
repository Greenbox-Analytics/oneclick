import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export type ExpenseCategory =
  | "studio"
  | "mixing_mastering"
  | "marketing"
  | "travel"
  | "equipment"
  | "distribution"
  | "other";

// Shared category options + label map, reused by the project Expenses tab,
// the expense form dialog, and the standalone Expense Tracker dashboard.
export const EXPENSE_CATEGORIES: { value: ExpenseCategory; label: string }[] = [
  { value: "studio", label: "Studio" },
  { value: "mixing_mastering", label: "Mixing / Mastering" },
  { value: "marketing", label: "Marketing" },
  { value: "travel", label: "Travel" },
  { value: "equipment", label: "Equipment" },
  { value: "distribution", label: "Distribution" },
  { value: "other", label: "Other" },
];

export const EXPENSE_CATEGORY_LABELS: Record<string, string> = Object.fromEntries(
  EXPENSE_CATEGORIES.map((c) => [c.value, c.label]),
);

export interface ProjectExpense {
  id: string;
  project_id: string;
  created_by: string;
  description: string;
  amount: number;
  category: ExpenseCategory | null;
  incurred_on: string | null;
  work_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface ExpenseInput {
  description: string;
  amount: number;
  category?: ExpenseCategory | null;
  incurred_on?: string | null;
  work_ids?: string[];
}

export interface ExpenseSummaryRow {
  id: string;
  project_id: string;
  project_name: string | null;
  artist_id: string | null;
  artist_name: string | null;
  description: string | null;
  amount: number;
  category: ExpenseCategory | null;
  incurred_on: string | null;
  is_tagged: boolean;
}

export function useProjectExpenses(projectId?: string) {
  const { user } = useAuth();
  return useQuery<ProjectExpense[]>({
    queryKey: ["project-expenses", projectId],
    queryFn: async () => {
      if (!user?.id || !projectId) return [];
      const data = await apiFetch<{ expenses: ProjectExpense[] }>(`${API_URL}/projects/${projectId}/expenses`);
      return data.expenses;
    },
    enabled: !!user?.id && !!projectId,
  });
}

export function useExpenseSummary() {
  const { user } = useAuth();
  return useQuery<ExpenseSummaryRow[]>({
    queryKey: ["expense-summary"],
    queryFn: async () => {
      if (!user?.id) return [];
      const data = await apiFetch<{ expenses: ExpenseSummaryRow[] }>(`${API_URL}/expenses/summary`);
      return data.expenses;
    },
    enabled: !!user?.id,
  });
}

export function useCreateProjectExpense() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ projectId, ...body }: { projectId: string } & ExpenseInput) =>
      apiFetch<{ expense: ProjectExpense }>(`${API_URL}/projects/${projectId}/expenses`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["project-expenses", projectId] });
      queryClient.invalidateQueries({ queryKey: ["expense-summary"] });
      toast.success("Expense added");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

export function useUpdateProjectExpense() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({
      projectId,
      expenseId,
      ...body
    }: { projectId: string; expenseId: string } & Partial<ExpenseInput>) =>
      apiFetch<{ expense: ProjectExpense }>(`${API_URL}/projects/${projectId}/expenses/${expenseId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["project-expenses", projectId] });
      queryClient.invalidateQueries({ queryKey: ["expense-summary"] });
      toast.success("Expense updated");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

export function useDeleteProjectExpense() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ projectId, expenseId }: { projectId: string; expenseId: string }) =>
      apiFetch(`${API_URL}/projects/${projectId}/expenses/${expenseId}`, {
        method: "DELETE",
      }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["project-expenses", projectId] });
      queryClient.invalidateQueries({ queryKey: ["expense-summary"] });
      toast.success("Expense removed");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}
