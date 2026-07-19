import { useQuery, useMutation, useQueryClient, keepPreviousData, type QueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";

// ---------------------------------------------------------------------------
// TypeScript interfaces (mirroring src/backend/oneclick/royalties/models.py)
// ---------------------------------------------------------------------------

export interface PayeeSummary {
  id: string;
  display_name: string;
  payout_currency: string;
  registry_user_id?: string;
  email?: string;
  collision: boolean;
  project_count: number;
  status: string; // "owed" | "scheduled" | "settled"
  // reporting-currency totals
  earned: number;
  paid: number;
  drafted: number;
  owed: number; // earned − paid − drafted (available to draft; drives status/eligibility)
  unpaid: number; // earned − paid (outstanding until actually paid; drives "Outstanding" displays)
  // payee payout-currency totals
  earned_native: number;
  paid_native: number;
  drafted_native: number;
  owed_native: number;
  unpaid_native: number;
}

export interface PayeeLine {
  line_id: string;
  song_title: string;
  role?: string;
  royalty_type?: string;
  percentage?: number;
  amount_owed: number; // statement currency
  statement_currency: string;
}

export interface PayeeStatement {
  royalty_statement_id: string;
  period_start?: string;
  period_end?: string;
  statement_currency: string;
  statement_total?: number;
  earned: number;
  paid: number;
  drafted: number;
  owed: number;
  unpaid: number; // earned − paid (outstanding until actually paid)
  state: string; // "owed" | "scheduled" | "settled"
  lines: PayeeLine[];
}

export interface PayeeProject {
  project_id: string;
  name: string;
  statements: PayeeStatement[];
}

export interface PayeeDetail {
  summary: PayeeSummary;
  projects: PayeeProject[];
  payouts: PayoutOut[];
}

export interface PeriodCell {
  royalty_statement_id: string;
  period_start?: string;
  period_end?: string;
  earned: number;
  state: string;
}

export interface PeriodLedgerRow {
  payee_id: string;
  display_name: string;
  cells: PeriodCell[];
  total: number;
}

export interface PeriodLedger {
  base: string;
  rows: PeriodLedgerRow[];
}

export interface PayoutOut {
  id: string;
  payee_id: string;
  status: string; // "draft" | "paid"
  pay_currency: string;
  fx_rate_date: string;
  total_amount: number;
  note?: string;
  created_at: string;
  paid_at?: string;
  breakdown_snapshot: Record<string, unknown>;
  orphan_state: string; // "none" | "partial" | "orphaned"
  payment_method?: string; // "manual" | "paypal"
  paypal_capture_id?: string;
}

// ---------------------------------------------------------------------------
// Request payload interfaces
// ---------------------------------------------------------------------------

export interface CreatePayoutPayload {
  payee_ids: string[];
  idempotency_key?: string;
  note?: string;
}

export interface PatchPayeePayload {
  payout_currency?: string;
  registry_user_id?: string;
  email?: string;
}

export interface SplitPayeePayload {
  line_ids: string[];
  new_display_name: string;
}

// ---------------------------------------------------------------------------
// Query hooks
// ---------------------------------------------------------------------------

/** GET /oneclick/royalties/payees?base={base} */
export function useRoyaltyPayees(base: string) {
  const { user } = useAuth();
  return useQuery<PayeeSummary[]>({
    queryKey: ["royalty-payees", user?.id, base],
    queryFn: () =>
      apiFetch<PayeeSummary[]>(
        `${API_URL}/oneclick/royalties/payees?base=${encodeURIComponent(base)}`,
      ),
    enabled: !!user?.id,
    staleTime: 60_000,
    placeholderData: keepPreviousData,
  });
}

/** GET /oneclick/royalties/payees/{payeeId}?base={base} */
export function useRoyaltyPayee(payeeId: string | null | undefined, base: string) {
  const { user } = useAuth();
  return useQuery<PayeeDetail>({
    queryKey: ["royalty-payee", user?.id, payeeId, base],
    queryFn: () =>
      apiFetch<PayeeDetail>(
        `${API_URL}/oneclick/royalties/payees/${payeeId}?base=${encodeURIComponent(base)}`,
      ),
    enabled: !!user?.id && !!payeeId,
    staleTime: 60_000,
    placeholderData: keepPreviousData,
  });
}

/** GET /oneclick/royalties/periods?base={base} */
export function useRoyaltyPeriods(base: string) {
  const { user } = useAuth();
  return useQuery<PeriodLedger>({
    queryKey: ["royalty-periods", user?.id, base],
    queryFn: () =>
      apiFetch<PeriodLedger>(
        `${API_URL}/oneclick/royalties/periods?base=${encodeURIComponent(base)}`,
      ),
    enabled: !!user?.id,
    staleTime: 60_000,
    placeholderData: keepPreviousData,
  });
}

/** GET /oneclick/royalties/payouts */
export function useRoyaltyPayouts() {
  const { user } = useAuth();
  return useQuery<PayoutOut[]>({
    queryKey: ["royalty-payouts", user?.id],
    queryFn: () => apiFetch<PayoutOut[]>(`${API_URL}/oneclick/royalties/payouts`),
    enabled: !!user?.id,
    staleTime: 60_000,
    placeholderData: keepPreviousData,
  });
}

// ---------------------------------------------------------------------------
// Mutation hooks
// ---------------------------------------------------------------------------

/** POST /oneclick/royalties/payouts */
/**
 * A payout/payee financial change also feeds the analytics dashboards
 * (overview "Paid over time" + "Top outstanding", per-artist, per-payee).
 * Invalidate those alongside the core lists on every such change, or the
 * charts stay stale until a full page reload.
 */
function invalidateRoyaltyData(qc: QueryClient) {
  qc.invalidateQueries({ queryKey: ["royalty-payees"] });
  qc.invalidateQueries({ queryKey: ["royalty-payee"] });
  qc.invalidateQueries({ queryKey: ["royalty-payouts"] });
  qc.invalidateQueries({ queryKey: ["royalty-periods"] });
  qc.invalidateQueries({ queryKey: ["royalty-analytics-overview"] });
  qc.invalidateQueries({ queryKey: ["royalty-analytics-artist"] });
  qc.invalidateQueries({ queryKey: ["royalty-analytics-payee"] });
}

export function useCreatePayout() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: CreatePayoutPayload) => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch<PayoutOut[]>(`${API_URL}/oneclick/royalties/payouts`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
    },
    onSuccess: () => invalidateRoyaltyData(queryClient),
  });
}

/** POST /oneclick/royalties/payouts/{id}/pay */
export function useMarkPayoutPaid() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch<PayoutOut>(`${API_URL}/oneclick/royalties/payouts/${id}/pay`, {
        method: "POST",
      });
    },
    onSuccess: () => invalidateRoyaltyData(queryClient),
  });
}

/** POST /oneclick/royalties/payouts/{id}/cancel */
export function useCancelPayout() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch<PayoutOut>(`${API_URL}/oneclick/royalties/payouts/${id}/cancel`, {
        method: "POST",
      });
    },
    onSuccess: () => invalidateRoyaltyData(queryClient),
  });
}

/** POST /oneclick/royalties/payouts/{id}/paypal/order — create a PayPal checkout order for a draft payout. */
export function useCreatePaypalOrder() {
  const { user } = useAuth();
  return useMutation({
    mutationFn: (id: string) => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch<{ paypal_order_id: string }>(
        `${API_URL}/oneclick/royalties/payouts/${id}/paypal/order`,
        { method: "POST" },
      );
    },
  });
}

/** POST /oneclick/royalties/payouts/{id}/paypal/capture — capture the approved order, marks payout paid. */
export function useCapturePaypalOrder() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch<PayoutOut>(`${API_URL}/oneclick/royalties/payouts/${id}/paypal/capture`, {
        method: "POST",
      });
    },
    onSuccess: () => invalidateRoyaltyData(queryClient),
  });
}

/** POST /oneclick/royalties/payouts/{id}/receipt/save — save the receipt PDF into a project's files. */
export function useSaveReceiptToProject() {
  const { user } = useAuth();
  return useMutation({
    mutationFn: ({ payoutId, artist_id, project_id }: { payoutId: string; artist_id: string; project_id: string }) => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch<Record<string, unknown>>(
        `${API_URL}/oneclick/royalties/payouts/${payoutId}/receipt/save`,
        { method: "POST", body: JSON.stringify({ artist_id, project_id }) },
      );
    },
  });
}

/** PATCH /oneclick/royalties/payees/{id} */
export function useSetPayeeCurrency() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...payload }: { id: string } & PatchPayeePayload) => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch<PayeeSummary>(`${API_URL}/oneclick/royalties/payees/${id}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["royalty-payees"] });
      queryClient.invalidateQueries({ queryKey: ["royalty-payee"] });
    },
  });
}

/** POST /oneclick/royalties/payees/{id}/split */
export function useSplitPayee() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...payload }: { id: string } & SplitPayeePayload) => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch<PayeeSummary[]>(`${API_URL}/oneclick/royalties/payees/${id}/split`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["royalty-payees"] });
      queryClient.invalidateQueries({ queryKey: ["royalty-payee"] });
    },
  });
}

export interface DeleteProjectRoyaltiesResult {
  deleted_calculations: number;
  project_id: string;
}

/** DELETE /oneclick/royalties/projects/{projectId}/entries */
export function useDeleteProjectRoyalties() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (projectId: string) => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch<DeleteProjectRoyaltiesResult>(
        `${API_URL}/oneclick/royalties/projects/${projectId}/entries`,
        { method: "DELETE" },
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["royalty-payees"] });
      queryClient.invalidateQueries({ queryKey: ["royalty-payee"] });
      queryClient.invalidateQueries({ queryKey: ["royalty-payouts"] });
      queryClient.invalidateQueries({ queryKey: ["royalty-periods"] });
      queryClient.invalidateQueries({ queryKey: ["royalty-fx"] });
    },
  });
}
