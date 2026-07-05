import { useEffect, useMemo, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Loader2, Trash2, Receipt, Plus } from "lucide-react";
import ExpenseFormDialog from "@/components/project/ExpenseFormDialog";
import { useWorksByProject } from "@/hooks/useRegistry";
import type { ProjectExpense } from "@/hooks/useProjectExpenses";

export interface ReviewExpense {
  id: string;
  description?: string;
  amount: number;
  category?: string | null;
  incurred_on?: string | null;
  work_ids?: string[];
  work_titles?: string[];
}

interface ExpenseReviewDialogProps {
  open: boolean;
  expenses: ReviewExpense[];
  isSubmitting: boolean;
  /** Project this calculation belongs to; enables adding expenses inline. */
  projectId?: string;
  onConfirm: (expenses: ReviewExpense[]) => void;
  onCancel: () => void;
}

const formatCurrency = (n: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(n || 0);

/**
 * Shown before finalizing a calculation when one or more collaborators are paid
 * on NET income. The user can confirm, edit amounts, or remove expenses; the
 * edits are session-scoped (they do not mutate the project's expense tracker).
 */
const ExpenseReviewDialog = ({
  open,
  expenses,
  isSubmitting,
  projectId,
  onConfirm,
  onCancel,
}: ExpenseReviewDialogProps) => {
  const [rows, setRows] = useState<ReviewExpense[]>(expenses);
  const [addOpen, setAddOpen] = useState(false);

  // Re-seed when a new calculation surfaces a fresh expense set.
  useEffect(() => {
    setRows(expenses);
  }, [expenses]);

  // Resolve work_ids -> titles so an inline-added expense allocates to the
  // right track (the net recalculation keys allocation on work_titles).
  const { data: works } = useWorksByProject(projectId || undefined);
  const workTitleById = useMemo(() => {
    const map = new Map<string, string>();
    (works ?? []).forEach((w) => map.set(w.id, w.title));
    return map;
  }, [works]);

  const handleExpenseAdded = (expense?: ProjectExpense) => {
    if (!expense) return;
    const workIds = expense.work_ids ?? [];
    const reviewRow: ReviewExpense = {
      id: expense.id,
      description: expense.description,
      amount: expense.amount,
      category: expense.category,
      incurred_on: expense.incurred_on,
      work_ids: workIds,
      work_titles: workIds
        .map((id) => workTitleById.get(id))
        .filter((t): t is string => Boolean(t)),
    };
    // Avoid duplicates if the same expense id is somehow added twice.
    setRows((rs) => (rs.some((r) => r.id === reviewRow.id) ? rs : [...rs, reviewRow]));
  };

  const updateAmount = (id: string, value: string) => {
    const amount = parseFloat(value);
    setRows((rs) => rs.map((r) => (r.id === id ? { ...r, amount: isNaN(amount) ? 0 : amount } : r)));
  };

  const removeRow = (id: string) => {
    setRows((rs) => rs.filter((r) => r.id !== id));
  };

  const total = rows.reduce((sum, r) => sum + (r.amount || 0), 0);

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onCancel()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Review track expenses</DialogTitle>
          <DialogDescription>
            Some collaborators on this contract are paid on <strong>net income</strong>, so these
            expenses will be deducted from each track's earnings before their share is applied.
            Confirm, edit, or remove any that shouldn't count. Changes here only affect this
            calculation.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-2 py-2 max-h-[50vh] overflow-y-auto">
          {rows.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <Receipt className="w-8 h-8 text-muted-foreground/40 mb-2" />
              <p className="text-sm text-muted-foreground">No expenses — net rows will equal gross.</p>
              {projectId && (
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-3"
                  onClick={() => setAddOpen(true)}
                >
                  <Plus className="w-3.5 h-3.5 mr-1.5" />
                  Add expense
                </Button>
              )}
            </div>
          ) : (
            rows.map((row) => (
              <div
                key={row.id}
                className="flex items-center gap-3 rounded-md border border-border p-3"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{row.description || "Expense"}</p>
                  <div className="flex items-center gap-1.5 flex-wrap mt-1">
                    {row.category && (
                      <Badge variant="outline" className="text-xs">
                        {row.category}
                      </Badge>
                    )}
                    {row.work_titles && row.work_titles.length > 0 ? (
                      row.work_titles.map((t) => (
                        <Badge key={t} variant="secondary" className="text-xs">
                          {t}
                        </Badge>
                      ))
                    ) : (
                      <span className="text-xs text-muted-foreground/70">
                        Project-wide (all tracks)
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  <span className="text-sm text-muted-foreground">$</span>
                  <Input
                    type="number"
                    min="0"
                    step="0.01"
                    value={row.amount}
                    onChange={(e) => updateAmount(row.id, e.target.value)}
                    className="w-28 h-8"
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-destructive hover:text-destructive"
                    onClick={() => removeRow(row.id)}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </Button>
                </div>
              </div>
            ))
          )}
        </div>

        {projectId && rows.length > 0 && (
          <div>
            <Button variant="ghost" size="sm" onClick={() => setAddOpen(true)}>
              <Plus className="w-3.5 h-3.5 mr-1.5" />
              Add expense
            </Button>
          </div>
        )}

        <div className="flex items-center justify-between text-sm border-t border-border pt-3">
          <span className="text-muted-foreground">Total expenses to apply</span>
          <span className="font-semibold">{formatCurrency(total)}</span>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onCancel} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button onClick={() => onConfirm(rows)} disabled={isSubmitting}>
            {isSubmitting && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            Confirm &amp; finalize
          </Button>
        </DialogFooter>
      </DialogContent>

      {projectId && (
        <ExpenseFormDialog
          open={addOpen}
          onOpenChange={setAddOpen}
          projectId={projectId}
          onSaved={handleExpenseAdded}
        />
      )}
    </Dialog>
  );
};

export default ExpenseReviewDialog;
