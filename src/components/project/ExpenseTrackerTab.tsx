import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Loader2, Receipt, Plus, Pencil, Trash2 } from "lucide-react";
import { useWorksByProject, type Work } from "@/hooks/useRegistry";
import {
  useProjectExpenses,
  useDeleteProjectExpense,
  EXPENSE_CATEGORY_LABELS,
  type ProjectExpense,
} from "@/hooks/useProjectExpenses";
import ExpenseFormDialog from "./ExpenseFormDialog";

interface ExpenseTrackerTabProps {
  projectId: string;
  userRole: string | null;
}

const canEdit = (role: string | null) =>
  role === "owner" || role === "admin" || role === "editor";

const formatCurrency = (n: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(n || 0);

export default function ExpenseTrackerTab({ projectId, userRole }: ExpenseTrackerTabProps) {
  const { data: expenses, isLoading, isError } = useProjectExpenses(projectId);
  const { data: works } = useWorksByProject(projectId);
  const deleteExpense = useDeleteProjectExpense();

  const [dialogOpen, setDialogOpen] = useState(false);
  const [editing, setEditing] = useState<ProjectExpense | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ProjectExpense | null>(null);

  const editable = canEdit(userRole);
  const worksById = new Map((works ?? []).map((w: Work) => [w.id, w.title]));

  const openAdd = () => {
    setEditing(null);
    setDialogOpen(true);
  };

  const openEdit = (expense: ProjectExpense) => {
    setEditing(expense);
    setDialogOpen(true);
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    await deleteExpense.mutateAsync({ projectId, expenseId: deleteTarget.id });
    setDeleteTarget(null);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <Receipt className="w-10 h-10 text-destructive/40 mb-3" />
        <p className="text-sm text-muted-foreground">Failed to load expenses</p>
        <p className="text-xs text-muted-foreground/60 mt-1">Please try refreshing the page</p>
      </div>
    );
  }

  const items = expenses ?? [];
  const total = items.reduce((sum, e) => sum + (e.amount || 0), 0);
  const isEmpty = items.length === 0;

  return (
    <>
      {isEmpty ? (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <Receipt className="w-10 h-10 text-muted-foreground/40 mb-3" />
          <p className="text-sm text-muted-foreground">No expenses yet</p>
          <p className="text-xs text-muted-foreground/60 mt-1">
            Track studio, marketing, and other costs so OneClick can calculate net royalties
          </p>
          {editable && (
            <div className="mt-4">
              <Button onClick={openAdd}>
                <Plus className="w-4 h-4 mr-2" /> Add Expense
              </Button>
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-4">
            <div className="text-sm text-muted-foreground">
              Total expenses:{" "}
              <span className="font-semibold text-foreground">{formatCurrency(total)}</span>
            </div>
            {editable && (
              <Button size="sm" onClick={openAdd}>
                <Plus className="w-4 h-4 mr-2" /> Add Expense
              </Button>
            )}
          </div>

          <div className="grid gap-3">
            {items.map((expense) => (
              <Card key={expense.id} className="p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-medium text-foreground">
                        {expense.description}
                      </span>
                      {expense.category && (
                        <Badge variant="outline" className="text-xs shrink-0">
                          {EXPENSE_CATEGORY_LABELS[expense.category] ?? expense.category}
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-2 flex-wrap mt-1.5">
                      {expense.incurred_on && (
                        <span className="text-xs text-muted-foreground">{expense.incurred_on}</span>
                      )}
                      {expense.work_ids.length === 0 ? (
                        <span className="text-xs text-muted-foreground/70">
                          Project-wide (all tracks)
                        </span>
                      ) : (
                        expense.work_ids.map((wid) => (
                          <Badge key={wid} variant="secondary" className="text-xs">
                            {worksById.get(wid) ?? "Track"}
                          </Badge>
                        ))
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-3 shrink-0">
                    <span className="text-sm font-semibold text-foreground">
                      {formatCurrency(expense.amount)}
                    </span>
                    {editable && (
                      <div className="flex items-center gap-1">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          onClick={() => openEdit(expense)}
                        >
                          <Pencil className="w-3.5 h-3.5" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-destructive hover:text-destructive"
                          onClick={() => setDeleteTarget(expense)}
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {editable && (
        <ExpenseFormDialog
          open={dialogOpen}
          onOpenChange={setDialogOpen}
          projectId={projectId}
          editing={editing}
        />
      )}

      <AlertDialog open={!!deleteTarget} onOpenChange={(open) => !open && setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Remove this expense?</AlertDialogTitle>
            <AlertDialogDescription>
              "{deleteTarget?.description}" ({formatCurrency(deleteTarget?.amount ?? 0)}) will be
              permanently removed. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Remove
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
