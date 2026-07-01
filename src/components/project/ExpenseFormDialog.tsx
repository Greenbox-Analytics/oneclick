import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Loader2 } from "lucide-react";
import { useWorksByProject, type Work } from "@/hooks/useRegistry";
import { useProjectsList } from "@/hooks/useProjectsList";
import {
  useCreateProjectExpense,
  useUpdateProjectExpense,
  EXPENSE_CATEGORIES,
  type ProjectExpense,
  type ExpenseCategory,
} from "@/hooks/useProjectExpenses";

interface ExpenseFormDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Fixed project (per-project tab). Omit to show a project picker (standalone tool). */
  projectId?: string;
  /** Pass an existing expense to edit; omit to add. */
  editing?: ProjectExpense | null;
  /** Called after a successful save; receives the created/updated expense. */
  onSaved?: (expense?: ProjectExpense) => void;
}

interface FormState {
  description: string;
  amount: string;
  category: ExpenseCategory | "";
  incurred_on: string;
  work_ids: string[];
}

const EMPTY_FORM: FormState = {
  description: "",
  amount: "",
  category: "",
  incurred_on: "",
  work_ids: [],
};

export default function ExpenseFormDialog({
  open,
  onOpenChange,
  projectId,
  editing,
  onSaved,
}: ExpenseFormDialogProps) {
  const showProjectPicker = !projectId;
  const [pickedProjectId, setPickedProjectId] = useState<string>(projectId ?? "");
  const [form, setForm] = useState<FormState>(EMPTY_FORM);

  const { projects } = useProjectsList();
  const createExpense = useCreateProjectExpense();
  const updateExpense = useUpdateProjectExpense();

  const activeProjectId = projectId ?? pickedProjectId;
  const { data: works } = useWorksByProject(activeProjectId || undefined);

  // Seed the form whenever the dialog opens (or the target expense changes).
  useEffect(() => {
    if (!open) return;
    if (editing) {
      setPickedProjectId(editing.project_id);
      setForm({
        description: editing.description,
        amount: String(editing.amount),
        category: editing.category ?? "",
        incurred_on: editing.incurred_on ?? "",
        work_ids: editing.work_ids ?? [],
      });
    } else {
      setPickedProjectId(projectId ?? "");
      setForm(EMPTY_FORM);
    }
  }, [open, editing, projectId]);

  const toggleWork = (workId: string) => {
    setForm((f) => ({
      ...f,
      work_ids: f.work_ids.includes(workId)
        ? f.work_ids.filter((id) => id !== workId)
        : [...f.work_ids, workId],
    }));
  };

  const saving = createExpense.isPending || updateExpense.isPending;
  const amountValid = !isNaN(parseFloat(form.amount)) && parseFloat(form.amount) >= 0;
  const canSubmit =
    !!activeProjectId && form.description.trim().length > 0 && amountValid && !saving;

  const handleSave = async () => {
    if (!canSubmit) return;
    const payload = {
      description: form.description.trim(),
      amount: parseFloat(form.amount),
      category: form.category || null,
      incurred_on: form.incurred_on || null,
      work_ids: form.work_ids,
    };
    let saved: ProjectExpense | undefined;
    if (editing) {
      const res = await updateExpense.mutateAsync({ projectId: activeProjectId, expenseId: editing.id, ...payload });
      saved = res?.expense;
    } else {
      const res = await createExpense.mutateAsync({ projectId: activeProjectId, ...payload });
      saved = res?.expense;
    }
    onOpenChange(false);
    onSaved?.(saved);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{editing ? "Edit expense" : "Add expense"}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-2">
          {showProjectPicker && (
            <div className="space-y-1.5">
              <Label>Project</Label>
              <Select
                value={pickedProjectId}
                onValueChange={(v) => setPickedProjectId(v)}
                disabled={!!editing}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select a project" />
                </SelectTrigger>
                <SelectContent>
                  {projects.map((p) => (
                    <SelectItem key={p.id} value={p.id}>
                      {p.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          <div className="space-y-1.5">
            <Label htmlFor="expense-description">Description</Label>
            <Input
              id="expense-description"
              value={form.description}
              onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))}
              placeholder="e.g. Studio time at XYZ"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label htmlFor="expense-amount">Amount (USD)</Label>
              <Input
                id="expense-amount"
                type="number"
                min="0"
                step="0.01"
                value={form.amount}
                onChange={(e) => setForm((f) => ({ ...f, amount: e.target.value }))}
                placeholder="0.00"
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="expense-date">Date</Label>
              <Input
                id="expense-date"
                type="date"
                value={form.incurred_on}
                onChange={(e) => setForm((f) => ({ ...f, incurred_on: e.target.value }))}
              />
            </div>
          </div>
          <div className="space-y-1.5">
            <Label>Category</Label>
            <Select
              value={form.category}
              onValueChange={(v) => setForm((f) => ({ ...f, category: v as ExpenseCategory }))}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a category" />
              </SelectTrigger>
              <SelectContent>
                {EXPENSE_CATEGORIES.map((c) => (
                  <SelectItem key={c.value} value={c.value}>
                    {c.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label>Linked tracks</Label>
            <p className="text-xs text-muted-foreground">
              Leave empty to apply this expense across all tracks (allocated by earnings).
            </p>
            <div className="max-h-40 overflow-y-auto rounded-md border border-border p-2 space-y-1.5">
              {!activeProjectId ? (
                <p className="text-xs text-muted-foreground py-2 text-center">
                  Select a project to see its tracks
                </p>
              ) : (works ?? []).length === 0 ? (
                <p className="text-xs text-muted-foreground py-2 text-center">
                  No tracks in this project yet
                </p>
              ) : (
                (works ?? []).map((w: Work) => (
                  <label
                    key={w.id}
                    className="flex items-center gap-2 text-sm cursor-pointer py-0.5"
                  >
                    <Checkbox
                      checked={form.work_ids.includes(w.id)}
                      onCheckedChange={() => toggleWork(w.id)}
                    />
                    <span className="truncate">{w.title}</span>
                  </label>
                ))
              )}
            </div>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={!canSubmit}>
            {saving && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            {editing ? "Save changes" : "Add expense"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
