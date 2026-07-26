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
import { useArtistsList } from "@/hooks/useArtistsList";
import {
  useCreateProjectExpense,
  useUpdateProjectExpense,
  EXPENSE_CATEGORIES,
  EXPENSE_CURRENCIES,
  type ProjectExpense,
  type ExpenseCategory,
  type ExpenseCurrency,
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
  currency: ExpenseCurrency;
  category: ExpenseCategory | "";
  incurred_on: string;
  work_ids: string[];
}

const EMPTY_FORM: FormState = {
  description: "",
  amount: "",
  currency: "USD",
  category: "",
  incurred_on: "",
  work_ids: [],
};

// Required-field marker, matching the Split Sheet form.
const Req = () => <span className="text-green-600">*</span>;

export default function ExpenseFormDialog({
  open,
  onOpenChange,
  projectId,
  editing,
  onSaved,
}: ExpenseFormDialogProps) {
  const showProjectPicker = !projectId;
  const [pickedArtistId, setPickedArtistId] = useState<string>("");
  const [pickedProjectId, setPickedProjectId] = useState<string>(projectId ?? "");
  const [form, setForm] = useState<FormState>(EMPTY_FORM);

  const { artists } = useArtistsList();
  const { projects } = useProjectsList(pickedArtistId ? [pickedArtistId] : undefined);
  const createExpense = useCreateProjectExpense();
  const updateExpense = useUpdateProjectExpense();

  const activeProjectId = projectId ?? pickedProjectId;
  const { data: works } = useWorksByProject(activeProjectId || undefined);

  // Seed the form whenever the dialog opens (or the target expense changes).
  useEffect(() => {
    if (!open) return;
    setPickedArtistId("");
    if (editing) {
      setPickedProjectId(editing.project_id);
      setForm({
        description: editing.description,
        amount: String(editing.amount),
        currency: editing.currency ?? "USD",
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
  // Fixed-project tab and edit mode imply the artist via the project.
  const artistOk = !showProjectPicker || !!editing || !!pickedArtistId;
  const canSubmit =
    artistOk && !!activeProjectId && amountValid && !!form.incurred_on && !!form.category && !saving;

  const handleSave = async () => {
    if (!canSubmit) return;
    const payload = {
      description: form.description.trim(),
      amount: parseFloat(form.amount),
      currency: form.currency,
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
          {showProjectPicker && !editing && (
            <div className="space-y-1.5">
              <Label>
                Artist <Req />
              </Label>
              <Select
                value={pickedArtistId}
                onValueChange={(v) => {
                  setPickedArtistId(v);
                  setPickedProjectId("");
                }}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select an artist" />
                </SelectTrigger>
                <SelectContent>
                  {artists.map((a) => (
                    <SelectItem key={a.id} value={a.id}>
                      {a.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          {showProjectPicker && (
            <div className="space-y-1.5">
              <Label>
                Project <Req />
              </Label>
              <Select
                value={pickedProjectId}
                onValueChange={(v) => setPickedProjectId(v)}
                disabled={!!editing || !pickedArtistId}
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
            <Label htmlFor="expense-description">Description (optional)</Label>
            <Input
              id="expense-description"
              value={form.description}
              onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))}
              placeholder="e.g. Studio time at XYZ"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label htmlFor="expense-amount">
                Amount <Req />
              </Label>
              <div className="flex gap-2">
                <Input
                  id="expense-amount"
                  type="number"
                  min="0"
                  step="0.01"
                  value={form.amount}
                  onChange={(e) => setForm((f) => ({ ...f, amount: e.target.value }))}
                  placeholder="0.00"
                  className="flex-1"
                />
                <Select
                  value={form.currency}
                  onValueChange={(v) => setForm((f) => ({ ...f, currency: v as ExpenseCurrency }))}
                >
                  <SelectTrigger className="w-24 shrink-0">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {EXPENSE_CURRENCIES.map((c) => (
                      <SelectItem key={c} value={c}>
                        {c}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="expense-date">
                Date <Req />
              </Label>
              <Input
                id="expense-date"
                type="date"
                value={form.incurred_on}
                onChange={(e) => setForm((f) => ({ ...f, incurred_on: e.target.value }))}
              />
            </div>
          </div>
          <div className="space-y-1.5">
            <Label>
              Category <Req />
            </Label>
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
            <Label>Linked works (optional)</Label>
            <p className="text-xs text-muted-foreground">
              Leave empty to apply this expense across all works (allocated by earnings).
            </p>
            <div className="max-h-40 overflow-y-auto rounded-md border border-border p-2 space-y-1.5">
              {!activeProjectId ? (
                <p className="text-xs text-muted-foreground py-2 text-center">
                  Select a project to see its works
                </p>
              ) : (works ?? []).length === 0 ? (
                <p className="text-xs text-muted-foreground py-2 text-center">
                  No works in this project yet
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
