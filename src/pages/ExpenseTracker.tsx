import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { PageHeader } from "@/components/layout/PageHeader";
import { ArrowLeft, BookOpen, Plus, Receipt, ChevronRight, Loader2, Download, FileText, FileSpreadsheet } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid } from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import {
  useExpenseSummary,
  useExportExpenses,
  EXPENSE_CATEGORIES,
  EXPENSE_CATEGORY_LABELS,
  type ExpenseSummaryRow,
  type ExportFormat,
} from "@/hooks/useProjectExpenses";
import ExpenseFormDialog from "@/components/project/ExpenseFormDialog";
import { useSmartBack } from "@/hooks/useSmartBack";

const formatCurrency = (n: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(n || 0);

const trendChartConfig: ChartConfig = {
  total: {
    label: "Spent",
    color: "hsl(var(--primary))",
  },
};

type Granularity = "week" | "month" | "year";

// ISO-ish week number for a date (good enough for grouping/labelling).
function isoWeek(d: Date): { year: number; week: number } {
  const date = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()));
  const dayNum = (date.getUTCDay() + 6) % 7; // Mon=0
  date.setUTCDate(date.getUTCDate() - dayNum + 3); // nearest Thursday
  const firstThursday = new Date(Date.UTC(date.getUTCFullYear(), 0, 4));
  const week =
    1 +
    Math.round(
      ((date.getTime() - firstThursday.getTime()) / 86400000 - 3 + ((firstThursday.getUTCDay() + 6) % 7)) / 7,
    );
  return { year: date.getUTCFullYear(), week };
}

function bucketKeyAndLabel(dateStr: string, g: Granularity): { key: string; label: string } {
  const d = new Date(dateStr);
  if (g === "year") {
    const y = d.getFullYear();
    return { key: String(y), label: String(y) };
  }
  if (g === "month") {
    const y = d.getFullYear();
    const m = d.getMonth();
    const key = `${y}-${String(m + 1).padStart(2, "0")}`;
    const label = d.toLocaleString("en-US", { month: "short" }) + ` ${y}`;
    return { key, label };
  }
  const { year, week } = isoWeek(d);
  const key = `${year}-W${String(week).padStart(2, "0")}`;
  return { key, label: `W${week} ${year}` };
}

// "all" is the sentinel for an inactive filter. Uncategorized expenses count as
// "other", matching how byCategory buckets them.
export function filterExpenseRows(
  rows: ExpenseSummaryRow[],
  projectFilter: string,
  categoryFilter: string,
): ExpenseSummaryRow[] {
  return rows.filter(
    (r) =>
      (projectFilter === "all" || r.project_id === projectFilter) &&
      (categoryFilter === "all" || (r.category ?? "other") === categoryFilter),
  );
}

const ExpenseTracker = () => {
  const navigate = useNavigate();
  const goBack = useSmartBack("/tools");
  const { data: expenses, isLoading, isError } = useExpenseSummary();
  const [granularity, setGranularity] = useState<Granularity>("month");
  const [addOpen, setAddOpen] = useState(false);
  const [projectFilter, setProjectFilter] = useState<string>("all");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const filtersActive = projectFilter !== "all" || categoryFilter !== "all";
  const exportExpenses = useExportExpenses();

  const handleExport = (format: ExportFormat) => {
    exportExpenses.mutate({ format, projectId: projectFilter, category: categoryFilter });
  };

  const rows: ExpenseSummaryRow[] = useMemo(() => expenses ?? [], [expenses]);

  // Options come from unfiltered rows so they don't vanish while a filter is active,
  // and only projects that actually have expenses are listed.
  const projectOptions = useMemo(() => {
    const map = new Map<string, string>();
    for (const r of rows) {
      if (!map.has(r.project_id)) map.set(r.project_id, r.project_name ?? "Untitled project");
    }
    return [...map.entries()]
      .map(([id, name]) => ({ id, name }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [rows]);

  const filteredRows = useMemo(
    () => filterExpenseRows(rows, projectFilter, categoryFilter),
    [rows, projectFilter, categoryFilter],
  );

  const clearFilters = () => {
    setProjectFilter("all");
    setCategoryFilter("all");
  };

  const grandTotal = useMemo(
    () => filteredRows.reduce((s, r) => s + (r.amount || 0), 0),
    [filteredRows],
  );

  const byCategory = useMemo(() => {
    const acc: Record<string, number> = {};
    for (const r of filteredRows) {
      const key = r.category ?? "other";
      acc[key] = (acc[key] || 0) + (r.amount || 0);
    }
    return Object.entries(acc)
      .map(([key, value]) => ({ key, label: EXPENSE_CATEGORY_LABELS[key] ?? key, value }))
      .sort((a, b) => b.value - a.value);
  }, [filteredRows]);

  const trend = useMemo(() => {
    const acc: Record<string, { label: string; total: number }> = {};
    for (const r of filteredRows) {
      if (!r.incurred_on) continue; // undated expenses are excluded from the trend
      const { key, label } = bucketKeyAndLabel(r.incurred_on, granularity);
      if (!acc[key]) acc[key] = { label, total: 0 };
      acc[key].total += r.amount || 0;
    }
    return Object.entries(acc)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([, v]) => v);
  }, [filteredRows, granularity]);

  const undatedCount = useMemo(() => filteredRows.filter((r) => !r.incurred_on).length, [filteredRows]);

  // Group by artist → project for the rollup list.
  const grouped = useMemo(() => {
    const byArtist: Record<
      string,
      { artistName: string; total: number; projects: Record<string, { name: string; total: number }> }
    > = {};
    for (const r of filteredRows) {
      const aid = r.artist_id ?? "unknown";
      if (!byArtist[aid]) {
        byArtist[aid] = { artistName: r.artist_name ?? "Unknown artist", total: 0, projects: {} };
      }
      byArtist[aid].total += r.amount || 0;
      const pid = r.project_id;
      if (!byArtist[aid].projects[pid]) {
        byArtist[aid].projects[pid] = { name: r.project_name ?? "Untitled project", total: 0 };
      }
      byArtist[aid].projects[pid].total += r.amount || 0;
    }
    return Object.values(byArtist).sort((a, b) => b.total - a.total);
  }, [filteredRows]);

  const maxCategory = byCategory[0]?.value ?? 0;

  return (
    <div className="min-h-screen bg-background">
      <PageHeader
        showBack={false}
        actions={
          <>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate("/docs")}
              title="Documentation"
              className="text-muted-foreground hover:text-foreground"
            >
              <BookOpen className="w-4 h-4" />
            </Button>
            <Button variant="outline" className="hidden md:inline-flex" onClick={goBack}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
          </>
        }
      />

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        <div className="mb-8 flex items-start justify-between gap-4">
          <div>
            <h2 className="text-3xl font-bold text-foreground mb-2">Expense Tracker</h2>
            <p className="text-muted-foreground">
              Track project expenses across your portfolio. Net royalty calculations in OneClick
              deduct these from each track's earnings.
            </p>
          </div>
          <Button onClick={() => setAddOpen(true)} className="shrink-0">
            <Plus className="w-4 h-4 mr-2" /> Add Expense
          </Button>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
          </div>
        ) : isError ? (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <Receipt className="w-10 h-10 text-destructive/40 mb-3" />
            <p className="text-sm text-muted-foreground">Failed to load expenses</p>
            <p className="text-xs text-muted-foreground/60 mt-1">Please try refreshing the page</p>
          </div>
        ) : rows.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <Receipt className="w-10 h-10 text-muted-foreground/40 mb-3" />
            <p className="text-sm text-muted-foreground">No expenses tracked yet</p>
            <p className="text-xs text-muted-foreground/60 mt-1 mb-4">
              Add your first expense, or record them from a project's Expenses tab.
            </p>
            <Button onClick={() => setAddOpen(true)}>
              <Plus className="w-4 h-4 mr-2" /> Add Expense
            </Button>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Filters — scope every section below */}
            <div className="flex flex-wrap items-center gap-2">
              <Select value={projectFilter} onValueChange={setProjectFilter}>
                <SelectTrigger className="w-full sm:w-56">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All projects</SelectItem>
                  {projectOptions.map((p) => (
                    <SelectItem key={p.id} value={p.id}>
                      {p.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select value={categoryFilter} onValueChange={setCategoryFilter}>
                <SelectTrigger className="w-full sm:w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All categories</SelectItem>
                  {EXPENSE_CATEGORIES.map((c) => (
                    <SelectItem key={c.value} value={c.value}>
                      {c.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {filtersActive && (
                <Button variant="ghost" size="sm" onClick={clearFilters}>
                  Clear filters
                </Button>
              )}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    className="ml-auto"
                    disabled={filteredRows.length === 0 || exportExpenses.isPending}
                  >
                    {exportExpenses.isPending ? (
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <Download className="w-4 h-4 mr-2" />
                    )}
                    Export
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => handleExport("pdf")}>
                    <FileText className="w-4 h-4 mr-2" /> PDF
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => handleExport("xlsx")}>
                    <FileSpreadsheet className="w-4 h-4 mr-2" /> Excel (.xlsx)
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>

            {filteredRows.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <Receipt className="w-10 h-10 text-muted-foreground/40 mb-3" />
                <p className="text-sm text-muted-foreground">No expenses match your filters</p>
                <Button variant="outline" size="sm" className="mt-3" onClick={clearFilters}>
                  Clear filters
                </Button>
              </div>
            ) : (
              <>
            {/* Grand total */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Total tracked expenses
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-3xl font-bold text-foreground">{formatCurrency(grandTotal)}</p>
                <p className="text-xs text-muted-foreground mt-1">
                  {filteredRows.length} expense{filteredRows.length === 1 ? "" : "s"} across{" "}
                  {grouped.length} artist{grouped.length === 1 ? "" : "s"}
                </p>
              </CardContent>
            </Card>

            <div className="grid gap-6 lg:grid-cols-2">
              {/* Trend over time */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
                  <CardTitle className="text-base">Spend over time</CardTitle>
                  <ToggleGroup
                    type="single"
                    value={granularity}
                    onValueChange={(v) => v && setGranularity(v as Granularity)}
                    size="sm"
                  >
                    <ToggleGroupItem value="week" className="text-xs px-2">Week</ToggleGroupItem>
                    <ToggleGroupItem value="month" className="text-xs px-2">Month</ToggleGroupItem>
                    <ToggleGroupItem value="year" className="text-xs px-2">Year</ToggleGroupItem>
                  </ToggleGroup>
                </CardHeader>
                <CardContent>
                  {trend.length === 0 ? (
                    <p className="text-sm text-muted-foreground py-10 text-center">
                      No dated expenses to chart yet.
                    </p>
                  ) : (
                    <ChartContainer config={trendChartConfig} className="h-64 w-full">
                      <BarChart data={trend} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" vertical={false} />
                        <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                        <YAxis
                          tick={{ fontSize: 11 }}
                          tickFormatter={(v) => `$${Number(v) >= 1000 ? `${(Number(v) / 1000).toFixed(0)}k` : v}`}
                        />
                        <ChartTooltip
                          content={
                            <ChartTooltipContent
                              formatter={(value) => [formatCurrency(Number(value)), "Spent"]}
                            />
                          }
                        />
                        <Bar dataKey="total" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ChartContainer>
                  )}
                  {undatedCount > 0 && (
                    <p className="text-xs text-muted-foreground/70 mt-2">
                      {undatedCount} undated expense{undatedCount === 1 ? "" : "s"} not shown in the trend.
                    </p>
                  )}
                </CardContent>
              </Card>

              {/* Per-category breakdown */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">By category</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {byCategory.map((c) => (
                    <div key={c.key}>
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span className="text-foreground">{c.label}</span>
                        <span className="font-medium">{formatCurrency(c.value)}</span>
                      </div>
                      <div className="h-2 rounded-full bg-muted overflow-hidden">
                        <div
                          className="h-full bg-primary rounded-full"
                          style={{ width: `${maxCategory > 0 ? (c.value / maxCategory) * 100 : 0}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>

            {/* Per-project totals grouped by artist */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">By project</CardTitle>
              </CardHeader>
              <CardContent className="space-y-5">
                {grouped.map((artist) => (
                  <div key={artist.artistName}>
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-sm font-semibold text-foreground">{artist.artistName}</h4>
                      <span className="text-xs text-muted-foreground">{formatCurrency(artist.total)}</span>
                    </div>
                    <div className="grid gap-2">
                      {Object.entries(artist.projects)
                        .sort(([, a], [, b]) => b.total - a.total)
                        .map(([projectId, p]) => (
                          <button
                            key={projectId}
                            onClick={() => navigate(`/projects/${projectId}?tab=expenses`)}
                            className="flex items-center justify-between rounded-md border border-border px-3 py-2 text-left hover:bg-muted/50 transition-colors"
                          >
                            <span className="text-sm text-foreground truncate">{p.name}</span>
                            <span className="flex items-center gap-2 shrink-0">
                              <span className="text-sm font-medium">{formatCurrency(p.total)}</span>
                              <ChevronRight className="w-4 h-4 text-muted-foreground" />
                            </span>
                          </button>
                        ))}
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
              </>
            )}
          </div>
        )}
      </main>

      <ExpenseFormDialog open={addOpen} onOpenChange={setAddOpen} />
    </div>
  );
};

export default ExpenseTracker;
