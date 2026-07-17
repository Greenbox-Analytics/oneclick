import { useRef, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Music,
  FileText,
  Users,
  DollarSign,
  Download,
  CheckCircle2,
  Loader2,
  RefreshCw,
  Share2,
  HardDrive,
  MessageSquare,
  Wallet,
  AlertTriangle,
  ShieldCheck,
  Disc3,
  Search,
  ChevronDown,
  PieChart as PieChartIcon,
  BarChart3,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ExcelJS from "exceljs";
import { toPng } from "html-to-image";
import { toast } from "sonner";
import { useIntegrations } from "@/hooks/useIntegrations";
import { API_URL, apiFetch, getAuthHeaders } from "@/lib/apiFetch";
import EarningsBreakdown from "@/components/oneclick/EarningsBreakdown";
import "./royalty-results.css";

export interface RoyaltyPayment {
  song_title: string;
  party_name: string;
  role: string;
  royalty_type: string;
  percentage: number;
  total_royalty: number;
  amount_to_pay: number;
  terms?: string;
  basis?: string;
  gross_amount?: number;
  expenses_applied?: number;
  net_amount?: number;
}

export interface SplitFinding {
  party_name: string;
  royalty_type: string;
  extracted_percentage: number;
  extracted_basis?: string | null;
  verdict: "verified" | "mismatch" | "unverified";
  contract_percentage?: number | null;
  contract_basis?: string | null;
  contract_quote?: string;
  note?: string;
}
export interface SplitReview { overall: "verified" | "needs_review" | "unavailable"; checked: number; flagged: number; findings: SplitFinding[]; }

interface CalculationResult {
  status: string;
  total_payments: number;
  payments: RoyaltyPayment[];
  excel_file_url?: string;
  message: string;
  is_cached?: boolean;
  calculation_id?: string;
  expense_review_required?: boolean;
  review?: SplitReview | null;
}

interface CalculationResultsProps {
  showProgressModal: boolean;
  calculationProgress: number;
  calculationStage: string;
  calculationMessage: string;
  calculationResult: CalculationResult | null;
  isUploading: boolean;
  handleCalculateRoyalties: (forceRecalculate: boolean) => void;
  calculationId?: string | null;
}

const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(amount);
};

const isNet = (p: RoyaltyPayment) => p.basis === "net";
const grossOf = (p: RoyaltyPayment) => (p.gross_amount ?? p.total_royalty);
const netOf = (p: RoyaltyPayment) => (p.net_amount ?? p.gross_amount ?? p.total_royalty);

const BasisBadge = ({ payment }: { payment: RoyaltyPayment }) => (
  <Badge
    variant="outline"
    className={isNet(payment)
      ? "text-amber-600 border-amber-500/40 bg-amber-500/10"
      : "text-emerald-600 border-emerald-500/40 bg-emerald-500/10"}
  >
    {isNet(payment) ? "Net" : "Gross"}
  </Badge>
);

const normKey = (s: string) => (s || "").trim().toLowerCase();
export const findingForRow = (review: SplitReview | null | undefined, row: RoyaltyPayment): SplitFinding | undefined =>
  review?.findings?.find(
    (f) =>
      normKey(f.party_name) === normKey(row.party_name) &&
      normKey(f.royalty_type) === normKey(row.royalty_type) &&
      Math.abs(f.extracted_percentage - row.percentage) < 0.01,
  );

const VERDICT_STYLES = {
  verified: "text-emerald-600 border-emerald-500/40 bg-emerald-500/10",
  mismatch: "text-red-600 border-red-500/40 bg-red-500/10",
  unverified: "text-amber-600 border-amber-500/40 bg-amber-500/10",
} as const;
const VERDICT_LABELS = { verified: "Verified", mismatch: "Needs review", unverified: "Not verified" } as const;

// Currently not rendered — the per-row Verification column was removed by user
// request (2026-07-10). Kept (exported) so the quote-popover UI can be re-added
// without rebuilding it; findingForRow is its row-join helper.
export const VerdictPill = ({ finding }: { finding: SplitFinding }) => (
  <Popover>
    <PopoverTrigger asChild>
      <Badge variant="outline" className={`${VERDICT_STYLES[finding.verdict]} cursor-pointer whitespace-nowrap`}>
        {VERDICT_LABELS[finding.verdict]}
      </Badge>
    </PopoverTrigger>
    <PopoverContent className="w-96 text-sm space-y-2">
      {finding.note && <p>{finding.note}</p>}
      {finding.verdict === "mismatch" && finding.contract_percentage != null && (
        <p>
          The contract states <strong>{finding.contract_percentage}%</strong>; this calculation used{" "}
          <strong>{finding.extracted_percentage}%</strong>.
        </p>
      )}
      {finding.contract_quote && (
        <blockquote className="border-l-2 border-border pl-2 text-muted-foreground italic">
          "{finding.contract_quote}"
        </blockquote>
      )}
    </PopoverContent>
  </Popover>
);

// Currently not rendered — banner hidden at user request (2026-07-11); the
// verification pass itself still runs and its result is still cached/persisted.
export const ReviewBanner = ({ review }: { review?: SplitReview | null }) => {
  if (!review || review.overall === "unavailable") {
    return (
      <div className="verify-banner" data-tone="muted">
        Split verification wasn't available for this result.
      </div>
    );
  }
  if (review.overall === "verified") {
    // "extracted splits match" — deliberately NOT "all splits verified": the check only
    // covers splits that extraction found; an omitted party is invisible to it.
    return (
      <div className="verify-banner" data-tone="ok">
        <ShieldCheck />
        All {review.checked} extracted split{review.checked === 1 ? "" : "s"} match the contract.
      </div>
    );
  }
  return (
    <div className="verify-banner" data-tone="warn">
      <AlertTriangle />
      {review.flagged} of {review.checked} split{review.checked === 1 ? "" : "s"} couldn't be confirmed against the
      contract — double-check the percentages below against your contract before paying.
    </div>
  );
};

// ---------------------------------------------------------------------------
// Breakdown table filtering + sorting (pure helpers — exported for tests).
// These only hide/reorder rows; every displayed amount comes straight from the
// row's real fields. No money value is ever recomputed here.
// ---------------------------------------------------------------------------

export type BasisFilter = "all" | "net" | "gross";
export type RowStatus = "verified" | "needs_review" | "not_verified";
export type StatusFilter = "any" | RowStatus;
export type SortKey = "song" | "payee" | "role" | "type" | "gross" | "expenses" | "net" | "share" | "amount";
export type SortDir = "asc" | "desc";
export type SegmentSelection = { kind: "payee" | "song"; name: string } | null;

export const statusForRow = (review: SplitReview | null | undefined, row: RoyaltyPayment): RowStatus => {
  const verdict = findingForRow(review, row)?.verdict;
  if (verdict === "verified") return "verified";
  if (verdict === "mismatch") return "needs_review";
  return "not_verified";
};

export interface RowFilters {
  search: string;
  song: string; // "all" or an exact song title
  basis: BasisFilter;
  status: StatusFilter;
  segment: SegmentSelection;
}

export const filterRows = (
  rows: RoyaltyPayment[],
  filters: RowFilters,
  review: SplitReview | null | undefined,
): RoyaltyPayment[] => {
  const q = filters.search.trim().toLowerCase();
  return rows.filter((row) => {
    if (q) {
      const haystack = [row.party_name, row.song_title, row.role, row.royalty_type].join(" ").toLowerCase();
      if (!haystack.includes(q)) return false;
    }
    if (filters.song !== "all" && row.song_title !== filters.song) return false;
    if (filters.basis === "net" && !isNet(row)) return false;
    if (filters.basis === "gross" && isNet(row)) return false;
    if (filters.status !== "any" && statusForRow(review, row) !== filters.status) return false;
    if (filters.segment) {
      if (filters.segment.kind === "payee" && row.party_name !== filters.segment.name) return false;
      if (filters.segment.kind === "song" && row.song_title !== filters.segment.name) return false;
    }
    return true;
  });
};

const SORT_VALUE: Record<SortKey, (r: RoyaltyPayment) => string | number> = {
  song: (r) => r.song_title,
  payee: (r) => r.party_name,
  role: (r) => r.role,
  type: (r) => r.royalty_type,
  gross: (r) => grossOf(r),
  expenses: (r) => (isNet(r) ? (r.expenses_applied ?? 0) : 0),
  net: (r) => netOf(r),
  share: (r) => r.percentage,
  amount: (r) => r.amount_to_pay,
};

const NUMERIC_SORT_KEYS: SortKey[] = ["gross", "expenses", "net", "share", "amount"];

export const sortRows = (rows: RoyaltyPayment[], key: SortKey, dir: SortDir): RoyaltyPayment[] => {
  const getValue = SORT_VALUE[key];
  const sign = dir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const va = getValue(a);
    const vb = getValue(b);
    if (typeof va === "string" || typeof vb === "string") {
      return sign * String(va).localeCompare(String(vb), undefined, { sensitivity: "base" });
    }
    return sign * ((va as number) - (vb as number));
  });
};

// Distribution chart segments. Colors are assigned by sorted (desc) order and
// cycle through the palette; the same payee color map drives the legend,
// donut, bars, and the table's payee dots.
const CHART_PALETTE = [
  "hsl(150 55% 42%)",
  "hsl(150 42% 30%)",
  "hsl(150 60% 55%)",
  "hsl(168 45% 46%)",
  "hsl(168 48% 50%)",
  "hsl(150 45% 32%)",
  "hsl(150 35% 62%)",
  "hsl(168 40% 36%)",
];
const UNALLOCATED_LABEL = "Unallocated";
const UNALLOCATED_COLOR = "hsl(150 8% 30%)";

interface ChartSegment { name: string; value: number; color: string; muted: boolean; }

const BREAKDOWN_COLUMNS: { key: SortKey | null; label: string; numeric?: boolean }[] = [
  { key: "song", label: "Song" },
  { key: "payee", label: "Payee" },
  { key: "role", label: "Role" },
  { key: "type", label: "Type" },
  { key: null, label: "Basis" },
  { key: "gross", label: "Gross", numeric: true },
  { key: "expenses", label: "Expenses", numeric: true },
  { key: "net", label: "Net", numeric: true },
  { key: "share", label: "Share", numeric: true },
  { key: "amount", label: "Amount owed", numeric: true },
];

const CalculationResults = ({
  showProgressModal,
  calculationProgress,
  calculationStage,
  calculationMessage,
  calculationResult,
  isUploading,
  handleCalculateRoyalties,
  calculationId,
}: CalculationResultsProps) => {
  const chartContentRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  // Prefer the id carried on the result (covers cache hits) and fall back to
  // the id captured when the calculation was saved.
  const breakdownCalculationId = calculationResult?.calculation_id ?? calculationId ?? null;

  // Parties from this calc that are actually owed money (net revenue > 0).
  // Collaborators with net ≤ 0 are excluded so they're never pre-selected for
  // payment. Names map 1:1 to royalty-tracking payees by normalized name.
  const payableParties = Object.entries(
    (calculationResult?.payments ?? []).reduce((acc, p) => {
      acc[p.party_name] = (acc[p.party_name] ?? 0) + p.amount_to_pay;
      return acc;
    }, {} as Record<string, number>),
  )
    .filter(([, total]) => total > 0)
    .map(([name]) => name);

  const handlePayRoyalties = () => {
    navigate("/tools/oneclick", { state: { openPayoutForNames: payableParties } });
  };

  const { connections } = useIntegrations();
  const driveConnected = connections.some(c => c.provider === "google_drive" && c.status === "active");
  const slackConnected = connections.some(c => c.provider === "slack" && c.status === "active");
  const [sharing, setSharing] = useState(false);

  // Distribution panel state
  const [chartType, setChartType] = useState<"donut" | "bar">("donut");
  const [distMode, setDistMode] = useState<"payee" | "song">("payee");
  const [hoverSeg, setHoverSeg] = useState<string | null>(null);
  const [selectedSeg, setSelectedSeg] = useState<SegmentSelection>(null);

  // Breakdown table state
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("amount");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const handleShare = async (target: "drive" | "slack") => {
    if (!calculationResult) return;
    setSharing(true);
    try {
      await apiFetch(`${API_URL}/oneclick/share`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target,
          artist_name: "Artist",
          payments: calculationResult.payments,
          // Real dollars owed — total_payments on the result object is the
          // payment COUNT, which used to print as "Total Payments: $4.00".
          total_payments: calculationResult.payments.reduce((sum, p) => sum + p.amount_to_pay, 0),
          message: calculationResult.message,
        }),
      });
      toast.success(target === "drive" ? "Results saved to Google Drive" : "Results shared to Slack");
    } catch (err) {
      toast.error(`Share failed: ${(err as Error).message}`);
    } finally {
      setSharing(false);
    }
  };

  const handleExportCSV = () => {
    if (!calculationResult) return;
    const headers = ["Song Title", "Payee", "Role", "Royalty Type", "Basis", "Gross Revenue", "Expenses", "Net Revenue", "Share %", "Amount Owed"];
    const rows = calculationResult.payments.map(p => [
      p.song_title,
      p.party_name,
      p.role,
      p.royalty_type,
      isNet(p) ? "Net" : "Gross",
      grossOf(p),
      isNet(p) ? (p.expenses_applied ?? 0) : 0,
      netOf(p),
      `${p.percentage}%`,
      p.amount_to_pay
    ]);
    const csvContent = [headers.join(","), ...rows.map(row => row.map(cell => `"${cell}"`).join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", `royalty_breakdown.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const [exportingPdf, setExportingPdf] = useState(false);
  const handleExportPDF = async () => {
    if (!calculationResult) return;
    setExportingPdf(true);
    try {
      // Backend builds the PDF with the same generator used for Drive/Slack shares.
      const headers = await getAuthHeaders();
      const res = await fetch(`${API_URL}/oneclick/export-pdf`, {
        method: "POST",
        headers: { ...headers, "Content-Type": "application/json" },
        body: JSON.stringify({
          artist_name: "Artist",
          payments: calculationResult.payments,
          // Real dollars owed, not the payment count.
          total_payments: calculationResult.payments.reduce((sum, p) => sum + p.amount_to_pay, 0),
          message: calculationResult.message,
        }),
      });
      if (!res.ok) throw new Error(`Export failed: ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "royalty_breakdown.pdf";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch {
      toast.error("Failed to export PDF");
    } finally {
      setExportingPdf(false);
    }
  };

  const handleExportExcel = async () => {
    if (!calculationResult) return;
    const excelData = [
      ["Song Title", "Payee", "Role", "Royalty Type", "Basis", "Gross Revenue", "Expenses", "Net Revenue", "Share %", "Amount Owed"],
      ...calculationResult.payments.map(p => [
        p.song_title, p.party_name, p.role, p.royalty_type, isNet(p) ? "Net" : "Gross",
        grossOf(p), isNet(p) ? (p.expenses_applied ?? 0) : 0, netOf(p), p.percentage, p.amount_to_pay
      ])
    ];
    const wb = new ExcelJS.Workbook();
    const ws = wb.addWorksheet("Royalty Breakdown");
    excelData.forEach(row => ws.addRow(row));
    const buffer = await wb.xlsx.writeBuffer();
    const blob = new Blob([buffer], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "royalty_breakdown.xlsx";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDownloadChart = async () => {
    if (!chartContentRef.current || !calculationResult) return;

    try {
      const dataUrl = await toPng(chartContentRef.current, {
        quality: 1,
        pixelRatio: 2,
        backgroundColor: '#ffffff'
      });

      const link = document.createElement('a');
      link.download = `royalty-distribution.png`;
      link.href = dataUrl;
      link.click();
      toast.success("Chart downloaded successfully!");
    } catch (error) {
      console.error("Error downloading chart:", error);
      toast.error("Failed to download chart. Please try again.");
    }
  };

  const payments = calculationResult?.payments ?? [];
  const totalRevenue = Array.from(
    new Map(payments.map(p => [p.song_title, p.total_royalty])).values()
  ).reduce((sum, val) => sum + val, 0);
  const totalToPay = payments.reduce((sum, p) => sum + p.amount_to_pay, 0);

  // Distribution segments: sum of amount_to_pay grouped by payee or song,
  // sorted desc, with a muted "Unallocated" remainder when revenue exceeds
  // what's owed (same > 0.005 threshold as the old pie chart).
  const buildSegments = (mode: "payee" | "song"): ChartSegment[] => {
    const totals = payments.reduce((acc, p) => {
      const key = mode === "payee" ? p.party_name : p.song_title;
      acc[key] = (acc[key] ?? 0) + p.amount_to_pay;
      return acc;
    }, {} as Record<string, number>);
    const segments: ChartSegment[] = Object.entries(totals)
      .map(([name, value]) => ({ name, value, color: "", muted: false }))
      .sort((a, b) => b.value - a.value)
      .map((s, i) => ({ ...s, color: CHART_PALETTE[i % CHART_PALETTE.length] }));
    const allocated = segments.reduce((sum, s) => sum + s.value, 0);
    const unallocated = Math.max(0, totalRevenue - allocated);
    if (unallocated > 0.005) {
      segments.push({ name: UNALLOCATED_LABEL, value: unallocated, color: UNALLOCATED_COLOR, muted: true });
    }
    return segments;
  };

  const payeeSegments = buildSegments("payee");
  const segments = distMode === "payee" ? payeeSegments : buildSegments("song");
  const payeeColors: Record<string, string> = {};
  payeeSegments.forEach((s) => { if (!s.muted) payeeColors[s.name] = s.color; });

  const segTotal = segments.reduce((sum, s) => sum + s.value, 0);
  const maxSegValue = segments.reduce((max, s) => Math.max(max, s.value), 0);
  const pctOf = (value: number) => (segTotal > 0 ? ((value / segTotal) * 100).toFixed(1) : "0.0");
  // Highlight priority: hover wins, else a clicked (selected) segment. Non-
  // highlighted slices fade toward grey but keep a tint of their own color, so
  // the highlighted portion stands out while the rest of the distribution
  // remains readable.
  const highlightName = hoverSeg ?? (selectedSeg ? selectedSeg.name : null);
  const fadedColor = (c: string) => `color-mix(in srgb, ${c} 38%, hsl(var(--muted)))`;
  let gradientAcc = 0;
  const donutBackground = segTotal > 0
    ? `conic-gradient(${segments
        .map((s) => {
          const from = (gradientAcc / segTotal) * 360;
          gradientAcc += s.value;
          const to = (gradientAcc / segTotal) * 360;
          const color = highlightName && s.name !== highlightName ? fadedColor(s.color) : s.color;
          return `${color} ${from}deg ${to}deg`;
        })
        .join(", ")})`
    : "hsl(var(--muted))";
  const hoveredSegment = hoverSeg ? segments.find((s) => s.name === hoverSeg) : null;

  const switchDistMode = (mode: "payee" | "song") => {
    setDistMode(mode);
    setHoverSeg(null);
    setSelectedSeg(null);
  };

  const toggleSegment = (name: string) => {
    setSelectedSeg((current) =>
      current && current.kind === distMode && current.name === name ? null : { kind: distMode, name },
    );
  };

  // Table rows: filter + sort only — never mutates or recomputes amounts.
  const anyFilterActive = search.trim() !== "" || selectedSeg !== null;
  const filteredRows = filterRows(
    payments,
    { search, song: "all", basis: "all", status: "any", segment: selectedSeg },
    calculationResult?.review,
  );
  const visibleRows = sortRows(filteredRows, sortKey, sortDir);
  const filteredTotal = filteredRows.reduce((sum, p) => sum + p.amount_to_pay, 0);

  const clearFilters = () => {
    setSearch("");
    setSelectedSeg(null);
  };

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((dir) => (dir === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(NUMERIC_SORT_KEYS.includes(key) ? "desc" : "asc");
    }
  };

  const payRoyaltiesButton = (
    <Button size="sm" onClick={handlePayRoyalties} disabled={payableParties.length === 0}>
      <Wallet className="mr-2 h-4 w-4" /> Pay Royalties
    </Button>
  );

  return (
    <>
      {showProgressModal && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="text-center">Calculating Royalties</CardTitle>
            <CardDescription className="text-center">
              Please wait while we process your documents
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6 py-4">
            <div className="flex flex-col items-center justify-center">
              <div className="relative w-32 h-32">
                <svg className="w-32 h-32 transform -rotate-90">
                  <circle
                    cx="64"
                    cy="64"
                    r="56"
                    stroke="currentColor"
                    strokeWidth="8"
                    fill="none"
                    className="text-muted"
                  />
                  <circle
                    cx="64"
                    cy="64"
                    r="56"
                    stroke="currentColor"
                    strokeWidth="8"
                    fill="none"
                    strokeDasharray={`${2 * Math.PI * 56}`}
                    strokeDashoffset={`${2 * Math.PI * 56 * (1 - calculationProgress / 100)}`}
                    className="text-primary transition-all duration-500 ease-out"
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-2xl font-bold text-foreground">{Math.round(calculationProgress)}%</span>
                </div>
              </div>

              <p className="mt-4 text-sm font-medium text-center text-muted-foreground">
                {calculationMessage}
              </p>
            </div>

            <div className="space-y-3">
              {[
                { id: 'downloading', label: 'Downloading files', icon: FileText },
                { id: 'extracting', label: 'Extracting data', icon: FileText },
                { id: 'calculating', label: 'Calculating payments', icon: DollarSign },
              ].map((stageItem) => {
                const getStageStatus = (stageId: string) => {
                  if (calculationStage.includes('download')) {
                    if (stageId === 'downloading') return 'active';
                    return 'pending';
                  } else if (calculationStage.includes('extract') || calculationStage.includes('parties') || calculationStage.includes('works') || calculationStage.includes('royalty') || calculationStage.includes('summary')) {
                    if (stageId === 'downloading') return 'complete';
                    if (stageId === 'extracting') return 'active';
                    return 'pending';
                  } else if (calculationStage.includes('processing') || calculationStage.includes('calculating')) {
                    if (stageId === 'downloading' || stageId === 'extracting') return 'complete';
                    if (stageId === 'calculating') return 'active';
                    return 'pending';
                  }
                  return 'pending';
                };

                const status = getStageStatus(stageItem.id);
                const Icon = stageItem.icon;

                return (
                  <div key={stageItem.id} className="flex items-center gap-3">
                    <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center ${
                      status === 'complete'
                        ? 'bg-primary text-primary-foreground'
                        : status === 'active'
                        ? 'bg-primary/20 text-primary'
                        : 'bg-muted text-muted-foreground'
                    }`}>
                      {status === 'complete' ? (
                        <CheckCircle2 className="w-4 h-4" />
                      ) : status === 'active' ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Icon className="w-4 h-4" />
                      )}
                    </div>
                    <span className={`text-sm ${
                      status === 'complete' || status === 'active'
                        ? 'text-foreground font-medium'
                        : 'text-muted-foreground'
                    }`}>
                      {stageItem.label}
                    </span>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {calculationResult && (
        /* The whole results section breaks out of the narrow page container on
           large screens; .results-wrap re-caps content at 1280px. */
        <div className="results-page xl:relative xl:left-1/2 xl:right-1/2 xl:-mx-[50vw] xl:w-screen">
          <div className="results-wrap">
            <div className="res-head">
              <div>
                <h1>Royalty Calculation Results</h1>
                <p className="page-sub">{calculationResult.message}</p>
              </div>
              <div className="res-actions">
                <Button variant="outline" size="sm" onClick={() => handleCalculateRoyalties(true)} disabled={isUploading}>
                  <RefreshCw className={`w-4 h-4 mr-2 ${isUploading ? 'animate-spin' : ''}`} />
                  Recalculate
                </Button>
                {(driveConnected || slackConnected) && (
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button variant="outline" size="sm" disabled={sharing}>
                        <Share2 className="w-4 h-4 mr-2" />
                        Share
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-48 p-1">
                      {driveConnected && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="w-full justify-start"
                          onClick={() => handleShare("drive")}
                        >
                          <HardDrive className="w-4 h-4 mr-2" />
                          Save to Drive
                        </Button>
                      )}
                      {slackConnected && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="w-full justify-start"
                          onClick={() => handleShare("slack")}
                        >
                          <MessageSquare className="w-4 h-4 mr-2" />
                          Send to Slack
                        </Button>
                      )}
                    </PopoverContent>
                  </Popover>
                )}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="sm">
                      <Download className="mr-2 h-4 w-4" /> Export
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent>
                    <DropdownMenuItem onClick={handleExportCSV}>Export as CSV</DropdownMenuItem>
                    <DropdownMenuItem onClick={handleExportExcel}>Export as Excel</DropdownMenuItem>
                    <DropdownMenuItem onClick={handleExportPDF} disabled={exportingPdf}>
                      {exportingPdf ? "Exporting PDF…" : "Export as PDF"}
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>

            {/* Verification banner intentionally not rendered (user request,
               2026-07-11): the split-verification pass still runs, and its
               result still rides on the payload (cached + persisted via
               confirm) — it just isn't surfaced in this UI. ReviewBanner
               stays exported below for if/when it's wanted again. */}

            <div className="res-stats">
              <div className="panel res-stat">
                <div className="res-stat-top">
                  <span className="res-stat-label">Songs processed</span>
                  <Music />
                </div>
                <div className="res-stat-num">{new Set(calculationResult.payments.map(p => p.song_title)).size}</div>
              </div>
              <div className="panel res-stat">
                <div className="res-stat-top">
                  <span className="res-stat-label">Total payees</span>
                  <Users />
                </div>
                <div className="res-stat-num">{new Set(calculationResult.payments.map(p => p.party_name)).size}</div>
              </div>
              <div className="panel res-stat">
                <div className="res-stat-top">
                  <span className="res-stat-label">Total revenue</span>
                  <Disc3 />
                </div>
                <div className="res-stat-num">{formatCurrency(totalRevenue)}</div>
              </div>
              <div className="panel res-stat">
                <div className="res-stat-top">
                  <span className="res-stat-label">Total to pay</span>
                  <Wallet />
                </div>
                <div className="res-stat-num accent">{formatCurrency(totalToPay)}</div>
              </div>
            </div>

            <Tabs defaultValue="distribution" className="res-tabs">
              <TabsList className="grid grid-cols-2 w-full max-w-md">
                <TabsTrigger value="distribution">Royalty Distribution</TabsTrigger>
                <TabsTrigger value="breakdown">Earnings Breakdown</TabsTrigger>
              </TabsList>
              <TabsContent value="distribution" className="mt-4">
              {/* ---------------- Distribution ---------------- */}
              <div className="panel">
                <div className="panel-head">
                  <h3 className="panel-title">Distribution</h3>
                  <div className="seg-sm seg-icon">
                    <button
                      type="button"
                      data-active={chartType === "donut"}
                      onClick={() => setChartType("donut")}
                      aria-label="Donut chart"
                      title="Donut chart"
                    >
                      <PieChartIcon />
                    </button>
                    <button
                      type="button"
                      data-active={chartType === "bar"}
                      onClick={() => setChartType("bar")}
                      aria-label="Bar chart"
                      title="Bar chart"
                    >
                      <BarChart3 />
                    </button>
                    <button type="button" onClick={handleDownloadChart} aria-label="Download chart" title="Download chart">
                      <Download />
                    </button>
                  </div>
                </div>
                <div className="panel-body" ref={chartContentRef}>
                  <div className="seg-sm dist-mode">
                    <button type="button" data-active={distMode === "payee"} onClick={() => switchDistMode("payee")}>
                      By payee
                    </button>
                    <button type="button" data-active={distMode === "song"} onClick={() => switchDistMode("song")}>
                      By song
                    </button>
                  </div>

                  {chartType === "donut" ? (
                    <div className="dist-donut-layout">
                      <div className="donut-wrap">
                        <div className="donut" data-highlight={highlightName ? "true" : undefined} style={{ background: donutBackground }}>
                          <div className="donut-center">
                            {hoveredSegment ? (
                              <>
                                <span className="dc-label">{hoveredSegment.name}</span>
                                <span className="dc-value">{formatCurrency(hoveredSegment.value)}</span>
                                <span className="dc-label">{pctOf(hoveredSegment.value)}%</span>
                              </>
                            ) : (
                              <>
                                <span className="dc-label">Total revenue</span>
                                <span className="dc-value">{formatCurrency(totalRevenue)}</span>
                                <span className="dc-label">{payments.length} payment{payments.length === 1 ? "" : "s"}</span>
                              </>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="legend">
                        <div className="legend-head" aria-hidden="true">
                          <span />
                          <span>{distMode === "payee" ? "Payee" : "Song"}</span>
                          <span className="num">Share</span>
                          <span className="num">Payout</span>
                        </div>
                        {segments.map((s) => {
                          const rowContent = (
                            <>
                              <span className="legend-swatch" style={{ background: s.color }} />
                              <span className="legend-name">{s.name}</span>
                              <span className="legend-share">{pctOf(s.value)}%</span>
                              <span className="legend-amt">{formatCurrency(s.value)}</span>
                            </>
                          );
                          if (s.muted) {
                            return (
                              <div
                                key={s.name}
                                className="legend-row"
                                data-muted="true"
                                data-static="true"
                                data-hovered={hoverSeg === s.name ? "true" : undefined}
                                onMouseEnter={() => setHoverSeg(s.name)}
                                onMouseLeave={() => setHoverSeg(null)}
                              >
                                {rowContent}
                              </div>
                            );
                          }
                          return (
                            <button
                              key={s.name}
                              type="button"
                              className="legend-row"
                              data-dim={hoverSeg ? hoverSeg !== s.name : selectedSeg ? selectedSeg.name !== s.name : false}
                              data-hovered={hoverSeg === s.name ? "true" : undefined}
                              onMouseEnter={() => setHoverSeg(s.name)}
                              onMouseLeave={() => setHoverSeg(null)}
                              onClick={() => toggleSegment(s.name)}
                            >
                              {rowContent}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  ) : (
                    <div className="barchart">
                      {segments.map((s) => {
                        const width = maxSegValue > 0 ? (s.value / maxSegValue) * 100 : 0;
                        const rowContent = (
                          <>
                            <span className="bar-label">{s.name}</span>
                            <span className="bar-track">
                              <span className="bar-fill" style={{ width: `${width}%`, background: s.color }} />
                            </span>
                            <span className="bar-val">
                              {formatCurrency(s.value)}
                              <span className="bar-pct">{pctOf(s.value)}%</span>
                            </span>
                          </>
                        );
                        if (s.muted) {
                          return (
                            <div key={s.name} className="bar-row" data-muted="true" data-static="true">
                              {rowContent}
                            </div>
                          );
                        }
                        return (
                          <button key={s.name} type="button" className="bar-row" onClick={() => toggleSegment(s.name)}>
                            {rowContent}
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
              </TabsContent>
              <TabsContent value="breakdown" className="mt-4">
                {/* EarningsBreakdown renders its own titled Card, so it isn't
                   wrapped in another .panel (that would double-box it). */}
                <EarningsBreakdown calculationId={breakdownCalculationId} />
              </TabsContent>
            </Tabs>

            {/* ---------------- Royalty Breakdown (full width) ---------------- */}
            <div className="panel res-table-panel">
                <div className="panel-head">
                  <h3 className="panel-title">Royalty Breakdown</h3>
                  {payRoyaltiesButton}
                </div>

                <div className="bd-toolbar">
                  <div className="bd-search">
                    <Search />
                    <input
                      type="text"
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      placeholder="Search payee, song, role…"
                      aria-label="Search payments"
                    />
                  </div>
                  {anyFilterActive && (
                    <button type="button" className="bd-clear" onClick={clearFilters}>Clear</button>
                  )}
                  <span className="bd-count">{visibleRows.length} of {payments.length} rows</span>
                </div>

                {visibleRows.length === 0 ? (
                  <div className="bd-empty">
                    <Search />
                    <div>No payments match these filters.</div>
                  </div>
                ) : (
                  <>
                    {/* Desktop: table */}
                    <div className="bd-scroll hidden sm:block">
                      <table className="bd">
                        <thead>
                          <tr>
                            {BREAKDOWN_COLUMNS.map((col) =>
                              col.key ? (
                                <th
                                  key={col.label}
                                  className={`sortable${col.numeric ? " num" : ""}`}
                                  data-sort={sortKey === col.key ? sortDir : undefined}
                                  onClick={() => handleSort(col.key)}
                                >
                                  <span className="th-inner">
                                    {col.label}
                                    <ChevronDown className="sort-caret" />
                                  </span>
                                </th>
                              ) : (
                                <th key={col.label}>{col.label}</th>
                              ),
                            )}
                          </tr>
                        </thead>
                        <tbody>
                          {visibleRows.map((row, i) => (
                            <tr key={i}>
                              <td className="td-song">{row.song_title}</td>
                              <td className="td-payee">
                                <span className="payee-cell">
                                  <span
                                    className="payee-dot"
                                    style={{ background: payeeColors[row.party_name] ?? UNALLOCATED_COLOR }}
                                  />
                                  {row.party_name}
                                </span>
                              </td>
                              <td className="capitalize">{row.role}</td>
                              <td className="capitalize">{row.royalty_type}</td>
                              <td>
                                <span className={`pill ${isNet(row) ? "pill-net" : "pill-gross"}`}>
                                  {isNet(row) ? "Net" : "Gross"}
                                </span>
                              </td>
                              <td className="td-num">{formatCurrency(grossOf(row))}</td>
                              <td className="td-num">
                                {isNet(row) && (row.expenses_applied ?? 0) > 0 ? (
                                  <span className="td-neg">−{formatCurrency(row.expenses_applied ?? 0)}</span>
                                ) : (
                                  <span className="td-muted">—</span>
                                )}
                              </td>
                              <td className="td-num">{formatCurrency(netOf(row))}</td>
                              <td className="td-num">{row.percentage}%</td>
                              <td className="td-amt">{formatCurrency(row.amount_to_pay)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {/* Mobile: stacked cards */}
                    <div className="sm:hidden space-y-3 p-4">
                      {visibleRows.map((row, i) => (
                        <div key={i} className="rounded-lg border border-border p-3 space-y-2 bg-card">
                          <div className="flex items-start justify-between gap-2">
                            <div>
                              <p className="text-sm font-semibold text-foreground">{row.song_title}</p>
                              <p className="text-xs text-muted-foreground">{row.party_name} • <span className="capitalize">{row.role}</span></p>
                            </div>
                            <BasisBadge payment={row} />
                          </div>
                          <div className="grid grid-cols-2 gap-2 text-xs">
                            <div>
                              <p className="text-muted-foreground">Royalty type</p>
                              <p className="font-medium capitalize">{row.royalty_type}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Share</p>
                              <p className="font-medium">{row.percentage}%</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Gross Revenue</p>
                              <p className="font-medium">{formatCurrency(grossOf(row))}</p>
                            </div>
                            {isNet(row) && (
                              <div>
                                <p className="text-muted-foreground">Expenses</p>
                                <p className="font-medium">-{formatCurrency(row.expenses_applied ?? 0)}</p>
                              </div>
                            )}
                            <div>
                              <p className="text-muted-foreground">Net Revenue</p>
                              <p className="font-medium">{formatCurrency(netOf(row))}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Amount Owed</p>
                              <p className="font-semibold text-foreground">{formatCurrency(row.amount_to_pay)}</p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}

                <div className="bd-foot">
                  <div className="bd-foot-total">
                    {anyFilterActive ? "Filtered total" : "Total owed"}
                    <b>{formatCurrency(anyFilterActive ? filteredTotal : totalToPay)}</b>
                  </div>
                </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default CalculationResults;
