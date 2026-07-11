import { useRef, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Music, FileText, Users, DollarSign, Download, CheckCircle2, Loader2, RefreshCw, Share2, HardDrive, MessageSquare, Wallet, AlertTriangle, ShieldCheck } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts";
import type { PieLabelRenderProps } from "recharts";
import ExcelJS from "exceljs";
import { toPng } from "html-to-image";
import { toast } from "sonner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useIntegrations } from "@/hooks/useIntegrations";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import EarningsBreakdown from "@/components/oneclick/EarningsBreakdown";

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

export const ReviewBanner = ({ review }: { review?: SplitReview | null }) => {
  if (!review || review.overall === "unavailable") {
    return <p className="text-sm text-muted-foreground">Split verification wasn't available for this result.</p>;
  }
  if (review.overall === "verified") {
    // "extracted splits match" — deliberately NOT "all splits verified": the check only
    // covers splits that extraction found; an omitted party is invisible to it.
    return (
      <div className="flex items-center gap-2 rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-600">
        <ShieldCheck className="h-4 w-4 shrink-0" />
        All {review.checked} extracted split{review.checked === 1 ? "" : "s"} match the contract.
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-600">
      <AlertTriangle className="h-4 w-4 shrink-0" />
      {review.flagged} of {review.checked} split{review.checked === 1 ? "" : "s"} couldn't be confirmed against the
      contract — double-check the percentages below against your contract before paying.
    </div>
  );
};

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
          total_payments: calculationResult.total_payments,
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

  const UNALLOCATED_LABEL = "Unallocated";
  const wrapLabel = (text: string, max = 28): string[] => {
    if (text.length <= max) return [text];
    let splitAt = text.lastIndexOf(" ", max);
    if (splitAt <= 0) splitAt = text.indexOf(" ", max);
    if (splitAt <= 0) splitAt = Math.floor(text.length / 2);
    return [text.slice(0, splitAt).trim(), text.slice(splitAt).trim()];
  };
  const renderPieLabel = (props: PieLabelRenderProps) => {
    const { cx, cy, midAngle, outerRadius, percent, value, name, fill } = props;
    const cxNum = Number(cx);
    const cyNum = Number(cy);
    const outerRadiusNum = Number(outerRadius);
    const midAngleNum = Number(midAngle);
    const fullLabel = `${name} (${((percent ?? 0) * 100).toFixed(1)}%): ${formatCurrency(Number(value))}`;
    const lines = wrapLabel(fullLabel);
    const RADIAN = Math.PI / 180;
    const radius = outerRadiusNum + 24;
    const x = cxNum + radius * Math.cos(-midAngleNum * RADIAN);
    const y = cyNum + radius * Math.sin(-midAngleNum * RADIAN);
    const textAnchor = x > cxNum ? "start" : "end";
    return (
      <text
        x={x}
        y={y}
        textAnchor={textAnchor}
        dominantBaseline="central"
        fill={fill}
        style={{ fontSize: 11 }}
      >
        {lines.map((line, i) => (
          <tspan key={i} x={x} dy={i === 0 ? 0 : 13}>{line}</tspan>
        ))}
      </text>
    );
  };
  const payments = calculationResult?.payments ?? [];
  const totalRevenue = Array.from(
    new Map(payments.map(p => [p.song_title, p.total_royalty])).values()
  ).reduce((sum, val) => sum + val, 0);
  const payeeTotals = Object.values(
    payments.reduce((acc, curr) => {
      if (!acc[curr.party_name]) acc[curr.party_name] = { name: curr.party_name, value: 0 };
      acc[curr.party_name].value += curr.amount_to_pay;
      return acc;
    }, {} as Record<string, { name: string; value: number }>)
  ).sort((a, b) => b.value - a.value);
  const allocated = payeeTotals.reduce((sum, p) => sum + p.value, 0);
  const unallocated = Math.max(0, totalRevenue - allocated);
  const pieData = unallocated > 0.005
    ? [...payeeTotals, { name: UNALLOCATED_LABEL, value: unallocated }]
    : payeeTotals;

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
        <div className="mt-8 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
            <div>
              <h2 className="text-2xl font-bold text-foreground">Royalty Calculation Results</h2>
              <p className="text-muted-foreground">{calculationResult.message}</p>
            </div>

            <div className="flex flex-wrap gap-2">
                <Button variant="outline" onClick={() => handleCalculateRoyalties(true)} disabled={isUploading}>
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
            </div>
          </div>

          <ReviewBanner review={calculationResult.review} />

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Songs Processed</CardTitle>
                <Music className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent><div className="text-2xl font-bold text-foreground">{new Set(calculationResult.payments.map(p => p.song_title)).size}</div></CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Payees</CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent><div className="text-2xl font-bold text-foreground">{new Set(calculationResult.payments.map(p => p.party_name)).size}</div></CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Song(s) Revenue</CardTitle>
                <DollarSign className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-foreground">
                  {formatCurrency(totalRevenue)}
                </div>
              </CardContent>
            </Card>
          </div>

          <Tabs defaultValue="distribution" className="w-full">
            <TabsList className="grid grid-cols-2 w-full max-w-md">
              <TabsTrigger value="distribution">Royalty Distribution</TabsTrigger>
              <TabsTrigger value="breakdown">Earnings Breakdown</TabsTrigger>
            </TabsList>

            <TabsContent value="distribution" className="mt-4">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>Royalty Distribution</CardTitle>
                    <Button variant="outline" size="sm" onClick={handleDownloadChart}>
                      <Download className="mr-2 h-4 w-4" />
                      Download Chart
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div ref={chartContentRef} className="h-[450px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={pieData}
                          cx="50%"
                          cy="50%"
                          outerRadius={130}
                          dataKey="value"
                          nameKey="name"
                          label={renderPieLabel}
                          labelLine={true}
                        >
                          {pieData.map((entry, index, arr) => {
                            const isUnallocated = entry.name === UNALLOCATED_LABEL;
                            const fill = isUnallocated
                              ? "hsl(150, 8%, 22%)"
                              : `hsl(150, ${50 + (index / (arr.length || 1)) * 10}%, ${25 + (index / (arr.length || 1)) * 40}%)`;
                            return <Cell key={`cell-${index}`} fill={fill} />;
                          })}
                        </Pie>
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="breakdown" className="mt-4">
              <EarningsBreakdown calculationId={breakdownCalculationId} />
            </TabsContent>
          </Tabs>

          {/* Royalty Breakdown breaks out of the narrow page container on large
              screens so all columns fit without horizontal scrolling. */}
          <div className="xl:relative xl:left-1/2 xl:right-1/2 xl:-mx-[50vw] xl:w-screen">
          <div className="xl:mx-auto xl:max-w-6xl xl:px-6">
          <Card>
              <CardHeader>
                  <div className="flex items-center justify-between">
                      <div>
                          <CardTitle>Royalty Breakdown</CardTitle>
                          <p className="text-sm text-muted-foreground mt-1">
                            {payments.some(isNet)
                              ? "Net-basis rows deduct each track's expenses from gross before applying the share; gross-basis rows apply the share to full earnings."
                              : "All rows are paid on gross revenue from the uploaded royalty statement."}
                          </p>
                      </div>
                      <DropdownMenu>
                          <DropdownMenuTrigger asChild><Button variant="outline" size="sm"><Download className="mr-2 h-4 w-4"/> Export</Button></DropdownMenuTrigger>
                          <DropdownMenuContent>
                              <DropdownMenuItem onClick={handleExportCSV}>Export as CSV</DropdownMenuItem>
                              <DropdownMenuItem onClick={handleExportExcel}>Export as Excel</DropdownMenuItem>
                          </DropdownMenuContent>
                      </DropdownMenu>
                  </div>
              </CardHeader>
              <CardContent>
                  {/* Desktop: table */}
                  <div className="hidden sm:block overflow-x-auto">
                    <Table className="w-full">
                        <TableHeader>
                            <TableRow>
                                <TableHead className="min-w-[140px]">Song</TableHead><TableHead className="min-w-[200px]">Payee</TableHead><TableHead>Role</TableHead><TableHead>Royalty Type</TableHead><TableHead>Basis</TableHead><TableHead className="whitespace-nowrap text-right">Gross Revenue</TableHead><TableHead className="whitespace-nowrap text-right">Expenses</TableHead><TableHead className="whitespace-nowrap text-right">Net Revenue</TableHead><TableHead className="whitespace-nowrap text-right">Share %</TableHead><TableHead className="whitespace-nowrap text-right">Amount Owed</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {calculationResult.payments.map((row, i) => (
                                <TableRow key={i}>
                                    <TableCell className="min-w-[140px]">{row.song_title}</TableCell>
                                    <TableCell className="min-w-[200px] font-medium">{row.party_name}</TableCell>
                                    <TableCell className="capitalize">{row.role}</TableCell>
                                    <TableCell className="capitalize">{row.royalty_type}</TableCell>
                                    <TableCell><BasisBadge payment={row} /></TableCell>
                                    <TableCell className="whitespace-nowrap text-right">{formatCurrency(grossOf(row))}</TableCell>
                                    <TableCell className="whitespace-nowrap text-right">{isNet(row) ? `-${formatCurrency(row.expenses_applied ?? 0)}` : "—"}</TableCell>
                                    <TableCell className="whitespace-nowrap text-right">{formatCurrency(netOf(row))}</TableCell>
                                    <TableCell className="whitespace-nowrap text-right">{row.percentage}%</TableCell>
                                    <TableCell className="whitespace-nowrap text-right font-medium">{formatCurrency(row.amount_to_pay)}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                  </div>
                  {/* Mobile: stacked cards */}
                  <div className="sm:hidden space-y-3">
                    {calculationResult.payments.map((row, i) => (
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

                  <div className="mt-6 flex justify-end border-t border-border pt-4">
                      <Button onClick={handlePayRoyalties} disabled={payableParties.length === 0}>
                          <Wallet className="mr-2 h-4 w-4" /> Pay Royalties
                      </Button>
                  </div>
              </CardContent>
          </Card>
          </div>
          </div>
        </div>
      )}
    </>
  );
};

export default CalculationResults;
