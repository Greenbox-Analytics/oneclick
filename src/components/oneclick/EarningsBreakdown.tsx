import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Download, Globe, CalendarDays, Layers, Store } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";
import ExcelJS from "exceljs";
import { toast } from "sonner";
import {
  useEarningsBreakdown,
  type BreakdownDimension,
  type BreakdownResponse,
} from "@/hooks/useEarningsBreakdown";

const formatCurrency = (amount: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(amount);

const DIMENSION_META: Record<
  BreakdownDimension,
  { label: string; columnHeader: string; icon: React.ComponentType<{ className?: string }>; emptyHint: string }
> = {
  country: {
    label: "By Country",
    columnHeader: "Country",
    icon: Globe,
    emptyHint: "Statement didn't include a country column.",
  },
  month: {
    label: "By Month",
    columnHeader: "Month",
    icon: CalendarDays,
    emptyHint: "Statement didn't include a sale date column.",
  },
  format: {
    label: "By Format",
    columnHeader: "Delivery Format",
    icon: Layers,
    emptyHint: "Statement didn't include a delivery format column.",
  },
  vendor: {
    label: "By Vendor",
    columnHeader: "Vendor",
    icon: Store,
    emptyHint: "Statement didn't include a vendor column.",
  },
};

interface EarningsBreakdownProps {
  calculationId: string | null | undefined;
}

const DimensionPanel = ({
  calculationId,
  dimension,
}: {
  calculationId: string | null | undefined;
  dimension: BreakdownDimension;
}) => {
  const meta = DIMENSION_META[dimension];
  const { data, isLoading, isError } = useEarningsBreakdown(calculationId, dimension);

  const handleExport = async (kind: "csv" | "xlsx") => {
    if (!data || data.rows.length === 0) return;
    const headers = [meta.columnHeader, "Net Payable", "Row Count", "% of Total"];
    const rows = data.rows.map((r) => [r.key, r.net_payable, r.row_count, `${r.percent_of_total}%`]);

    if (kind === "csv") {
      const csv = [headers, ...rows]
        .map((row) => row.map((cell) => `"${cell}"`).join(","))
        .join("\n");
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `earnings_${dimension}.csv`;
      link.click();
      return;
    }

    const wb = new ExcelJS.Workbook();
    const ws = wb.addWorksheet(meta.label);
    ws.addRow(headers);
    rows.forEach((r) => ws.addRow(r));
    const buf = await wb.xlsx.writeBuffer();
    const blob = new Blob([buf], {
      type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `earnings_${dimension}.xlsx`;
    link.click();
  };

  if (!calculationId) {
    return (
      <div className="text-sm text-muted-foreground py-8 text-center">
        Save the calculation to see the earnings breakdown.
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-6 w-1/3" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="text-sm text-destructive py-8 text-center">
        Failed to load breakdown. Try refreshing.
      </div>
    );
  }

  const result: BreakdownResponse = data ?? {
    dimension,
    total: 0,
    row_count: 0,
    rows: [],
  };

  if (result.rows.length === 0) {
    return (
      <div className="text-sm text-muted-foreground py-8 text-center">
        No data to break down for this dimension. {meta.emptyHint}
      </div>
    );
  }

  const chartData = result.rows.slice(0, 12).map((r) => ({
    name: r.key.length > 18 ? `${r.key.slice(0, 17)}…` : r.key,
    fullName: r.key,
    value: r.net_payable,
  }));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-4">
        <div className="text-sm text-muted-foreground">
          Total {meta.label.toLowerCase()}:{" "}
          <span className="font-semibold text-foreground">{formatCurrency(result.total)}</span>{" "}
          across {result.row_count.toLocaleString()} line items
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => handleExport("csv")}>
            <Download className="w-4 h-4 mr-2" /> CSV
          </Button>
          <Button variant="outline" size="sm" onClick={() => handleExport("xlsx")}>
            <Download className="w-4 h-4 mr-2" /> Excel
          </Button>
        </div>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 16, left: 8, bottom: 24 }}>
            <XAxis
              dataKey="name"
              tick={{ fontSize: 11 }}
              angle={-30}
              textAnchor="end"
              interval={0}
              height={60}
            />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v: number) => formatCurrency(v)} width={80} />
            <Tooltip
              formatter={(v: number) => formatCurrency(Number(v))}
              labelFormatter={(_, payload) =>
                payload && payload[0]?.payload?.fullName ? payload[0].payload.fullName : ""
              }
            />
            <Bar dataKey="value" fill="hsl(150, 50%, 35%)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{meta.columnHeader}</TableHead>
              <TableHead>Net Payable</TableHead>
              <TableHead>Line items</TableHead>
              <TableHead>% of total</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {result.rows.map((row) => (
              <TableRow key={row.key}>
                <TableCell>{row.key}</TableCell>
                <TableCell>{formatCurrency(row.net_payable)}</TableCell>
                <TableCell>{row.row_count.toLocaleString()}</TableCell>
                <TableCell>{row.percent_of_total.toFixed(2)}%</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
};

const EarningsBreakdown = ({ calculationId }: EarningsBreakdownProps) => {
  const [active, setActive] = useState<BreakdownDimension>("country");

  return (
    <Card>
      <CardHeader>
        <CardTitle>Earnings Breakdown</CardTitle>
        <CardDescription>
          Where your royalties came from — drilled by country, month, delivery format, and vendor.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={active} onValueChange={(v) => setActive(v as BreakdownDimension)}>
          <TabsList className="grid grid-cols-4 w-full max-w-xl">
            {(Object.keys(DIMENSION_META) as BreakdownDimension[]).map((dim) => {
              const Icon = DIMENSION_META[dim].icon;
              return (
                <TabsTrigger key={dim} value={dim} className="gap-1.5">
                  <Icon className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">{DIMENSION_META[dim].label}</span>
                </TabsTrigger>
              );
            })}
          </TabsList>
          {(Object.keys(DIMENSION_META) as BreakdownDimension[]).map((dim) => (
            <TabsContent key={dim} value={dim} className="mt-4">
              <DimensionPanel calculationId={calculationId} dimension={dim} />
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default EarningsBreakdown;
