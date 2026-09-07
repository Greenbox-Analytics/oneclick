import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, within } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { UsageDetailDialog } from "../UsageDetailDialog";
import type { UsageSubject } from "@/lib/orgUsage";

// Recharts measures its container with ResizeObserver, which jsdom lacks.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver;

const SUBJECT: UsageSubject = {
  kind: "key",
  id: "k1",
  title: "Production",
  subtitle: "API key · Ingest",
  total: 80,
  runs: 3,
  tools: { oneclick: 60, registry: 0, splitsheet: 20, zoe: 0 },
  series: [
    { day: "2026-09-02", actions: [{ action: "partner_oneclick_run", credits: 60, runs: 2 }] },
    { day: "2026-09-03", actions: [{ action: "partner_split_sheet", credits: 20, runs: 1 }] },
  ],
  facts: [
    { label: "Key", value: "mk_live_abcd…" },
    { label: "Status", value: "revoked" },
    { label: "Last used", value: "Never" },
  ],
};

const show = (over: Partial<React.ComponentProps<typeof UsageDetailDialog>> = {}) =>
  render(
    <UsageDetailDialog
      subject={SUBJECT}
      range="mtd"
      since="2026-09-01T00:00:00+00:00"
      windowTotal={200}
      onClose={() => {}}
      {...over}
    />,
  );

const tile = (label: string) => screen.getByText(label).parentElement!;

afterEach(cleanup);

describe("UsageDetailDialog", () => {
  it("names the subject and the window it covers", () => {
    show();
    const dialog = within(screen.getByRole("dialog"));
    expect(dialog.getByRole("heading", { name: "Production" })).toBeInTheDocument();
    expect(dialog.getByText(/API key · Ingest/)).toHaveTextContent("Sep 1, 2026 – ");
  });

  it("shows credits, runs and the share of the window's total", () => {
    show();
    expect(tile("Credits used")).toHaveTextContent("80");
    expect(tile("Runs")).toHaveTextContent("3");
    expect(tile("Share of total")).toHaveTextContent("40%");
    expect(tile("Share of total")).toHaveTextContent("of 200 credits in this window");
  });

  it("has no share to show when the window itself is empty", () => {
    show({ windowTotal: 0 });
    expect(tile("Share of total")).toHaveTextContent("—");
    expect(screen.queryByText(/credits in this window/)).not.toBeInTheDocument();
  });

  it("says so instead of charting when nothing was spent in the window", () => {
    show({ subject: { ...SUBJECT, series: [] } });
    expect(screen.getByText("No credits used in this window.")).toBeInTheDocument();
    // The heading stays — only the chart is replaced.
    expect(screen.getByText("Credits over time")).toBeInTheDocument();
  });

  it("lists the subject's facts", () => {
    show();
    const dialog = within(screen.getByRole("dialog"));
    for (const f of SUBJECT.facts) {
      expect(dialog.getByText(f.label).parentElement).toHaveTextContent(f.value);
    }
  });

  it("renders nothing until a subject is selected", () => {
    show({ subject: null });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("closes on Escape and on the close button", () => {
    const onClose = vi.fn();
    show({ onClose });
    fireEvent.click(screen.getByRole("button", { name: /close/i }));
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});
