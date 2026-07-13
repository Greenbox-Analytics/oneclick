import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { WorkRow, type DashboardWork } from "@/components/registry/WorkRow";

afterEach(cleanup);

const work = (over: Partial<DashboardWork>): DashboardWork =>
  ({
    id: "w1",
    title: "Vanilla",
    status: "draft",
    released: false,
    isrc: "USAT22204902",
    ...over,
  }) as unknown as DashboardWork;

describe("WorkRow registration status badge", () => {
  it("hides the registration status badge for a released work", () => {
    render(<WorkRow work={work({ released: true, status: "draft" })} onOpen={vi.fn()} />);
    expect(screen.getByText("Released")).toBeTruthy();
    // "Draft" is a registration state — it must not appear next to "Released".
    expect(screen.queryByText("Draft")).toBeNull();
  });

  it("shows the registration status badge for an unreleased work", () => {
    render(<WorkRow work={work({ released: false, status: "draft" })} onOpen={vi.fn()} />);
    expect(screen.getByText("Unreleased")).toBeTruthy();
    expect(screen.getByText("Draft")).toBeTruthy();
  });
});
