import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
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

// WorkRow renders <Link>s, so it needs a router context.
const renderRow = (w: DashboardWork) =>
  render(
    <MemoryRouter>
      <WorkRow work={w} />
    </MemoryRouter>,
  );

describe("WorkRow registration status badge", () => {
  it("hides the registration status badge for a released work", () => {
    renderRow(work({ released: true, status: "draft" }));
    expect(screen.getByText("Released")).toBeTruthy();
    // "Draft" is a registration state — it must not appear next to "Released".
    expect(screen.queryByText("Draft")).toBeNull();
  });

  it("shows the registration status badge for an unreleased work", () => {
    renderRow(work({ released: false, status: "draft" }));
    expect(screen.getByText("Unreleased")).toBeTruthy();
    expect(screen.getByText("Draft")).toBeTruthy();
  });
});
