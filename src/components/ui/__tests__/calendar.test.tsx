import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";

import { Calendar } from "@/components/ui/calendar";

// Regression guard for the react-day-picker v8 -> v10 API migration. The old v8
// classNames (head_row / head_cell / cell / day) and components (IconLeft /
// IconRight) silently no-op on v10, producing the broken, unstyled grid we saw in
// the task date pickers. These assertions verify the grid structure and month nav
// render, and that the `disabled` matcher actually gates which days are selectable.

afterEach(cleanup);

// Fixed month keeps the rendered grid deterministic regardless of the current date.
const JULY_2026 = new Date(2026, 6, 1);

describe("Calendar", () => {
  it("renders a full 7-column weekday header laid out with flexbox", () => {
    const { container } = render(<Calendar mode="single" defaultMonth={JULY_2026} />);
    expect(container.querySelectorAll("thead th")).toHaveLength(7);
    // The weekday row must carry our `weekdays` class (flex). This is the key that
    // did not exist in the v8 API (it was `head_row`), so on the broken config the
    // row is unstyled and the header overflows — this assertion fails there.
    const weekdayRow = container.querySelector("thead tr");
    expect(weekdayRow?.className).toContain("flex");
  });

  it("renders previous and next month navigation buttons", () => {
    render(<Calendar mode="single" defaultMonth={JULY_2026} />);
    expect(screen.getByRole("button", { name: /previous month/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: /next month/i })).toBeTruthy();
  });

  it("renders a full month of selectable day cells", () => {
    render(<Calendar mode="single" defaultMonth={JULY_2026} />);
    // 31 days in July plus outside days filling the weeks -> >= 31 gridcells.
    expect(screen.getAllByRole("gridcell").length).toBeGreaterThanOrEqual(31);
    expect(screen.getByText("15").closest("button")).not.toBeNull();
  });

  it("gates selection via the `disabled` matcher (start/due bounds)", () => {
    render(
      <Calendar mode="single" defaultMonth={JULY_2026} disabled={{ before: new Date(2026, 6, 15) }} />,
    );
    const blocked = screen.getByText("10").closest("button");
    const allowed = screen.getByText("20").closest("button");
    expect(blocked?.hasAttribute("disabled")).toBe(true);
    expect(allowed?.hasAttribute("disabled")).toBe(false);
  });
});
