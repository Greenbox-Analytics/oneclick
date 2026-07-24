import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
// Registers jest-dom matchers (toBeDisabled, etc.) on vitest's `expect` for
// this file — no shared vitest.config.ts setupFiles entry exists yet.
import "@testing-library/jest-dom/vitest";
import { ConflictResolutionDialog } from "../ConflictResolutionDialog";
import { RevisionPromptDialog } from "../RevisionPromptDialog";

const conflict = {
  party_key: "romes", song_key: "home", royalty_type_key: "streaming",
  party_name: "Romes", song_title: "Home",
  claims: [
    { contract_ids: ["K"], contract_names: ["Kenji Contract.pdf"], amount: 3.0, percentage: 30 },
    { contract_ids: ["L"], contract_names: ["Lebron Contract.pdf"], amount: 3.5, percentage: 35 },
  ],
};

it("blocks submit until every conflict is resolved, then returns resolutions", () => {
  const onResolve = vi.fn();
  render(<ConflictResolutionDialog open conflicts={[conflict]} onResolve={onResolve} onCancel={() => {}} />);
  expect(screen.getByRole("button", { name: /use selected/i })).toBeDisabled();
  fireEvent.click(screen.getByLabelText(/35%/i));
  fireEvent.click(screen.getByRole("button", { name: /use selected/i }));
  expect(onResolve).toHaveBeenCalledWith([
    { party_key: "romes", song_key: "home", royalty_type_key: "streaming", governing_contract_id: "L" },
  ]);
});

it("revision prompt returns replace decision or none", () => {
  const onDecide = vi.fn();
  render(
    <RevisionPromptDialog open onDecide={onDecide} onCancel={() => {}}
      candidates={[{ statement_id: "A", name: "Q1-statement.xlsx", period_start: "2024-01-01", period_end: "2024-03-31", total: 10 }]} />
  );
  fireEvent.click(screen.getByLabelText(/Q1-statement/i));
  fireEvent.click(screen.getByRole("button", { name: /replace/i }));
  expect(onDecide).toHaveBeenCalledWith({ replace: "A" });
});
