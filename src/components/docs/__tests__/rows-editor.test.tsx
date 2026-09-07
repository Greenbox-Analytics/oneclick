import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { useState } from "react";
import { RequestPreview, RowsEditor, rowsToObjects, validateRows, type Column, type Row } from "../RowsEditor";

const COLS: Column[] = [
  { key: "name", label: "Name", kind: "text", required: true },
  { key: "role", label: "Role", kind: "select", options: ["Producer", "Writer"], required: true },
  { key: "master_percentage", label: "Master %", kind: "number", min: 0, max: 100 },
];

afterEach(cleanup);

describe("validateRows", () => {
  it("flags required, out-of-range, not-a-number and unknown-choice cells by row and key", () => {
    const rows: Row[] = [
      { name: "Jane", role: "Producer", master_percentage: "50" },
      { name: "", role: "DJ", master_percentage: "150" },
      { name: "Sam", role: "Writer", master_percentage: "abc" },
    ];
    expect(validateRows(COLS, rows)).toEqual([
      { row: 1, key: "name", message: "Name is required" },
      { row: 1, key: "role", message: "Role must be one of the choices" },
      { row: 1, key: "master_percentage", message: "Master % must be between 0 and 100" },
      { row: 2, key: "master_percentage", message: "Master % must be a number" },
    ]);
  });

  it("accepts options computed from other rows", () => {
    const cols: Column[] = [{ key: "party", label: "Party", kind: "select", options: () => ["Jane"], required: true }];
    expect(validateRows(cols, [{ party: "Jane" }])).toEqual([]);
    expect(validateRows(cols, [{ party: "Sam" }])).toHaveLength(1);
  });
});

describe("rowsToObjects", () => {
  it("casts numbers and drops empty optional cells", () => {
    expect(
      rowsToObjects(COLS, [
        { name: "Jane", role: "Producer", master_percentage: "50" },
        { name: "Sam", role: "Writer", master_percentage: "" },
      ])
    ).toEqual([{ name: "Jane", role: "Producer", master_percentage: 50 }, { name: "Sam", role: "Writer" }]);
  });
});

describe("RowsEditor", () => {
  function Harness() {
    const [rows, setRows] = useState<Row[]>([{ name: "Jane", role: "Producer", master_percentage: "50" }]);
    return (
      <>
        <RowsEditor label="Contributor" columns={COLS} rows={rows} onChange={setRows} blank={() => ({ name: "", role: "", master_percentage: "" })} addLabel="Add contributor" errors={validateRows(COLS, rows)} />
        <RequestPreview body={{ contributors: rowsToObjects(COLS, rows) }} />
      </>
    );
  }

  it("adds, edits and removes rows, keeps at least one, and previews the body", () => {
    render(<Harness />);
    expect(screen.getByLabelText("Remove Contributor 1")).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Add contributor" }));
    expect(screen.getAllByLabelText(/^Contributor \d Name$/)).toHaveLength(2);
    fireEvent.change(screen.getByLabelText("Contributor 2 Name"), { target: { value: "Sam" } });
    // The empty role on row 2 is an inline error, never a silent send.
    expect(screen.getByRole("alert")).toHaveTextContent("Role is required");
    fireEvent.change(screen.getByLabelText("Contributor 2 Role"), { target: { value: "Writer" } });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(JSON.parse(screen.getByLabelText("Request body").textContent!)).toEqual({
      contributors: [{ name: "Jane", role: "Producer", master_percentage: 50 }, { name: "Sam", role: "Writer" }],
    });
    fireEvent.click(screen.getByLabelText("Remove Contributor 2"));
    expect(screen.getAllByLabelText(/^Contributor \d Name$/)).toHaveLength(1);
  });

  it("keeps each row's own values when a middle row is removed (guards index keys)", () => {
    function ThreeRows() {
      const [rows, setRows] = useState<Row[]>([
        { name: "Jane", role: "Producer", master_percentage: "50" },
        { name: "Sam", role: "Writer", master_percentage: "30" },
        { name: "Alex", role: "Producer", master_percentage: "20" },
      ]);
      return (
        <>
          <RowsEditor label="Contributor" columns={COLS} rows={rows} onChange={setRows} blank={() => ({ name: "", role: "", master_percentage: "" })} addLabel="Add contributor" errors={validateRows(COLS, rows)} />
          <RequestPreview body={{ contributors: rowsToObjects(COLS, rows) }} />
        </>
      );
    }
    render(<ThreeRows />);
    fireEvent.click(screen.getByLabelText("Remove Contributor 2"));
    expect(JSON.parse(screen.getByLabelText("Request body").textContent!)).toEqual({
      contributors: [
        { name: "Jane", role: "Producer", master_percentage: 50 },
        { name: "Alex", role: "Producer", master_percentage: 20 },
      ],
    });
  });

  it("keeps a stale select value visible and flagged as an unknown choice", () => {
    const cols: Column[] = [{ key: "party", label: "Party", kind: "select", options: () => ["Jane"] }];
    const rows: Row[] = [{ party: "Sam" }];
    render(<RowsEditor label="Share" columns={cols} rows={rows} onChange={() => {}} blank={() => ({ party: "" })} addLabel="Add share" errors={validateRows(cols, rows)} />);
    const select = screen.getByLabelText("Share 1 Party") as HTMLSelectElement;
    expect(select.value).toBe("Sam");
    expect(screen.getByRole("option", { name: "Sam (no longer listed)" }).selected).toBe(true);
    expect(screen.getByRole("alert")).toHaveTextContent("Party must be one of the choices");
  });
});
