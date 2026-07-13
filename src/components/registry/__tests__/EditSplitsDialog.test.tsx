import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, cleanup } from "@testing-library/react";
import { EditSplitsDialog } from "@/components/registry/EditSplitsDialog";
import type { SplitRow } from "@/components/registry/RoyaltySplitsTable";

afterEach(cleanup);

const rows = (): SplitRow[] => [
  { key: "a", name: "Rome", role: "artist", master: 30, publishing: 25 },
  { key: "you", name: "Khaprito", role: "Primary Artist", isYou: true, master: 70, publishing: 75 },
];

function setup(over: Partial<React.ComponentProps<typeof EditSplitsDialog>> = {}) {
  const onSave = vi.fn();
  const onOpenChange = vi.fn();
  render(
    <EditSplitsDialog
      open
      onOpenChange={onOpenChange}
      initialRows={rows()}
      onSave={onSave}
      {...over}
    />
  );
  return { onSave, onOpenChange };
}

describe("EditSplitsDialog", () => {
  it("renders an editable name field for a collaborator", () => {
    setup();
    expect(screen.getByDisplayValue("Rome")).toBeTruthy();
  });

  it("locks the 'You' row — shown as text, with no name field and no remove button", () => {
    setup();
    // Name appears as static text, not an editable input.
    expect(screen.getByText("Khaprito")).toBeTruthy();
    expect(screen.queryByDisplayValue("Khaprito")).toBeNull();
    expect(screen.queryByLabelText(/remove khaprito/i)).toBeNull();
  });

  it("shows a balance warning when shares don't total 100%", () => {
    setup({ initialRows: [{ key: "a", name: "Rome", role: "", master: 40, publishing: 40 }] });
    expect(screen.getAllByText(/should total 100%/i).length).toBeGreaterThan(0);
  });

  it("adds a new empty party block when 'Add party' is clicked", () => {
    setup();
    expect(screen.getAllByPlaceholderText("e.g. Jordan Lee")).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: /add party/i }));
    expect(screen.getAllByPlaceholderText("e.g. Jordan Lee")).toHaveLength(2);
  });

  it("removes a collaborator when its remove button is clicked", () => {
    setup();
    fireEvent.click(screen.getByLabelText(/remove rome/i));
    expect(screen.queryByDisplayValue("Rome")).toBeNull();
  });

  it("saves the current rows and closes on 'Save changes'", async () => {
    const { onSave, onOpenChange } = setup();
    fireEvent.click(screen.getByRole("button", { name: /save changes/i }));
    await waitFor(() => expect(onSave).toHaveBeenCalledTimes(1));
    const savedRows = onSave.mock.calls[0][0] as SplitRow[];
    expect(savedRows).toHaveLength(2);
    expect(savedRows[0].name).toBe("Rome");
    await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false));
  });

  it("does not persist edits when 'Cancel' is clicked", () => {
    const { onSave, onOpenChange } = setup();
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(onSave).not.toHaveBeenCalled();
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });
});
