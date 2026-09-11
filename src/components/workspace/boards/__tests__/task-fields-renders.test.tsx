import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { BoardTaskDetail } from "@/types/integrations";

vi.mock("@/hooks/useOrgs", () => ({
  useOrgRoster: () => ({ data: [] }),
}));
vi.mock("@/hooks/useBoardsList", () => ({
  useBoardsList: () => ({ data: [] }),
}));
vi.mock("@/hooks/useTaskAssignees", () => ({
  useAddAssignee: () => ({ mutate: vi.fn(), isPending: false }),
  useRemoveAssignee: () => ({ mutate: vi.fn(), isPending: false }),
}));
vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => ({ user: { id: "u1" } }),
}));

import { TaskFields } from "../TaskFields";

// Radix's Popover (MultiSelectCombobox) measures its content via ResizeObserver,
// which jsdom doesn't implement. Stub is enough — nothing here asserts on layout.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver;

const wrap = (ui: React.ReactElement) => (
  <QueryClientProvider client={new QueryClient()}>
    <MemoryRouter>{ui}</MemoryRouter>
  </QueryClientProvider>
);

// The same stub TaskDetailPanel hands TaskFields in create mode.
const createModeTask = {
  id: "",
  user_id: "",
  title: "",
  position: 0,
  is_parent: false,
  parent: null,
  artists: [],
  projects: [],
  documents: [],
  comments: [],
  created_at: "",
  updated_at: "",
} as unknown as BoardTaskDetail;

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("TaskFields", () => {
  // Regression: TaskFields once used a React hook it never imported. The board
  // rendered fine, and the whole page went blank the moment the task panel
  // (create OR edit) mounted this component. Mounting it must not throw.
  it("renders in create mode without throwing", () => {
    const noop = () => {};
    render(
      wrap(
        <TaskFields
          task={createModeTask}
          description=""
          setDescription={noop}
          priority=""
          setPriority={noop}
          startDate={undefined}
          setStartDate={noop}
          dueDate={undefined}
          setDueDate={noop}
          color=""
          setColor={noop}
          teamId={null}
          statusColumnId=""
          setStatusColumnId={noop}
          columns={[]}
          parents={[]}
          artists={[]}
          projects={[]}
          contracts={[]}
          selectedArtistIds={[]}
          setSelectedArtistIds={noop}
          selectedProjectIds={[]}
          setSelectedProjectIds={noop}
          selectedContractIds={[]}
          setSelectedContractIds={noop}
          saveField={noop}
          saveFields={noop}
        />,
      ),
    );
    expect(screen.getByText(/^priority$/i)).toBeTruthy();
  });
});
