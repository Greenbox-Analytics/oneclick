import { describe, it, expect } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import type { ReactNode } from "react";

import { useTabParam } from "@/hooks/useTabParam";

const TABS = ["works", "files", "settings"] as const;

const wrapper = (initial: string) =>
  function Wrapper({ children }: { children: ReactNode }) {
    return <MemoryRouter initialEntries={[initial]}>{children}</MemoryRouter>;
  };

const setup = (initial: string) =>
  renderHook(
    () => ({ tab: useTabParam(TABS, "files"), search: useLocation().search }),
    { wrapper: wrapper(initial) },
  );

describe("useTabParam", () => {
  it("reads the tab out of the URL", () => {
    expect(setup("/p?tab=settings").result.current.tab[0]).toBe("settings");
  });

  it("falls back for an absent or unrecognised tab, without rewriting the URL", () => {
    const absent = setup("/p");
    expect(absent.result.current.tab[0]).toBe("files");
    expect(absent.result.current.search).toBe("");

    const bogus = setup("/p?tab=nope");
    expect(bogus.result.current.tab[0]).toBe("files");
    expect(bogus.result.current.search).toBe("?tab=nope");
  });

  it("writes the tab back to the URL so a refresh or Back restores it", () => {
    const { result } = setup("/p");
    act(() => result.current.tab[1]("settings"));
    expect(result.current.search).toBe("?tab=settings");
    expect(result.current.tab[0]).toBe("settings");
  });

  it("merges rather than replacing — sibling params survive a tab switch", () => {
    const { result } = setup("/p?taskId=t1&tab=works");
    act(() => result.current.tab[1]("settings"));
    expect(new URLSearchParams(result.current.search).get("taskId")).toBe("t1");
    expect(new URLSearchParams(result.current.search).get("tab")).toBe("settings");
  });
});
