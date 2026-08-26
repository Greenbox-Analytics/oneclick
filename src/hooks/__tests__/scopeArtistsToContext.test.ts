import { describe, expect, it } from "vitest";
import { scopeArtistsToContext } from "@/hooks/useArtistTeam";

const ORG_A = "org-a";
const ORG_B = "org-b";

const ROSTER = [
  { id: "personal-1", team_id: null },
  { id: "personal-2" }, // team_id absent entirely — same as null
  { id: "a-1", team_id: ORG_A },
  { id: "b-1", team_id: ORG_B },
];

describe("scopeArtistsToContext", () => {
  it("shows only personal artists in the personal context", () => {
    expect(scopeArtistsToContext(ROSTER, null, true).map((a) => a.id)).toEqual([
      "personal-1",
      "personal-2",
    ]);
  });

  it("shows only that org's artists in an org context", () => {
    expect(scopeArtistsToContext(ROSTER, ORG_A, true).map((a) => a.id)).toEqual(["a-1"]);
  });

  it("never leaks a sibling org's artists", () => {
    expect(scopeArtistsToContext(ROSTER, ORG_B, true).map((a) => a.id)).toEqual(["b-1"]);
  });

  it("hides nothing when there is nothing to switch between", () => {
    // The rollback guard: no pill in the header (one context, or licensing off)
    // means a filtered-out artist would be unreachable, not merely hidden.
    expect(scopeArtistsToContext(ROSTER, null, false)).toHaveLength(4);
  });
});
