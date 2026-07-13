import { describe, expect, it } from "vitest";
import { mergeParsedContracts } from "../mergeParsedContracts";
import type { ParsedParty } from "@/hooks/useParseContractSplits";

const party = (over: Partial<ParsedParty>): ParsedParty => ({
  name: "Someone",
  role: "Producer",
  master_pct: 0,
  publishing_pct: 0,
  is_main_artist: false,
  ...over,
});

describe("mergeParsedContracts", () => {
  it("does not fabricate a 'you' row when no party is the main artist", () => {
    const { rows, mainArtistFoundAny } = mergeParsedContracts([
      {
        displayName: "deal.pdf",
        parties: [party({ name: "Marcus Adebayo", master_pct: 40 })],
        mainArtistFound: false,
      },
    ]);
    expect(mainArtistFoundAny).toBe(false);
    expect(rows).toHaveLength(1);
    expect(rows[0].isYou).toBe(false);
    expect(rows[0].name).toBe("Marcus Adebayo");
  });

  it("maps the main-artist party onto the 'you' row keeping the contract's name and alias note", () => {
    const { rows, mainArtistFoundAny } = mergeParsedContracts([
      {
        displayName: "deal.pdf",
        parties: [
          party({
            name: "Jane Q. Doe",
            role: "Artist",
            aliases: ["Jasmine Kiara"],
            master_pct: 60,
            publishing_pct: 50,
            is_main_artist: true,
          }),
        ],
        mainArtistFound: true,
      },
    ]);
    expect(mainArtistFoundAny).toBe(true);
    expect(rows[0]).toMatchObject({
      key: "you",
      isYou: true,
      name: "Jane Q. Doe",
      aliasNote: "p/k/a Jasmine Kiara",
      master: 60,
      publishing: 50,
    });
  });

  it("sets an alias note on non-main parties too", () => {
    const { rows } = mergeParsedContracts([
      {
        displayName: "deal.pdf",
        parties: [party({ name: "Marcus Adebayo", aliases: ["M-Bay"], master_pct: 40 })],
      },
    ]);
    expect(rows[0].aliasNote).toBe("p/k/a M-Bay");
    expect(rows[0].isYou).toBe(false);
  });

  it("leaves aliasNote undefined when there are no aliases", () => {
    const { rows } = mergeParsedContracts([
      { displayName: "deal.pdf", parties: [party({ name: "Marcus Adebayo", master_pct: 40 })] },
    ]);
    expect(rows[0].aliasNote).toBeUndefined();
  });

  it("merges the same party across contracts via alias identity", () => {
    const { rows, conflicts } = mergeParsedContracts([
      {
        displayName: "one.pdf",
        parties: [party({ name: "Jane Doe", aliases: ["Nova"], master_pct: 20 })],
      },
      {
        displayName: "two.pdf",
        parties: [party({ name: "Nova", master_pct: 20 })],
      },
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0].name).toBe("Jane Doe");
    expect(conflicts).toHaveLength(0);
  });

  it("raises a conflict (not a sum) when contracts disagree on a party's split", () => {
    const { rows, conflicts } = mergeParsedContracts([
      { displayName: "one.pdf", parties: [party({ name: "Marcus", master_pct: 20 })] },
      { displayName: "two.pdf", parties: [party({ name: "Marcus", master_pct: 35 })] },
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0].master).toBe(20); // first value kept, never summed
    expect(conflicts).toHaveLength(1);
    expect(conflicts[0].values).toHaveLength(2);
  });

  it("raises a conflict when an alias-matched duplicate disagrees", () => {
    const { rows, conflicts } = mergeParsedContracts([
      {
        displayName: "one.pdf",
        parties: [party({ name: "Jane Doe", aliases: ["Nova"], master_pct: 20 })],
      },
      { displayName: "two.pdf", parties: [party({ name: "Nova", master_pct: 50 })] },
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0].master).toBe(20);
    expect(conflicts).toHaveLength(1);
  });

  it("carries soundexchange_pct onto rows, separate from master", () => {
    const { rows } = mergeParsedContracts([
      {
        displayName: "deal.pdf",
        parties: [party({ name: "Marcus", master_pct: 40, soundexchange_pct: 15 })],
      },
    ]);
    expect(rows[0].master).toBe(40);
    expect(rows[0].soundexchange).toBe(15);
  });

  it("defaults soundexchange to 0 when the parse response omits it", () => {
    const { rows } = mergeParsedContracts([
      { displayName: "deal.pdf", parties: [party({ name: "Marcus", master_pct: 40 })] },
    ]);
    expect(rows[0].soundexchange).toBe(0);
  });

  it("raises a conflict when contracts disagree on a soundexchange share", () => {
    const { rows, conflicts } = mergeParsedContracts([
      {
        displayName: "one.pdf",
        parties: [party({ name: "Marcus", master_pct: 20, soundexchange_pct: 10 })],
      },
      {
        displayName: "two.pdf",
        parties: [party({ name: "Marcus", master_pct: 20, soundexchange_pct: 25 })],
      },
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0].soundexchange).toBe(10); // first value kept, never summed
    expect(conflicts).toHaveLength(1);
    expect(conflicts[0].values).toEqual([
      { file: "one.pdf", master: 20, publishing: 0, soundexchange: 10 },
      { file: "two.pdf", master: 20, publishing: 0, soundexchange: 25 },
    ]);
  });
});
