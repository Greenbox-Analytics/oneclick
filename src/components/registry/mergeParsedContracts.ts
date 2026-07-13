import type { ParsedParty } from "@/hooks/useParseContractSplits";
import type { SplitRow } from "./RoyaltySplitsTable";

export interface MergeConflict {
  name: string;
  values: Array<{ file: string; master: number; publishing: number; soundexchange: number }>;
}

const normName = (name: string) => name.trim().toLowerCase().replace(/\s+/g, " ");

/** "p/k/a Stage Name" note rendered under a party's contract (legal) name. */
const aliasNoteFor = (p: ParsedParty): string | undefined =>
  p.aliases && p.aliases.length > 0 ? `p/k/a ${p.aliases.join(" · ")}` : undefined;

/**
 * Merge the parsed parties of several contracts into one split table.
 * Correctness rules: values are NEVER silently summed — the same person in two
 * contracts keeps the first value and raises a conflict; the main artist's
 * share is stated relative to each individual deal, so summing it would
 * double-count. Conflicts are surfaced for the user to reconcile in the
 * editable table. Parties are matched across contracts by normalized name OR
 * any alias, so "Jane Doe p/k/a Nova" in one contract and "Nova" in another
 * merge into a single row. No row is fabricated for the primary artist — the
 * "you" row exists only when a contract actually names them.
 */
export function mergeParsedContracts(
  contracts: Array<{ displayName: string; parties?: ParsedParty[]; mainArtistFound?: boolean }>
): { rows: SplitRow[]; mainArtistFoundAny: boolean; conflicts: MergeConflict[] } {
  const byKey = new Map<string, SplitRow>();
  // normalized alias -> byKey key, for cross-contract identity via stage names
  const aliasIndex = new Map<string, string>();
  const conflictByKey = new Map<string, MergeConflict>();
  const firstSeen = new Map<
    string,
    { file: string; master: number; publishing: number; soundexchange: number }
  >();
  let youRow: SplitRow | null = null;
  const mainArtistFoundAny = contracts.some((c) => c.mainArtistFound);

  const noteConflict = (
    key: string,
    name: string,
    file: string,
    master: number,
    publishing: number,
    soundexchange: number
  ) => {
    let conflict = conflictByKey.get(key);
    if (!conflict) {
      conflict = { name, values: [firstSeen.get(key)!] };
      conflictByKey.set(key, conflict);
    }
    conflict.values.push({ file, master, publishing, soundexchange });
  };

  for (const c of contracts) {
    for (const p of c.parties || []) {
      const master = Math.round(p.master_pct);
      const publishing = Math.round(p.publishing_pct);
      const soundexchange = Math.round(p.soundexchange_pct ?? 0);
      if (p.is_main_artist) {
        if (!youRow) {
          youRow = {
            key: "you",
            name: p.name,
            role: p.role || "Primary Artist",
            isYou: true,
            master,
            publishing,
            soundexchange,
            aliasNote: aliasNoteFor(p),
          };
          firstSeen.set("you", { file: c.displayName, master, publishing, soundexchange });
        } else if (
          master !== youRow.master ||
          publishing !== youRow.publishing ||
          soundexchange !== (youRow.soundexchange ?? 0)
        ) {
          noteConflict("you", youRow.name, c.displayName, master, publishing, soundexchange);
        }
        continue;
      }
      const candidates = [normName(p.name), ...(p.aliases || []).map(normName)].filter(Boolean);
      let key: string | undefined;
      for (const cand of candidates) {
        if (byKey.has(cand)) {
          key = cand;
          break;
        }
        const mapped = aliasIndex.get(cand);
        if (mapped) {
          key = mapped;
          break;
        }
      }
      const existing = key ? byKey.get(key) : undefined;
      if (existing) {
        if (
          master !== existing.master ||
          publishing !== existing.publishing ||
          soundexchange !== (existing.soundexchange ?? 0)
        ) {
          noteConflict(key!, existing.name, c.displayName, master, publishing, soundexchange);
        }
        // equal values = duplicate mention of the same deal — dedupe silently
      } else {
        key = normName(p.name);
        byKey.set(key, {
          key,
          name: p.name,
          role: p.role,
          isYou: false,
          master,
          publishing,
          soundexchange,
          aliasNote: aliasNoteFor(p),
        });
        firstSeen.set(key, { file: c.displayName, master, publishing, soundexchange });
        for (const alias of p.aliases || []) {
          const aliasKey = normName(alias);
          if (aliasKey && aliasKey !== key && !aliasIndex.has(aliasKey)) {
            aliasIndex.set(aliasKey, key);
          }
        }
      }
    }
  }

  return {
    rows: youRow ? [youRow, ...byKey.values()] : [...byKey.values()],
    mainArtistFoundAny,
    conflicts: Array.from(conflictByKey.values()),
  };
}
