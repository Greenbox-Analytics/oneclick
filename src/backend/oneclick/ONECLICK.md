# OneClick Royalty Calculator

Computes per-party royalty payments by reading a contract + a royalty statement: parse the contract's splits, match works to statement songs, and apply each party's percentage to the song's payable total.

## Calculation Flow

```
/oneclick/calculate-royalties[-stream]  (main.py)
    │
    ├─ Load contract markdown (project_files.contract_markdown) + the royalty statement (Excel/CSV)
    │
    ├─ Parse contract  (contract_parser.MusicContractParser.parse_contract, one LLM call)
    │   → ContractData: parties[], works[], royalty_shares[] (party, royalty_type, percentage, terms)
    │   → only streaming-equivalent royalty types are kept for calculation
    │
    ├─ Read statement  (read_royalty_statement → song_totals)
    │   → auto-detects the NET-PAYABLE column (excludes withheld/deduction/fee/commission/advance)
    │
    ├─ Calculate  (_calculate_payments_from_data)
    │   → for each work matched to a statement song:
    │       amount_to_pay = song_total × (percentage / 100)
    │
    └─ Step 4 (log-only): contract-nuance DETECTION  ← see below
```

Entry points: `RoyaltyCalculator.calculate_payments(... full_text, contract_id, user_id)` (single) and
`calculate_payments_from_contract_ids(...)` (multi). The wrapper `zoe_chatbot.helpers.calculate_royalty_payments`
forwards `contract_id` + `user_id` so logs/analytics are accountable.

**Note:** the calc is intentionally a **flat percentage** on the statement's net-payable total. It does not model deductions, net-vs-gross basis, recoupment, escalations, or reserves.

**Shared markdown / page markers:** `contract_markdown` is the same cache Zoe uses, which now carries `[[PAGE n]]` markers for Zoe's contract page-jump. OneClick **strips** them (`strip_page_markers`, applied where `contract_markdowns` is built in `main.py`) before parsing, so the contract parser and the nuance **verbatim-quote** check never see marker noise (a page-straddling clause would otherwise fail the substring match). Markers are a Zoe-only concern; royalty math is unaffected. The other `calculate_payments` call site passes no `contract_markdowns` (`full_text=None`), so it skips the nuance audit entirely and never sees markers either.

## Contract-Nuance Detection (log-only v1)

Because the base is already net-payable and the calc applies a flat percentage, some contracts describe a *different basis* (net receipts, a stated % deduction, wholesale-vs-retail) that the flat calc doesn't capture. v1 **detects and logs** such clauses for human review — it **changes no payout numbers**. (Auto-adjustment is deferred to v2, gated on resolving whether the statement base sits upstream or downstream of a given deduction, and on the streaming-only-vs-physical-deduction mismatch.)

Module: `oneclick/nuance_adjuster.py`.

- `audit_contract_basis(contract_markdown, royalty_types_present, *, openai_client, search_fn)` — one strict LLM pass that detects an explicit basis/deduction clause and returns a `BasisFinding` (`basis`, `implied_factor`, `clause_quote`, `affected_types`, `kb_reference`) or `None`.
  - **Guardrails:** the `clause_quote` must be **verbatim** in the contract (normalizing curly/straight quotes, en/em/soft dashes, and hyphen spacing) and at least 12 chars; `implied_factor` must be in `(0, 1]`; any LLM/parse exception → `None`. The reference book (`search_reference`) only enriches the note — it never sets the factor.
- `log_basis_finding(payments, finding, *, contract_id, user_id)` — writes the full detail (incl. verbatim clause + book passage) to the backend **`oneclick.audit`** logger, emits an aggregate **`oneclick_basis_detected`** PostHog event (props: `implied_factor`, `affected_count`, `has_kb` — **never** the clause text), and returns payments **unchanged**.

Wired into the **single-contract** path (`calculate_payments`), wrapped in try/except so nuance logging can never break a payout calc. Multi-contract logs a deferred note (the merge loses per-contract attribution).

## Where findings surface

- **Backend audit log** (`oneclick.audit`) — full detail incl. the quoted clause + book reference + the contract_id + the lines/amount a human should review.
- **PostHog** (`oneclick_basis_detected`) — aggregate-only observability (no contract content), so the team can measure how often basis clauses actually fire before investing in auto-adjustment or a review UI.

## Response (unchanged by v1)

`OneClickRoyaltyResponse` → `payments[]` of `RoyaltyPaymentResponse` (`song_title`, `party_name`, `role`, `royalty_type`, `percentage`, `total_royalty`, `amount_to_pay`, `terms`). The user sees one value per line; nuance detection is backend-only.

## Analytics

`oneclick_calc_started` / `oneclick_calc_completed` / `oneclick_calc_failed` (funnel, in `main.py`), `tool_used` (`tool=oneclick`), and `oneclick_basis_detected` (this feature).

## Tests

- `tests/test_nuance_adjuster.py` — unit: `audit_contract_basis` guardrails (verbatim quote / factor range / exception → `None`) and `log_basis_finding` (payout **unchanged**, audit-log record, aggregate-only analytics).
- `tests/test_oneclick_nuance_integration.py` — the wired `calculate_payments` path (nuance is logged **without changing the payout**, and is skipped when `full_text` is absent), the **namespace** consultation (`search_fn` is the real `search_reference`, called with the basis-keyed query, and its passage is threaded into the finding), a **golden-output** math regression for a normal contract, and an **opt-in live namespace smoke** (`RUN_LIVE_REFERENCE_TESTS=1`; needs `PINECONE_API_KEY` + `OPENAI_API_KEY` + the book uploaded — skipped in CI).

---

# Royalty Ledger & Payment Tracking (2026-07 reconciliation)

Everything below the calculator: how confirmed calculations become durable payment obligations,
how re-runs/revisions reconcile instead of double-counting, and how payouts settle them.
Full design rationale: `docs/superpowers/specs/2026-07-23-oneclick-ledger-reconciliation-design.md` (gitignored, local).

## The mental model

- **`royalty_lines`** = what each payee has EARNED. One row per obligation, identified by
  `(royalty_statement_id, project_id, payee_id, song_key, royalty_type_key)` — "Kenji is owed
  $2 for *Home* on this statement." Enforced by unique index `royalty_lines_identity_ux`.
- **`royalty_payouts` + `royalty_payout_coverage`** = what has been PAID. `covered_amount` is a
  frozen snapshot in **statement currency**, never recomputed, never FX-converted.
- **Everything else is derived at read time** — owed / outstanding / settled / overpaid are never
  stored. `service._bucket_state(earned, paid, drafted)` is the single state ladder.
- **`royalty_ledger_history`** = append-only audit. Every mutation of a line or coverage row writes
  its prior state first (`oneclick/royalties/history.py:record`). A failed history write aborts the
  mutation. Valid actions are enforced by the table's CHECK constraint (single source of truth —
  do not mirror the list in Python).

## Provenance: files justify numbers, they don't identify them

Each line carries `source_contracts` — jsonb snapshots `{id, name, hash}` of the contract file(s)
asserting it, captured at sync time (they outlive file deletion), plus
`statement_content_hash`/`statement_file_name` for the statement. File ids are **provenance, not
identity**: reuploading a file mints a new UUID, but content-hash adoption maps it back to the
same obligations.

## The gated sync — `oneclick/royalties/ledger_sync.py`

`gated_sync` is **the only writer of `royalty_lines`**. Both write paths go through it:
`POST /oneclick/confirm` (gate failure → HTTP 409 `{"gate", "payload"}`, raised BEFORE the calc
cache is written) and the SSE cache-hit branch (gate failure → `needs_confirmation` event; the
resolution comes back through confirm). Five gates run before ANY write; a raise leaves the ledger
byte-identical:

1. **Tombstone** — statement was superseded by a revision → refuse ("run the newer file").
2. **Statement adoption** — new file id, same `content_hash` as a ledgered statement → sync under
   the existing statement id (delete + reupload of an identical file never forks a bucket).
3. **Contract adoption** — a run contract's hash matches a dead source id → sources re-pointed
   (history `adopted`), no false conflict, no staleness. Computed at gate time, applied only after
   all gates pass.
4. **Cross-run conflict** — the run's split for `(party, song)` differs from a stored line asserted
   by a contract NOT in this run. Compared on **percentages** when both known (expense recalcs
   change amounts, not splits — not a conflict). The user picks the governing contract
   (`conflict_resolutions`); choosing the absent contract leaves the stored line untouched.
5. **Revision overlap** — the run overlaps a different statement's lines (same payees/songs,
   overlapping period) → "is this a revision?" prompt (`revision_decision`). **Replace** →
   `execute_replace`: old lines superseded to history, its coverage re-pointed onto the new bucket
   (merged on PK collision, `moved_from` set), tombstone written, old calc cache dropped — paid
   amounts then net against the revised earned. **"New earnings"** → persisted as a
   `not_related` row so the pair never re-prompts. The client-supplied replace id is validated
   against the gate's own candidates (fail closed).

Then the **run-scoped upsert**: a run is authoritative only for the contracts it includes. It
upserts what its contracts assert, stale-deletes only lines sourced **entirely** from its own
contract set, strips its ids from shared lines it no longer asserts, and NEVER touches lines owned
by other contracts or lines with empty sources (legacy/unknown owner). Same run twice = zero
writes, zero history. This is what killed the old wipe-and-rewrite bug where running `{contract A}`
erased collaborators from `{contract B}`.

Split payees survive re-runs: `split_payee` sets `payee_locked` + `locked_party_key` (the ORIGINAL
party), and the upsert routes that party's payment to the locked line — per party, so multi-party
songs can't cross-wire.

## Credits (overpayment) and payouts

Overpaid = `paid > earned` per bucket, derived from **paid** coverage only (drafts can be canceled;
credit backed by them would be phantom), tracked **per statement currency** as `credit_by_ccy`,
never FX-netted. At payout creation (`create_payouts`), same-currency credits reduce the payout
total by re-allocating the excess coverage from the overpaid bucket onto the owed bucket —
attributed to the ORIGINAL payout, `moved_from` set, history-recorded. A payout whose coverage was
ever moved can't be reverted to draft (the money is spoken for elsewhere). Payout creation also
409s (`{"stale_lines"}`) when a line's every source file is gone — `?force=true` overrides.

## Deletion behavior

- **Contract file delete** (`DELETE /contracts/{id}`) → `remove_contract_from_ledger` first:
  shared lines lose that source entry; sole-source lines are deleted (history-recorded). Payment
  records are NEVER touched — a paid bucket whose earned side shrinks surfaces as an explicit
  credit, not silent loss. Failure aborts the file delete (retryable). The frontend shows an
  impact warning first (`GET /oneclick/royalties/contracts/{id}/impact`, wired in FilesTab).
- **Statement file delete** → does nothing to the ledger (the FK CASCADEs were removed — the
  statement id is an opaque bucket key that outlives the file). The revision flow reconciles when
  the corrected file arrives.
- **Explicit purge** (`DELETE /oneclick/royalties/projects/{id}/entries`) is the only thing that
  deletes coverage, and it history-records every row (`manual_purge`).

## Invariants (do not break these)

1. No file deletion ever touches payment records.
2. A run is authoritative only for the contracts it includes; empty-source lines are untouchable.
3. Every line/coverage mutation writes prior state to history first.
4. Balances are always derived, never stored.
5. Gates fail closed on every write path — no ledger or calc-cache write while a gate is unresolved.
6. All ledger queries are user-scoped in code — the service-role client bypasses RLS, so
   per-function scoping is the ONLY authorization.

## Tests

`tests/test_ledger_sync_engine.py` (gates + upsert, with the spec's 28-row scenario matrix mapped
in its header comment), `test_ledger_sync_core.py` (aggregation/conflict semantics),
`test_royalties_credit_netting.py` (credits, staleness, revert guard, splits),
`test_oneclick_confirm_gates.py` / `test_oneclick_stream_gates.py` (endpoint wiring),
`test_royalties_delete.py` (deletion guardrails, impact, audited purge). Shared in-memory fake:
`tests/fake_supabase.py` — it mirrors the real post-migration schema; never give it columns the
real schema lacks. Migration SQL (`20260723000000_royalty_ledger_reconciliation.sql`) has no
automated coverage — verify manually when altering it.
