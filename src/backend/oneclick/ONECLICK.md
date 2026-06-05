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
