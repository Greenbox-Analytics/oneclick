# LLM cost & credit model — pricing dashboard

A self-contained, interactive page that models what each Msanii tool spends on OpenAI per
use and how many credits to charge to cover it. Use it to reason about per-tool credit
allocation for the subscriptions/credits work.

## Open it

```bash
task pricing
# …or, without go-task:
python3 src/backend/subscriptions/pricing_model/open.py
# …or just open src/backend/subscriptions/pricing_model/index.html in any browser
```

No server and no dependencies — `index.html` is fully standalone (inline CSS + JS).

## What it shows

Broken down per tool / integration:

- **The two plans (live).** Free vs Pro with the exact caps the app enforces today, read
  from the `tier_entitlements` table (3/3/50 artists/projects/tasks, 1 GB, 5 split-sheets,
  1 OneClick run/mo on Free; unlimited + Zoe/Registry/Slack on Pro at $25/mo · $250/yr).
- **Per-tool cards** — Free vs Pro behaviour for every tool and integration, tagged AI
  (credit-metered) or plan-gated.
- **AI credit cost per use** — the LLM call graph, model tiers and token estimates behind
  Zoe / OneClick / Registry, with OneClick **cold vs warm** (warm = extraction served from
  the shared `contract_parse_cache`) blended by a cache-hit-rate slider, expressed as
  suggested credits.
- **Credits instead of AI caps (proposed)** — models replacing the on/off flags + run caps
  with a monthly credit balance: what each plan's allowance buys per AI tool, and the max
  OpenAI COGS it exposes.
- **Storage economics (proposed)** — Pro includes 100 GB (covered by the $25 base); above
  it, per-GB overage or add-on bundles, priced against Supabase's ~$0.0213/GB so you can
  see your margin.

## Live vs proposed

- **Live** (enforced today): the Free/Pro plan caps and gating.
- **Modeled**: all AI token costs (the backend doesn't log `response.usage` yet).
- **Proposed** (not built): the credit ledger/allowance and the storage overage/add-ons —
  there is no `credits` column and no storage overage in the code today.

## Caveats

- **Modeled, not measured.** Token counts are estimated from prompt sizes and the call
  graph; the backend does not yet log `response.usage`. Treat this as a planning tool and
  reconcile against the OpenAI invoice before locking prices.
- **Rates and model IDs are hard-coded** in `index.html` (the `MODELS` map and the rates
  footer). Update them there when OpenAI pricing or the models change — e.g. the extractor
  (`OPENAI_LLM_MODEL_LARGE`, default `gpt-5.2`), Zoe (`OPENAI_LLM_MODEL`, default
  `gpt-5-mini`), or the OneClick payable-column detector (`gpt-5.4-mini`).
