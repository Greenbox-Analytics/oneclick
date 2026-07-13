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

- **Cost per use** for each tool (SplitSheet, Zoe, Registry Add Work, OneClick): the exact
  LLM call graph, model tiers, token estimates, `$/use`, and suggested credits.
- OneClick **cold vs warm** runs — *warm* = the contract extraction served from the shared
  `contract_parse_cache` — plus a **blended** steady-state row driven by a cache-hit-rate
  slider.
- Adjustable assumptions: contract length, contracts per run, retail price per credit,
  markup on COGS, and parse-cache hit rate.
- **Cost levers** (what's shipped, what's next) and a plan to bill on measured
  `response.usage` instead of estimates.

## Caveats

- **Modeled, not measured.** Token counts are estimated from prompt sizes and the call
  graph; the backend does not yet log `response.usage`. Treat this as a planning tool and
  reconcile against the OpenAI invoice before locking prices.
- **Rates and model IDs are hard-coded** in `index.html` (the `MODELS` map and the rates
  footer). Update them there when OpenAI pricing or the models change — e.g. the extractor
  (`OPENAI_LLM_MODEL_LARGE`, default `gpt-5.2`), Zoe (`OPENAI_LLM_MODEL`, default
  `gpt-5-mini`), or the OneClick payable-column detector (`gpt-5.4-mini`).
