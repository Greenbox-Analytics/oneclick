# Credit & licensing economics — pricing dashboard

A self-contained, interactive page for pricing decisions: turn one dial (the price of a
credit) and see it fall through per-action COGS, plan grants, the price each plan must
charge to hold its margin, the cost of carrying a free user, and what an enterprise pool
earns per credit.

## Open it

```bash
task pricing
# …or, without go-task:
python3 src/backend/subscriptions/pricing_model/open.py
# …or just open src/backend/subscriptions/pricing_model/index.html in any browser
```

No server and no dependencies — `index.html` is fully standalone (inline CSS + JS).
`open.py` refuses to open it if the model rates have drifted from `ai_pricing.py`.

## The four tabs

1. **Cost per action.** Price per credit, target markup, contract length, cache-hit rate →
   modeled COGS per action, the credits each action *should* cost, and the margin the
   *shipped* `credit_prices` values actually earn. Lower the credit price and the suggested
   credits rise to hold the dollar margin; the live prices don't move until you change the
   DB, which is what the Margin column shows you.
2. **Plans & margins.** Grants and prices per tier, utilization per tier, fixed infra per
   user, target margin → cost per free user, margin per user per plan, the price each plan
   would need at *full* grant consumption, and a blended P&L across a user-count mix.
3. **Licensing (Enterprise).** The org structure diagram (solid = in the code, dashed amber
   = drawn but not built), a manipulable dispersal → caps example, margin per entry point
   (each credit pack, the monthly dispersal, pay-per-use, and the grants bundled into
   Basic/Pro), contract P&L across a committed term, and a utilization sensitivity table.

   The model is a **monthly dispersal**: the org negotiates a credit volume per month
   (e.g. 10,000 cr), pays monthly against a committed term, and the credits land in ONE
   pool. Admins set a per-user monthly **cap**; users spend straight from the pool and are
   cut off at their own cap, then ask for a raise. Because a cap is a ceiling rather than a
   reservation, caps may deliberately **overcommit** the dispersal — the page shows that
   ratio and flags the point where demand outruns the pool.
4. **Model rates.** The OpenAI per-1M rates and where the modeled token counts come from.

## What's live vs modeled vs unbuilt

- **Live** (values shipped in the code): credit prices (`credit_prices`), plan grants
  (`tier_entitlements`), plan prices (`lib/tiers.ts`), pack prices (`credit_packs`), the
  pay-per-use rate (`CREDIT_OVERAGE_USD`), the org activation floor
  (`ENTERPRISE_MIN_INITIAL_CREDITS`).
- **Modeled**: token counts per call. `ai_usage_log` now records real spend via the
  `TrackedOpenAI` proxy — reconcile against it before locking prices in.
- **Not built** (drawn dashed in the licensing tab): the 12-month contract with monthly
  payments and monthly dispersal, per-user caps on a shared pool (the code moves credits
  into per-member seat wallets instead), org-level pay-as-you-go (overage is personal-plan
  only; a dry seat 402s), any org subscription object to upgrade or cancel, cancellation
  grace/migration terms, time-gated permissions, and role-gated payment drafting.

## Checks

The pure model lives in `<script id="pricing-math">` with no DOM access, so it can be
exercised headlessly:

```bash
cd src/backend/subscriptions/pricing_model
node -e "eval(require('fs').readFileSync('index.html','utf8').split('<script id=\"pricing-math\">')[1].split('</'+'script>')[0]); console.log(PricingMath.selfCheck())"
```

`selfCheck()` asserts the directional invariants the page rests on — halving the credit
price doubles the credits per action, a cache hit costs less than a cold run, free-tier
margin is negative, an early cancellation still owes the remaining term, and so on. It
caught a real bug: `0.10 * 3 / 0.02` is `15.000000000000002` in floating point, so a naive
`Math.ceil` was adding a phantom credit to every suggested price.

## Caveats

- **Modeled, not measured.** Token counts are estimated from prompt sizes and the call
  graph. Treat this as a planning tool and reconcile against the OpenAI invoice.
- **Utilization is the biggest unknown.** Margins move more with the utilization sliders
  than with any price you set — unspent credits are pure margin. Guessing high makes
  everything look profitable.
- **Rollover is a margin cliff.** Unspent dispersal that never expires is *deferred* COGS,
  not profit: the org banks credits and can burn a year of them in one month, which removes
  the margin floor the dispersal model otherwise gives you. The toggle in the licensing tab
  shows both.
- **Rates and model IDs are hard-coded** in `index.html` (the `MODELS` map), because a
  static page can't import Python. `open.py` diffs them against
  `subscriptions/ai_pricing.py` (`MODEL_RATES` — the authoritative table) and refuses to
  open the dashboard while they disagree, so update `index.html` when OpenAI pricing or
  the models change — e.g. the extractor (`OPENAI_LLM_MODEL_LARGE`, default `gpt-5.2`),
  Zoe (`OPENAI_LLM_MODEL`, default `gpt-5-mini`), or the OneClick payable-column
  detector (`gpt-5.4-mini`).
