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

> **Superseded in part.** `credit_prices` is no longer what a run costs — it is the
> pre-flight ESTIMATE the balance check reserves against. The actual charge is metered
> from the tokens the run burned: `credits_for_cost()` in `ai_pricing.py` applies
> `CREDIT_MARKUP` (default 3.0, the same dial as tab 1's markup slider) to the real
> OpenAI cost and divides by `CREDIT_OVERAGE_USD`. So tab 1's "suggested credits" column
> is now what the code actually charges at the modeled token counts, and the "Margin"
> column no longer drifts from it — margin is held by construction at every request size.
> Reconcile the modeled token counts against `ai_usage_log` and `credit_ledger.metadata`
> (which now carries `input_tokens` / `output_tokens` / `cost_usd` per charge).

- **Live** (values shipped in the code): credit estimates (`credit_prices`), plan grants
  (`tier_entitlements`), plan prices (`lib/tiers.ts`), pack prices (`credit_packs`), the
  pay-per-use rate (`CREDIT_OVERAGE_USD`), the org activation floor
  (`ENTERPRISE_MIN_INITIAL_CREDITS`).
- **Modeled**: token counts per call. `ai_usage_log` now records real spend via the
  `TrackedOpenAI` proxy — reconcile against it before locking prices in.
- **Built since this page was written**: the monthly dispersal into one org pool (expiring
  each period) and per-member caps on that pool, enforced inside `debit_credits`.
- **Not built** (drawn dashed in the licensing tab): the commercial wrapper — no org
  subscription object, so the 12-month term, the monthly payment, the cancellation grace
  period and the "remaining months owed" rule live in the signed contract rather than the
  app. Also unbuilt: org-level pay-as-you-go (overage is personal-plan only; a member past
  their cap asks for a raise and a dry pool needs an admin to buy credits), time-gated
  permissions, and role-gated payment drafting.

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
