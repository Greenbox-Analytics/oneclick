# OneClick Royalty Extraction Context

## Audience & Intent
OneClick is built for independent artists and managers who are NOT signed to a major
record label (Sony Music Entertainment, Warner Music Group, Universal Music Group, or
any of their subsidiaries / labels). Always extract royalty splits that reflect the
artist's CURRENT independent status.

## Conditional Major-Label / Upstream Agreement Royalties — CRITICAL

Some contracts define a baseline royalty split that APPLIES NOW, plus a conditional
override that would replace it ONLY IF the artist later signs to a major label
(commonly called an "Upstream Agreement", "Label Deal", or similar arrangement).

**Rule:** When such a condition exists, extract ONLY the baseline (independent /
pre-Upstream) percentage. Ignore the major-label override entirely — do not emit
it as a separate royalty share, and do not average, blend, or annotate the two
figures together. The OneClick user is, by definition, not signed to a major; the
conditional override is inapplicable to their actual revenue and would mislead the
calculator.

### How to detect this pattern
Look for clauses with phrases such as:
- "in the event that Artist enters into a subsequent recording agreement with a
  major record label"
- "Upstream Agreement"
- "in lieu of the royalty set forth above" / "in lieu of the foregoing"
- explicit references to Sony Music Entertainment, Warner Music Group, or Universal
  Music Group as the triggering counterparty
- "Notwithstanding the foregoing" followed by a reduced royalty that is tied to a
  label deal or PPD basis
- a "Base Rate" defined only for the post-major-label scenario

### Worked example
Contract excerpt:
> Company agrees to pay Lender Fifty Percent (50%) of Net Receipts in connection
> with the commercial exploitation of the Master.
> Notwithstanding the foregoing, in the event that Artist enters into a subsequent
> recording agreement with a major record label (i.e., Sony Music Entertainment,
> Warner Music Group or Universal Music Group)... in lieu of the royalty set forth
> in the paragraph above, Artist shall pay, or cause Label to pay, Side Artist a
> royalty in the amount of Five Percent (5%) ("Base Rate") PPD.

Correct OneClick extraction for the Side Artist / Lender: **50% of Net Receipts**.
Do NOT extract 5%. Do NOT extract both. Do NOT add a note about the conditional
override in the `terms` field — the override is out of scope for OneClick.

### What this rule does NOT cover
- Standard sales-based escalations, recoupment thresholds, or tier breakpoints that
  do not depend on signing to a major label — extract these normally.
- Conditions tied to non-major-label events (e.g., sync placements, territory
  expansions). Extract those normally.
- Publishing / mechanical royalties unrelated to a major-label recording deal.
