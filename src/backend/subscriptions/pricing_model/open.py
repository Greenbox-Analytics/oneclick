#!/usr/bin/env python3
"""Open the Msanii per-tool LLM cost & credit model dashboard in your browser.

    python3 src/backend/subscriptions/pricing_model/open.py
    # or:  task pricing

Self-contained static page — no server, no dependencies, stdlib only. Drag the sliders to
see how per-tool OpenAI cost and suggested credits change. See README.md for what it
models and its caveats.

The page carries its own copy of the model rates (a browser can't import Python).
Rather than trusting a "remember to update both" comment, this entry point diffs
that copy against the authoritative MODEL_RATES in ../ai_pricing.py and refuses
to open a stale dashboard — a stale one gives confidently wrong pricing advice.
"""

import pathlib
import re
import sys
import webbrowser

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from subscriptions.ai_pricing import MODEL_RATES  # noqa: E402

# One MODELS entry: {in:1.25, out:10, cached:0.125, ... name:'gpt-5', ...}
ENTRY = re.compile(r"\{in:\s*([\d.]+),\s*out:\s*([\d.]+),.*?name:'([^']+)'")


def stale_rates(html_text: str) -> list[str]:
    """Rate lines in the page that disagree with MODEL_RATES. Empty == in sync."""
    bad = []
    for tin, tout, model in ENTRY.findall(html_text):
        rate = MODEL_RATES.get(model)
        if rate is None:
            bad.append(f"{model}: in the dashboard but not in MODEL_RATES")
        elif (float(tin), float(tout)) != (rate.input_usd, rate.output_usd):
            bad.append(f"{model}: dashboard {tin}/{tout} vs ai_pricing.py {rate.input_usd}/{rate.output_usd}")
    return bad


html = pathlib.Path(__file__).with_name("index.html").resolve()
if not html.exists():
    raise SystemExit(f"Dashboard not found at {html}")

if problems := stale_rates(html.read_text()):
    raise SystemExit(
        "Dashboard rates are stale — fix the MODELS map in index.html, then reopen:\n  " + "\n  ".join(problems)
    )

webbrowser.open(html.as_uri())
print(f"Opened {html} (rates match ai_pricing.py)")
