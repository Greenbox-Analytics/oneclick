#!/usr/bin/env python3
"""Open the Msanii per-tool LLM cost & credit model dashboard in your browser.

    python3 src/backend/subscriptions/pricing_model/open.py
    # or:  task pricing

Self-contained static page — no server, no dependencies, stdlib only. Drag the sliders to
see how per-tool OpenAI cost and suggested credits change. See README.md for what it
models and its caveats.
"""

import pathlib
import webbrowser

html = pathlib.Path(__file__).with_name("index.html").resolve()
if not html.exists():
    raise SystemExit(f"Dashboard not found at {html}")
webbrowser.open(html.as_uri())
print(f"Opened {html}")
