"""FX-rate service — Bank of Canada only.

Provides:
  convert(db, amount, frm, to, on_missing="amount") -> float | None

Currency conversion triangulates via CAD using the Bank of Canada Valet API
(official, free, no key). Covered currencies include all G10 + major EMs:
AUD BRL CHF CNY EUR GBP HKD IDR INR JPY KRW MXN MYR NOK NZD PEN PLN RUB SAR
SEK SGD THB TRY TWD USD VND ZAR and CAD itself.

Rates are mid-market (no spread). All inputs are normalised to lowercase at the
boundary.

Cache table: fx_rate_snapshots (base text, rate_date date, rates jsonb,
fetched_at timestamptz) — PK (base, rate_date).

  - BoC CAD-per-unit map caches under base = 'cadboc'; rates = {code_lower: cad_per_unit}.

On a total network failure _boc_cad_rates returns the newest cached snapshot, or
{} if there is no cached data at all. It NEVER raises.

convert returns None (when on_missing="none") or the unconverted amount (default)
if the rate is unavailable, so aggregators can exclude unconvertible buckets.
Same-currency conversions short-circuit and return amount unchanged.
"""

import time

import httpx

# ---------------------------------------------------------------------------
# In-process TTL memo — FX data is daily; 10 min in-process reuse is safe
# ---------------------------------------------------------------------------

_RATE_MEMO: dict = {}
_RATE_MEMO_TTL = 600.0  # seconds


def _clear_rate_memo() -> None:
    """Clear the in-process FX memo (used by tests + available for manual reset)."""
    _RATE_MEMO.clear()


def _memo_get(key):
    hit = _RATE_MEMO.get(key)
    if hit is not None and (time.monotonic() - hit[0]) < _RATE_MEMO_TTL:
        return hit[1]
    return None


def _memo_put(key, value):
    _RATE_MEMO[key] = (time.monotonic(), value)


# ---------------------------------------------------------------------------
# API URL helpers — Bank of Canada Valet
# ---------------------------------------------------------------------------

_BOC_OBS = "https://www.bankofcanada.ca/valet/observations/{series}/json?recent=1"
_BOC_CACHE_BASE = "cadboc"  # synthetic base key reused in fx_rate_snapshots: rates = {code_lower: cad_per_unit}


# ---------------------------------------------------------------------------
# Bank of Canada helper
# ---------------------------------------------------------------------------


def _boc_cad_rates(db, codes) -> dict[str, float]:
    """Return {code_lower: CAD-per-1-unit} from Bank of Canada for the requested codes
    that BoC actually publishes. 'cad' -> 1.0. Codes BoC doesn't publish are omitted.
    Never raises. Cached (merged) in fx_rate_snapshots under base='cadboc'."""
    from datetime import date

    wanted = {c.lower() for c in codes if c and c.lower() != "cad"}
    out: dict[str, float] = {"cad": 1.0}

    # In-process memo check: if the full memo map already covers all requested codes,
    # serve from it without touching Supabase or the network.
    boc_mkey = ("boc", "latest")
    memo_map = _memo_get(boc_mkey)
    if memo_map is not None:
        needed = wanted | {"cad"}
        if needed.issubset(memo_map.keys()) or all(c in memo_map for c in wanted):
            result = {"cad": 1.0}
            for c in wanted:
                if c in memo_map:
                    result[c] = memo_map[c]
            return result

    # newest cached BoC map (Supabase)
    cached: dict = {}
    try:
        res = (
            db.table("fx_rate_snapshots")
            .select("rate_date, rates")
            .eq("base", _BOC_CACHE_BASE)
            .order("rate_date", desc=True)
            .limit(1)
            .execute()
        )
        if res.data:
            cached = res.data[0].get("rates") or {}
    except Exception:
        cached = {}

    missing = {c for c in wanted if c not in cached}
    fetched = dict(cached)
    if missing:
        series_csv = ",".join(f"FX{c.upper()}CAD" for c in sorted(missing))
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(_BOC_OBS.format(series=series_csv))
                resp.raise_for_status()
                payload = resp.json()
            observations = payload.get("observations") or []
            for c in missing:
                s = f"FX{c.upper()}CAD"
                val = None
                for row in reversed(observations):
                    cell = row.get(s)
                    if cell and cell.get("v") not in (None, ""):
                        val = float(cell["v"])
                        break
                if val is not None:
                    fetched[c] = val
            try:
                db.table("fx_rate_snapshots").upsert(
                    {"base": _BOC_CACHE_BASE, "rate_date": date.today().isoformat(), "rates": fetched},
                    on_conflict="base,rate_date",
                ).execute()
            except Exception:
                pass
        except Exception:
            fetched = cached  # network failure -> use stale cache only

    # Store the full fetched map in the in-process memo so subsequent calls within
    # the same request (or within TTL) skip Supabase + network entirely.
    _memo_put(boc_mkey, fetched)

    for c in wanted:
        if c in fetched:
            out[c] = fetched[c]
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert(db, amount: float, frm: str, to: str, on_missing: str = "amount") -> float | None:
    """Convert *amount* from currency *frm* to currency *to* using Bank of Canada rates.

    Triangulates via CAD: amount * cad_per_unit(from) / cad_per_unit(to).

    Args:
        db:         Supabase client.
        amount:     Amount in *frm* currency.
        frm:        Source currency code (case-insensitive).
        to:         Target currency code (case-insensitive).
        on_missing: What to do when the rate is unavailable:
                    "amount" (default) — return the amount unconverted (never raises / 500s).
                    "none"            — return None so the caller can exclude the bucket.

    Returns:
        Converted amount (mid-market, no spread), or None if on_missing="none" and
        the rate is unavailable. Same-currency conversions always return amount unchanged.
    """
    if frm.lower() == to.lower():
        return amount
    cad = _boc_cad_rates(db, [frm, to])
    f, t = frm.lower(), to.lower()
    if f in cad and t in cad and cad[t]:
        return amount * cad[f] / cad[t]
    if on_missing == "none":
        return None
    print(f"[fx] no BoC rate {frm}->{to}; returning amount unconverted")
    return amount
