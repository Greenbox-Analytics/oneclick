"""FX-rate service — Bank of Canada (official) first, fawazahmed0 fallback.

Provides:
  get_rates(db, base, on="latest") -> dict[str, float]
  convert(db, amount, frm, to, on="latest") -> float

Currency conversion uses a **hybrid** strategy:
  1. Bank of Canada Valet API (official, free, no key) when BOTH currencies are
     covered by BoC (all G10 + major EMs: AUD BRL CHF CNY EUR GBP HKD IDR INR
     JPY KRW MXN MYR NOK NZD PEN PLN RUB SAR SEK SGD THB TRY TWD USD VND ZAR
     and CAD itself). Triangulates via CAD: amount * cad_per_unit(from) / cad_per_unit(to).
  2. fawazahmed0 exchange-rate API (covers everything incl. NGN/KES/crypto) when
     BoC does not cover at least one of the two currencies.
  3. If neither source has the rate, returns the amount UNCONVERTED (never raises /
     500s over FX).

Rates are mid-market (no spread). All inputs are normalised to lowercase at the
boundary.

Cache table: fx_rate_snapshots (base text, rate_date date, rates jsonb,
fetched_at timestamptz) — PK (base, rate_date).

  - fawazahmed0 rates cache under base = the lowercase currency code (e.g. 'usd').
  - BoC CAD-per-unit map caches under base = 'cadboc'; rates = {code_lower: cad_per_unit}.

On a total network failure get_rates returns the newest cached snapshot, or {}
if there is no cached data at all. It NEVER raises.

convert returns the amount UNCONVERTED if the target rate is unavailable (FX
fetch failed / unknown code), so read endpoints never 500 over FX. Same-currency
conversions short-circuit and return amount unchanged.
"""

import httpx

# ---------------------------------------------------------------------------
# API URL helpers — fawazahmed0
# ---------------------------------------------------------------------------

_CDN_PRIMARY = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{on}/v1/currencies/{base}.json"
_CDN_FALLBACK = "https://{on}.currency-api.pages.dev/v1/currencies/{base}.json"


def _build_urls(base: str, on: str) -> tuple[str, str]:
    return (
        _CDN_PRIMARY.format(on=on, base=base),
        _CDN_FALLBACK.format(on=on, base=base),
    )


# ---------------------------------------------------------------------------
# API URL helpers — Bank of Canada Valet
# ---------------------------------------------------------------------------

_BOC_OBS = "https://www.bankofcanada.ca/valet/observations/{series}/json?recent=1"
_BOC_CACHE_BASE = "cadboc"  # synthetic base key reused in fx_rate_snapshots: rates = {code_lower: cad_per_unit}


# ---------------------------------------------------------------------------
# Bank of Canada helper
# ---------------------------------------------------------------------------


def _boc_cad_rates(db, codes, on: str = "latest") -> dict[str, float]:
    """Return {code_lower: CAD-per-1-unit} from Bank of Canada for the requested codes
    that BoC actually publishes. 'cad' -> 1.0. Codes BoC doesn't publish are omitted.
    Never raises. Cached (merged) in fx_rate_snapshots under base='cadboc'."""
    from datetime import date

    wanted = {c.lower() for c in codes if c and c.lower() != "cad"}
    out: dict[str, float] = {"cad": 1.0}

    # newest cached BoC map
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

    for c in wanted:
        if c in fetched:
            out[c] = fetched[c]
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_rates(db, base: str, on: str = "latest") -> dict[str, float]:
    """Return a dict of currency_code -> float rate (mid-market, no spread).

    Args:
        db:   Supabase client.
        base: Base currency (e.g. "USD"). Normalised to lowercase internally.
        on:   Date string in YYYY-MM-DD format, or "latest". Defaults to "latest".

    Returns:
        Dict of lowercase currency code -> float exchange rate from *base*.
        Falls back to the newest cached snapshot on total network failure.
        Returns {} if no cached data and both URLs fail.

    Never raises.
    """
    base_lower = base.lower()

    # -----------------------------------------------------------------------
    # Cache check — only possible for a specific date (not "latest"), because
    # "latest" resolves to a server-side date we don't know until we fetch.
    # -----------------------------------------------------------------------
    if on != "latest":
        try:
            res = (
                db.table("fx_rate_snapshots")
                .select("rate_date, rates")
                .eq("base", base_lower)
                .eq("rate_date", on)
                .execute()
            )
            rows = res.data or []
            if rows:
                return rows[0]["rates"]
        except Exception:
            pass  # Cache unavailable — fall through to network fetch

    # -----------------------------------------------------------------------
    # Network fetch — try primary CDN, then fallback
    # -----------------------------------------------------------------------
    primary_url, fallback_url = _build_urls(base_lower, on)
    payload = None

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(primary_url)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        # Primary failed — try fallback
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(fallback_url)
                response.raise_for_status()
                payload = response.json()
        except Exception:
            payload = None

    if payload is not None:
        try:
            resolved_date: str = payload["date"]
            rates: dict[str, float] = payload[base_lower]

            # Upsert into cache
            try:
                db.table("fx_rate_snapshots").upsert(
                    {
                        "base": base_lower,
                        "rate_date": resolved_date,
                        "rates": rates,
                    },
                    on_conflict="base,rate_date",
                ).execute()
            except Exception:
                pass  # Cache write failure is non-fatal

            return rates
        except (KeyError, TypeError):
            pass  # Malformed payload — fall through to stale cache

    # -----------------------------------------------------------------------
    # Total failure — return newest cached snapshot for base currency
    # -----------------------------------------------------------------------
    try:
        res = (
            db.table("fx_rate_snapshots")
            .select("rate_date, rates")
            .eq("base", base_lower)
            .order("rate_date", desc=True)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if rows:
            return rows[0]["rates"]
    except Exception:
        pass

    return {}


def convert(db, amount: float, frm: str, to: str, on: str = "latest") -> float:
    """Convert *amount* from currency *frm* to currency *to*.

    Uses a hybrid strategy:
      1. Bank of Canada Valet API (official) when it covers BOTH currencies —
         triangulates via CAD: amount * cad_per_unit(from) / cad_per_unit(to).
      2. fawazahmed0 exchange-rate API as fallback (covers NGN, KES, crypto, etc.).
      3. Returns amount UNCONVERTED if neither source has the rate — read endpoints
         never 500 over FX.

    Args:
        db:     Supabase client.
        amount: Amount in *frm* currency.
        frm:    Source currency code (case-insensitive).
        to:     Target currency code (case-insensitive).
        on:     Date string or "latest" (passed through to get_rates).

    Returns:
        Converted amount (mid-market, no spread). If the rate is unavailable
        (the FX fetch failed leaving an empty cache, or *to* is an unknown code),
        returns the amount UNCONVERTED so read endpoints never 500 over FX.
    """
    if frm.lower() == to.lower():
        return amount

    # 1) Bank of Canada (official) when it covers BOTH currencies — triangulate via CAD.
    cad = _boc_cad_rates(db, [frm, to], on)
    f, t = frm.lower(), to.lower()
    if f in cad and t in cad and cad[t]:
        return amount * cad[f] / cad[t]

    # 2) Fallback: fawazahmed0 (covers everything incl. NGN/KES/crypto).
    rates = get_rates(db, frm, on)
    rate = rates.get(to.lower())
    if rate is None:
        # 3) Rate unavailable anywhere -> degrade to unconverted (never 500 over FX).
        print(f"[fx] missing rate {frm}->{to}; returning amount unconverted")
        return amount
    return amount * rate
