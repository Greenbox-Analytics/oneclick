/**
 * Keeps the org-invite link's `?email=` out of PostHog.
 *
 * The invite email links to /orgs/invite/{token}?email=… so /auth can prefill
 * the address. The claim page strips it from the address bar on mount, but
 * that is too late for analytics: posthog-js fires the first `$pageview`
 * synchronously inside `init()` (before React mounts) with the raw
 * `location.href`, and it also persists that URL as `$initial_person_info` /
 * `$session_entry_url` and replays it on every later event. So the scrub has
 * to happen at the event boundary — this is the `before_send` hook.
 *
 * Pure and dependency-free so it can be unit-tested without a token.
 */
import type { BeforeSendFn, CaptureResult } from "posthog-js";

const SENSITIVE_PARAM = "email";

/** Drop the `email` query param from a URL string. Non-URLs pass through. */
export function stripEmailParam(url: string): string {
  if (!url.includes(`${SENSITIVE_PARAM}=`)) return url;
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return url;
  }
  if (!parsed.searchParams.has(SENSITIVE_PARAM)) return url;
  parsed.searchParams.delete(SENSITIVE_PARAM);
  // `searchParams.delete` leaves `?` behind when the list is now empty.
  if (![...parsed.searchParams.keys()].length) parsed.search = "";
  return parsed.toString();
}

type Bag = Record<string, unknown>;

/** Shallow-scrub every string value of a flat object; returns the same
 * reference when nothing changed so untouched events stay identical. */
function scrubStrings(bag: unknown): unknown {
  if (!bag || typeof bag !== "object" || Array.isArray(bag)) return bag;
  let out: Bag | null = null;
  for (const [key, value] of Object.entries(bag as Bag)) {
    if (typeof value !== "string") continue;
    const cleaned = stripEmailParam(value);
    if (cleaned === value) continue;
    out ??= { ...(bag as Bag) };
    out[key] = cleaned;
  }
  return out ?? bag;
}

/** The nested bags posthog-js hangs URLs off. `$initial_person_info` is
 * `{ r: referrer, u: url }`, persisted at first load. */
const NESTED_KEYS = ["$set", "$set_once", "$initial_person_info"] as const;

function scrubProperties(props: unknown): unknown {
  if (!props || typeof props !== "object") return props;
  let out = scrubStrings(props) as Bag;
  for (const key of NESTED_KEYS) {
    const nested = (out as Bag)[key];
    const cleaned = scrubStrings(nested);
    if (cleaned === nested) continue;
    if (out === props) out = { ...(props as Bag) };
    out[key] = cleaned;
  }
  return out;
}

export const scrubEmailFromEvent: BeforeSendFn = (cr: CaptureResult | null) => {
  if (!cr) return cr;
  const properties = scrubProperties(cr.properties);
  const $set = scrubStrings(cr.$set);
  const $set_once = scrubStrings(cr.$set_once);
  if (properties === cr.properties && $set === cr.$set && $set_once === cr.$set_once) return cr;
  return {
    ...cr,
    properties: properties as CaptureResult["properties"],
    ...($set !== cr.$set ? { $set: $set as CaptureResult["$set"] } : {}),
    ...($set_once !== cr.$set_once ? { $set_once: $set_once as CaptureResult["$set_once"] } : {}),
  };
};
