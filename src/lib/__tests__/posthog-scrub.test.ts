import { describe, expect, it } from "vitest";
import { scrubEmailFromEvent, stripEmailParam } from "../posthogScrub";

const LINK = "https://app.msanii.test/orgs/invite/tok?email=a%40b.c&signup=1";

describe("stripEmailParam", () => {
  it("removes only the email param", () => {
    expect(stripEmailParam(LINK)).toBe("https://app.msanii.test/orgs/invite/tok?signup=1");
  });

  it("leaves no dangling ? when email was the only param", () => {
    expect(stripEmailParam("https://x.test/orgs/invite/tok?email=a%40b.c")).toBe("https://x.test/orgs/invite/tok");
  });

  it("keeps the hash", () => {
    expect(stripEmailParam("https://x.test/p?email=a%40b.c#frag")).toBe("https://x.test/p#frag");
  });

  it("returns other urls unchanged", () => {
    expect(stripEmailParam("https://x.test/p?signup=1")).toBe("https://x.test/p?signup=1");
    expect(stripEmailParam("https://x.test/p")).toBe("https://x.test/p");
  });

  it("returns a non-url string unchanged", () => {
    expect(stripEmailParam("not a url email=a")).toBe("not a url email=a");
  });
});

describe("scrubEmailFromEvent", () => {
  it("passes null through", () => {
    expect(scrubEmailFromEvent(null)).toBeNull();
  });

  it("scrubs every url-bearing property posthog attaches, without mutating the input", () => {
    const event = {
      uuid: "u",
      event: "$pageview",
      properties: {
        $current_url: LINK,
        $session_entry_url: LINK,
        $referrer: LINK,
        $pathname: "/orgs/invite/tok",
        environment: "prod",
        $set: { $current_url: LINK },
        $set_once: { $initial_current_url: LINK, $initial_referrer: LINK },
        $initial_person_info: { r: LINK, u: LINK },
      },
      $set: { $current_url: LINK },
      $set_once: { $initial_current_url: LINK },
    };
    const snapshot = JSON.stringify(event);

    const out = scrubEmailFromEvent(event as never)!;

    expect(JSON.stringify(event)).toBe(snapshot);
    const clean = "https://app.msanii.test/orgs/invite/tok?signup=1";
    expect(out.properties.$current_url).toBe(clean);
    expect(out.properties.$session_entry_url).toBe(clean);
    expect(out.properties.$referrer).toBe(clean);
    expect(out.properties.$set.$current_url).toBe(clean);
    expect(out.properties.$set_once.$initial_current_url).toBe(clean);
    expect(out.properties.$set_once.$initial_referrer).toBe(clean);
    expect(out.properties.$initial_person_info).toEqual({ r: clean, u: clean });
    expect(out.$set!.$current_url).toBe(clean);
    expect(out.$set_once!.$initial_current_url).toBe(clean);
    expect(out.properties.$pathname).toBe("/orgs/invite/tok");
    expect(out.properties.environment).toBe("prod");
    expect(JSON.stringify(out)).not.toContain("email=");
  });

  it("returns the same event when there is nothing to scrub", () => {
    const event = { uuid: "u", event: "x", properties: { $current_url: "https://x.test/p", n: 1 } };
    expect(scrubEmailFromEvent(event as never)).toEqual(event);
  });
});
