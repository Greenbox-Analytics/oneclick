/**
 * Guards for supabase/config.toml + supabase/templates/, which
 * `supabase config push` applies to the hosted auth project on merge.
 *
 * A push sends CLI DEFAULTS for any recognised key missing from the file, and
 * enable_confirmations defaults to false — pushing without it once
 * auto-confirmed every signup in production. So the keys we rely on must be
 * pinned, every template the file names must exist and still carry the
 * confirmation link, and no secret may be committed inline.
 */
import { describe, it, expect } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

const ROOT = process.cwd();
const config = readFileSync(path.join(ROOT, "supabase/config.toml"), "utf8");

const templatePaths = [...config.matchAll(/content_path\s*=\s*"([^"]+)"/g)].map((m) => m[1]);

describe("supabase/config.toml", () => {
  it("pins email confirmations on so a config push cannot silently auto-confirm signups", () => {
    expect(config).toMatch(/^\[auth\.email\]\s*$/m);
    expect(config).toMatch(/^enable_confirmations\s*=\s*true\s*$/m);
    expect(config).toMatch(/^enable_signup\s*=\s*true\s*$/m);
  });

  it("pins the redirect origins the app signs up from", () => {
    expect(config).toMatch(/^site_url\s*=\s*"https:\/\/www\.msanii-beta\.com"/m);
    expect(config).toContain('"http://localhost:8080/**"');
    // The dev site must be allowlisted or Auth silently swaps the dev
    // emailRedirectTo for site_url and dev confirmation links open PROD.
    expect(config).toContain('"https://msanii-dev.vercel.app/**"');
  });

  it("pins the hourly email rate limit above the CLI default of 2", () => {
    // Unpinned, a push re-applies the CLI default (2/hour project-wide) and
    // the third auth email in an hour fails with "email rate limit exceeded".
    expect(config).toMatch(/^\[auth\.rate_limit\]\s*$/m);
    const match = config.match(/^email_sent\s*=\s*(\d+)\s*$/m);
    expect(match).not.toBeNull();
    expect(Number(match![1])).toBeGreaterThanOrEqual(30);
  });

  it("reads the SMTP password from the environment, never inline", () => {
    expect(config).toMatch(/^pass\s*=\s*"env\(RESEND_API_KEY\)"\s*$/m);
    expect(config).not.toMatch(/re_[A-Za-z0-9]{10,}/);
  });

  it("names a template for every auth email", () => {
    for (const key of ["confirmation", "magic_link", "recovery", "email_change", "invite"]) {
      expect(config).toContain(`[auth.email.template.${key}]`);
    }
    expect(templatePaths).toHaveLength(5);
  });
});

describe("supabase/templates", () => {
  it.each(templatePaths)("%s exists, links the confirmation URL, and has no raw link dump", (rel) => {
    const abs = path.join(ROOT, rel);
    expect(existsSync(abs)).toBe(true);
    const html = readFileSync(abs, "utf8");
    expect(html).toMatch(/href="\{\{ \.ConfirmationURL \}\}"/);
    expect(html.toLowerCase()).not.toContain("copy this link");
  });
});
