import { CSSProperties, ReactNode, SVGProps } from "react";
import { Link, useNavigate } from "react-router-dom";
import { CreditCard, LogOut, Shield, User } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAuth } from "@/contexts/AuthContext";
import { useIsAdmin } from "@/hooks/useAdmin";
import { TOOLS, TOPICS, type Tool } from "./ToolDemos";

// ──────────────────────────────────────────────────────────────────────
// Inline icon set
// ──────────────────────────────────────────────────────────────────────
function IconArrow(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M5 12h14M13 5l7 7-7 7" />
    </svg>
  );
}

function IconCheck(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M5 13l4 4L19 7" />
    </svg>
  );
}

function IconExternal(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M7 17L17 7M9 7h8v8" />
    </svg>
  );
}

export const LandingIcons = { arrow: IconArrow, check: IconCheck, external: IconExternal };

// ──────────────────────────────────────────────────────────────────────
// Header — sticky, auth-aware, reused on every landing-scope page
// ──────────────────────────────────────────────────────────────────────
const NAV_LINKS: Array<{ label: string; href: string }> = [
  { label: "Tools", href: "/#tools" },
  { label: "Team", href: "/team" },
  { label: "Pricing", href: "/pricing" },
  { label: "Docs", href: "/docs" },
];

export function LandingHeader() {
  const navigate = useNavigate();
  const { user, loading, signOut } = useAuth();
  const { isAdmin } = useIsAdmin();

  return (
    <header
      style={{
        position: "sticky",
        top: 0,
        zIndex: 50,
        borderBottom: "1px solid var(--border)",
        background: "color-mix(in srgb, var(--bg) 80%, transparent)",
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
      }}
    >
      <div
        style={{
          maxWidth: 1200,
          margin: "0 auto",
          padding: "18px 32px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 24,
        }}
      >
        <Link to="/" style={{ display: "flex", alignItems: "center", gap: 10, textDecoration: "none", color: "inherit" }}>
          <BrandMark />
          <span style={{ fontSize: 19, fontWeight: 700, letterSpacing: "-0.02em" }}>Msanii</span>
        </Link>

        <nav className="lp-nav" style={{ display: "flex", alignItems: "center", gap: 6 }}>
          {NAV_LINKS.map((item) => (
            <LinkOrAnchor
              key={item.label}
              href={item.href}
              style={{
                padding: "8px 12px",
                fontSize: 14,
                color: "var(--fg)",
                textDecoration: "none",
                opacity: 0.78,
                fontWeight: 500,
              }}
            >
              {item.label}
            </LinkOrAnchor>
          ))}
        </nav>

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {!loading && user ? (
            <>
              <Link
                to="/dashboard"
                style={{
                  fontSize: 14,
                  color: "var(--fg)",
                  textDecoration: "none",
                  opacity: 0.8,
                  fontWeight: 500,
                  padding: "8px 12px",
                }}
              >
                Dashboard
              </Link>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button
                    type="button"
                    style={{
                      width: 36,
                      height: 36,
                      borderRadius: 999,
                      background: "color-mix(in srgb, var(--primary) 18%, transparent)",
                      color: "var(--primary)",
                      border: "none",
                      cursor: "pointer",
                      fontSize: 14,
                      fontWeight: 600,
                      display: "grid",
                      placeItems: "center",
                    }}
                    aria-label="Account menu"
                  >
                    {(user.email ?? "U")[0].toUpperCase()}
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-56" align="end" forceMount>
                  <DropdownMenuLabel className="font-normal">
                    <p className="text-xs leading-none text-muted-foreground">{user.email}</p>
                  </DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={() => navigate("/profile")}>
                    <User className="mr-2 h-4 w-4" />
                    <span>Profile settings</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => navigate("/subscription")}>
                    <CreditCard className="mr-2 h-4 w-4" />
                    <span>Subscription</span>
                  </DropdownMenuItem>
                  {isAdmin ? (
                    <DropdownMenuItem onClick={() => navigate("/admin/users")}>
                      <Shield className="mr-2 h-4 w-4" />
                      <span>Admin</span>
                    </DropdownMenuItem>
                  ) : null}
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    onClick={async () => {
                      await signOut();
                      navigate("/");
                    }}
                  >
                    <LogOut className="mr-2 h-4 w-4" />
                    <span>Log out</span>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </>
          ) : (
            <>
              <Link
                to="/auth"
                style={{
                  fontSize: 14,
                  color: "var(--fg)",
                  textDecoration: "none",
                  opacity: 0.8,
                  fontWeight: 500,
                  padding: "8px 12px",
                }}
              >
                Sign in
              </Link>
              <Link
                to="/auth"
                style={{
                  fontSize: 14,
                  fontWeight: 600,
                  padding: "9px 16px",
                  borderRadius: 999,
                  background: "var(--primary)",
                  color: "var(--bg)",
                  textDecoration: "none",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 6,
                  whiteSpace: "nowrap",
                }}
              >
                Get started <IconArrow style={{ width: 14, height: 14 }} />
              </Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
}

// ──────────────────────────────────────────────────────────────────────
// Hero
// ──────────────────────────────────────────────────────────────────────
type HeroProps = { primaryHref: string; primaryLabel: string };

export function Hero({ primaryHref, primaryLabel }: HeroProps) {
  return (
    <section style={{ position: "relative", padding: "112px 32px 72px", overflow: "hidden" }}>
      <div
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          pointerEvents: "none",
          background:
            "radial-gradient(800px 360px at 50% -40%, color-mix(in srgb, var(--accent) 16%, transparent), transparent 60%)",
        }}
      />
      <div style={{ maxWidth: 1080, margin: "0 auto", position: "relative", textAlign: "center" }}>
        <h1
          className="tighter"
          style={{
            margin: 0,
            fontSize: "clamp(56px, 8vw, 92px)",
            lineHeight: 0.98,
            fontWeight: 700,
            letterSpacing: "-0.04em",
          }}
        >
          The operating system
          <br />
          for the{" "}
          <span className="lp-hero-gradient" style={{ fontStyle: "italic", fontWeight: 700 }}>
            music business.
          </span>
        </h1>
        <p
          style={{
            maxWidth: 640,
            margin: "28px auto 0",
            fontSize: 19,
            lineHeight: 1.55,
            color: "var(--muted-fg)",
            fontWeight: 400,
          }}
        >
          Streamline your workflow with powerful tools built for music professionals.
        </p>
        <div style={{ display: "flex", gap: 12, justifyContent: "center", marginTop: 36 }}>
          <LinkOrAnchor
            href={primaryHref}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              padding: "14px 22px",
              borderRadius: 999,
              background: "var(--primary)",
              color: "var(--bg)",
              textDecoration: "none",
              fontWeight: 600,
              fontSize: 15,
              boxShadow:
                "0 8px 24px -8px color-mix(in srgb, var(--primary) 50%, transparent)",
            }}
          >
            {primaryLabel} <IconArrow style={{ width: 16, height: 16 }} />
          </LinkOrAnchor>
        </div>
        <div
          style={{
            marginTop: 36,
            display: "flex",
            justifyContent: "center",
            gap: 32,
            fontSize: 13,
            color: "var(--muted-fg)",
            flexWrap: "wrap",
          }}
        >
          {HERO_PROOFS.map((p) => (
            <span key={p} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <IconCheck style={{ width: 14, height: 14, color: "var(--accent)" }} />
              {p}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}

const HERO_PROOFS = ["Free tier available", "No credit card required", "Cancel anytime"];

// ──────────────────────────────────────────────────────────────────────
// Logo strip
// ──────────────────────────────────────────────────────────────────────
const LOGO_STRIP = ["Spotify", "Apple Music", "SoundCloud", "Google Drive", "Slack", "Notion", "Atlassian", "Monday"];

export function LogoStrip() {
  return (
    <section style={{ padding: "8px 32px 32px" }}>
      <div style={{ maxWidth: 1080, margin: "0 auto" }}>
        <p
          style={{
            textAlign: "center",
            fontSize: 12,
            letterSpacing: "0.16em",
            textTransform: "uppercase",
            color: "var(--muted-fg)",
            fontWeight: 500,
            margin: "0 0 24px",
          }}
        >
          Plugged into the tools you already use
        </p>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 24,
            opacity: 0.6,
          }}
        >
          {LOGO_STRIP.map((n) => (
            <div key={n} style={{ fontSize: 18, fontWeight: 600, letterSpacing: "-0.01em" }}>
              {n}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ──────────────────────────────────────────────────────────────────────
// Tools section (topics → tool rows, alternating sides)
// ──────────────────────────────────────────────────────────────────────
type ToolRow = { topic: (typeof TOPICS)[number]; tool: Tool; isTopicFirst: boolean };

function buildRows(): ToolRow[] {
  const byId = new Map(TOOLS.map((t) => [t.id, t]));
  const rows: ToolRow[] = [];
  for (const topic of TOPICS) {
    topic.toolIds.forEach((tid, idx) => {
      const tool = byId.get(tid);
      if (tool) rows.push({ topic, tool, isTopicFirst: idx === 0 });
    });
  }
  return rows;
}

const TOOL_ROWS = buildRows();

export function ToolsSection() {
  return (
    <>
      <section id="tools" style={{ padding: "32px 32px 0" }}>
        <div style={{ maxWidth: 1080, margin: "0 auto", textAlign: "center" }}>
          <p
            style={{
              fontSize: 12,
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              color: "var(--accent)",
              fontWeight: 600,
              margin: "0 0 14px",
            }}
          >
            What&apos;s inside
          </p>
          <h2
            className="tighter"
            style={{ fontSize: "clamp(40px, 6vw, 60px)", lineHeight: 1.02, margin: 0, fontWeight: 700, letterSpacing: "-0.035em" }}
          >
            Everything in one suite.
          </h2>
          <p
            style={{
              maxWidth: 580,
              margin: "24px auto 0",
              fontSize: 17,
              color: "var(--muted-fg)",
              lineHeight: 1.55,
            }}
          >
            A focused tool for each job. Each one runs live below — no signup required.
          </p>
        </div>
      </section>

      <section style={{ padding: "64px 32px 32px" }}>
        <div style={{ maxWidth: 1080, margin: "0 auto", display: "grid", gap: 28 }}>
          {TOOL_ROWS.map((row, i) => {
            const { topic, tool, isTopicFirst } = row;
            const flipped = i % 2 === 1;
            const Demo = tool.Demo;
            return (
              <div key={tool.id}>
                {isTopicFirst ? (
                  <div
                    style={{
                      padding: "40px 0 8px",
                      borderTop: i === 0 ? "none" : "1px solid var(--border)",
                      marginTop: i === 0 ? 0 : 24,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 12,
                        letterSpacing: "0.16em",
                        textTransform: "uppercase",
                        color: "var(--accent)",
                        fontWeight: 600,
                        marginBottom: 10,
                      }}
                    >
                      {topic.label}
                    </div>
                    <h3
                      className="tighter"
                      style={{
                        fontSize: "clamp(28px, 4vw, 40px)",
                        margin: 0,
                        fontWeight: 700,
                        letterSpacing: "-0.035em",
                        lineHeight: 1.05,
                        maxWidth: 760,
                      }}
                    >
                      {topic.headline}
                    </h3>
                  </div>
                ) : null}

                <article
                  id={tool.id}
                  style={{
                    display: "grid",
                    gridTemplateColumns: flipped ? "1.25fr 0.85fr" : "0.85fr 1.25fr",
                    gap: 56,
                    alignItems: "center",
                    padding: 36,
                    borderRadius: "var(--radius-lg)",
                    background: "var(--card)",
                    border: "1px solid var(--border)",
                    boxShadow: "var(--shadow-md)",
                  }}
                  className="lp-tool-row"
                >
                  <div style={{ order: flipped ? 2 : 1 }}>
                    <div
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 8,
                        fontSize: 11,
                        fontWeight: 700,
                        color: "var(--accent)",
                        letterSpacing: "0.14em",
                        textTransform: "uppercase",
                        marginBottom: 14,
                      }}
                    >
                      <span style={{ width: 6, height: 6, borderRadius: 999, background: "var(--accent)" }} />
                      {tool.tagline}
                    </div>
                    <h4
                      className="tighter"
                      style={{
                        fontSize: "clamp(28px, 4vw, 40px)",
                        margin: 0,
                        fontWeight: 700,
                        letterSpacing: "-0.035em",
                        lineHeight: 1.04,
                      }}
                    >
                      {tool.name}
                    </h4>
                    <p style={{ fontSize: 16, lineHeight: 1.6, color: "var(--muted-fg)", marginTop: 14 }}>{tool.blurb}</p>
                    <ul style={{ listStyle: "none", padding: 0, margin: "20px 0 0", display: "grid", gap: 10 }}>
                      {tool.bullets.map((b) => (
                        <li key={b} style={{ display: "flex", alignItems: "start", gap: 10, fontSize: 14 }}>
                          <span
                            style={{
                              width: 18,
                              height: 18,
                              borderRadius: 999,
                              marginTop: 1,
                              background: "color-mix(in srgb, var(--accent) 18%, transparent)",
                              color: "var(--accent)",
                              display: "grid",
                              placeItems: "center",
                              fontSize: 11,
                              fontWeight: 700,
                            }}
                          >
                            ✓
                          </span>
                          <span>{b}</span>
                        </li>
                      ))}
                    </ul>
                    <div style={{ marginTop: 24, display: "flex", gap: 10, flexWrap: "wrap" }}>
                      <LinkOrAnchor
                        href={tool.href}
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          gap: 6,
                          padding: "11px 18px",
                          borderRadius: 999,
                          background: "var(--primary)",
                          color: "var(--bg)",
                          textDecoration: "none",
                          fontWeight: 600,
                          fontSize: 14,
                        }}
                      >
                        {tool.cta} <IconArrow style={{ width: 13, height: 13 }} />
                      </LinkOrAnchor>
                      <LinkOrAnchor
                        href="/docs"
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          gap: 6,
                          padding: "11px 18px",
                          borderRadius: 999,
                          background: "transparent",
                          color: "var(--fg)",
                          border: "1px solid var(--border-strong)",
                          textDecoration: "none",
                          fontWeight: 600,
                          fontSize: 14,
                        }}
                      >
                        Docs
                      </LinkOrAnchor>
                    </div>
                  </div>
                  <div style={{ order: flipped ? 1 : 2 }}>
                    <Demo playing={true} height={420} />
                  </div>
                </article>
              </div>
            );
          })}
        </div>
      </section>
    </>
  );
}

// ──────────────────────────────────────────────────────────────────────
// About + stats
// ──────────────────────────────────────────────────────────────────────
const ABOUT_STATS = [
  { v: "$4.2M", l: "royalties calculated" },
  { v: "120+", l: "artists onboarded" },
  { v: "11", l: "integrations" },
  { v: "2023", l: "founded in Toronto" },
];

export function AboutSection() {
  return (
    <section
      id="about"
      style={{
        padding: "96px 32px",
        background: "var(--muted-bg)",
        borderTop: "1px solid var(--border)",
        borderBottom: "1px solid var(--border)",
      }}
      className="lp-about"
    >
      <div style={{ maxWidth: 1080, margin: "0 auto" }}>
        <div className="lp-about-grid" style={{ display: "grid", gridTemplateColumns: "0.9fr 1.1fr", gap: 80, alignItems: "start" }}>
          <div>
            <p
              style={{
                fontSize: 12,
                letterSpacing: "0.16em",
                textTransform: "uppercase",
                color: "var(--accent)",
                fontWeight: 600,
                margin: "0 0 16px",
              }}
            >
              The Company
            </p>
            <h2
              className="tighter"
              style={{
                fontSize: "clamp(34px, 5vw, 52px)",
                lineHeight: 1.02,
                fontWeight: 700,
                letterSpacing: "-0.035em",
                margin: 0,
              }}
            >
              Built by music
              <br />
              people, for music
              <br />
              people.
            </h2>
          </div>
          <div>
            <p style={{ fontSize: 19, lineHeight: 1.6, margin: 0, color: "var(--fg)" }}>
              Msanii (&quot;artist&quot; in Swahili) is a product of <strong>Greenbox Analytics Inc.</strong> — a
              small Toronto-based team building the back-office that independent artists, managers and labels
              actually need, but rarely have time to build themselves.
            </p>
            <p style={{ fontSize: 16, lineHeight: 1.7, marginTop: 20, color: "var(--muted-fg)" }}>
              We started with one painfully unglamorous question: <em>where did the streaming money go, and
              who&apos;s owed what?</em> Two years later we have answers — and a suite of tools that compose into a
              single source of truth for ownership, contracts, and cash.
            </p>
            <div
              style={{
                marginTop: 36,
                display: "grid",
                gridTemplateColumns: "repeat(4, 1fr)",
                gap: 24,
                borderTop: "1px solid var(--border)",
                paddingTop: 28,
              }}
              className="lp-stats-grid"
            >
              {ABOUT_STATS.map((s) => (
                <div key={s.l}>
                  <div className="tighter" style={{ fontSize: 30, fontWeight: 700, letterSpacing: "-0.03em" }}>
                    {s.v}
                  </div>
                  <div style={{ fontSize: 12.5, color: "var(--muted-fg)", marginTop: 4 }}>{s.l}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// ──────────────────────────────────────────────────────────────────────
// Final CTA
// ──────────────────────────────────────────────────────────────────────
type FinalCTAProps = { primaryHref: string; primaryLabel: string };

export function FinalCTA({ primaryHref, primaryLabel }: FinalCTAProps) {
  return (
    <section style={{ padding: "24px 32px 96px" }}>
      <div style={{ maxWidth: 1080, margin: "0 auto" }}>
        <div
          className="lp-cta"
          style={{
            position: "relative",
            overflow: "hidden",
            borderRadius: "var(--radius-lg)",
            background: "var(--primary)",
            color: "var(--bg)",
            padding: "72px 64px",
            display: "grid",
            gridTemplateColumns: "1.4fr 1fr",
            alignItems: "center",
            gap: 48,
          }}
        >
          <div
            aria-hidden
            style={{
              position: "absolute",
              inset: 0,
              backgroundImage:
                "radial-gradient(circle, color-mix(in srgb, var(--bg) 18%, transparent) 1px, transparent 1px)",
              backgroundSize: "20px 20px",
              opacity: 0.5,
              maskImage: "radial-gradient(ellipse at 80% 50%, black, transparent 70%)",
              WebkitMaskImage: "radial-gradient(ellipse at 80% 50%, black, transparent 70%)",
            }}
          />
          <div style={{ position: "relative" }}>
            <h2 className="tighter" style={{ fontSize: "clamp(36px, 5vw, 56px)", lineHeight: 1.02, margin: 0, fontWeight: 700, letterSpacing: "-0.035em" }}>
              Stop reconciling.
              <br />
              Start releasing.
            </h2>
            <p style={{ fontSize: 17, lineHeight: 1.55, marginTop: 20, opacity: 0.78, maxWidth: 440 }}>
              Set up your first artist project in under two minutes. We&apos;ll import your back-catalog
              statements for free.
            </p>
          </div>
          <div style={{ position: "relative", display: "flex", flexDirection: "column", gap: 10 }}>
            <LinkOrAnchor
              href={primaryHref}
              style={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "18px 24px",
                borderRadius: 999,
                background: "var(--bg)",
                color: "var(--primary)",
                textDecoration: "none",
                fontWeight: 600,
                fontSize: 16,
                whiteSpace: "nowrap",
              }}
            >
              {primaryLabel} <IconArrow style={{ width: 18, height: 18 }} />
            </LinkOrAnchor>
            <LinkOrAnchor
              href="mailto:hello@msanii.app"
              style={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "17px 24px",
                borderRadius: 999,
                background: "transparent",
                color: "var(--bg)",
                border: "1px solid color-mix(in srgb, var(--bg) 30%, transparent)",
                textDecoration: "none",
                fontWeight: 600,
                fontSize: 16,
                whiteSpace: "nowrap",
              }}
            >
              Get in touch <IconExternal style={{ width: 16, height: 16 }} />
            </LinkOrAnchor>
          </div>
        </div>
      </div>
    </section>
  );
}

// ──────────────────────────────────────────────────────────────────────
// Footer
// ──────────────────────────────────────────────────────────────────────
const FOOTER_COLS: Array<[string, Array<[string, string]>]> = [
  [
    "Product",
    [
      ["OneClick", "/tools/oneclick"],
      ["Zoe", "/tools/zoe"],
      ["Split Sheet", "/tools/split-sheet"],
      ["Portfolio", "/portfolio"],
      ["Workspace", "/workspace"],
    ],
  ],
  [
    "Company",
    [
      ["About", "/#about"],
      ["Team", "/team"],
      ["Contact", "mailto:hello@msanii.app"],
    ],
  ],
  ["Resources", [["Docs", "/docs"]]],
  ["Legal", [["Privacy", "/privacy"], ["Security", "/security"]]],
];

export function LandingFooter() {
  return (
    <footer style={{ padding: "64px 32px 40px", borderTop: "1px solid var(--border)", background: "var(--bg)" }}>
      <div style={{ maxWidth: 1080, margin: "0 auto" }}>
        <div
          className="lp-footer-grid"
          style={{ display: "grid", gridTemplateColumns: "1fr repeat(4, 1fr)", gap: 48, paddingBottom: 56 }}
        >
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <BrandMark />
              <span style={{ fontSize: 19, fontWeight: 700, letterSpacing: "-0.02em" }}>Msanii</span>
            </div>
          </div>
          {FOOTER_COLS.map(([title, items]) => (
            <div key={title}>
              <div
                style={{
                  fontSize: 12,
                  letterSpacing: "0.12em",
                  textTransform: "uppercase",
                  color: "var(--muted-fg)",
                  fontWeight: 600,
                  marginBottom: 16,
                }}
              >
                {title}
              </div>
              <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: 10 }}>
                {items.map(([label, href]) => (
                  <li key={label}>
                    <LinkOrAnchor
                      href={href}
                      style={{ fontSize: 14, color: "var(--fg)", opacity: 0.75, textDecoration: "none" }}
                    >
                      {label}
                    </LinkOrAnchor>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        <div
          style={{
            borderTop: "1px solid var(--border)",
            paddingTop: 24,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 16,
            flexWrap: "wrap",
          }}
        >
          <p style={{ fontSize: 12.5, color: "var(--muted-fg)", margin: 0 }}>
            © {new Date().getFullYear()} Greenbox Analytics Inc.
          </p>
          <p
            className="mono"
            style={{ fontSize: 12.5, color: "var(--muted-fg)", margin: 0 }}
          >
            Made in Toronto 🌿
          </p>
        </div>
      </div>
    </footer>
  );
}

// ──────────────────────────────────────────────────────────────────────
// Reusable bits
// ──────────────────────────────────────────────────────────────────────
export function BrandMark() {
  return (
    <div
      style={{
        width: 28,
        height: 28,
        borderRadius: 8,
        background: "var(--primary)",
        color: "var(--bg)",
        display: "grid",
        placeItems: "center",
      }}
    >
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M9 18V5l12-2v13" />
        <circle cx="6" cy="18" r="3" />
        <circle cx="18" cy="16" r="3" />
      </svg>
    </div>
  );
}

type LinkLikeProps = {
  href: string;
  style?: CSSProperties;
  className?: string;
  children: ReactNode;
};

const HTML_TO_ROUTE: Record<string, string> = {
  "team.html": "/team",
  "privacy.html": "/privacy",
  "security.html": "/security",
  "terms.html": "/terms",
};

// Internal routes go through react-router (no full reload, no scroll reset);
// external / hash / mailto links fall back to a plain <a>.
export function LinkOrAnchor({ href, style, className, children }: LinkLikeProps) {
  const resolved = HTML_TO_ROUTE[href] ?? href;
  const isExternal =
    resolved.startsWith("http") ||
    resolved.startsWith("mailto:") ||
    resolved.startsWith("tel:") ||
    resolved.startsWith("#");
  if (isExternal) {
    return (
      <a href={resolved} style={style} className={className}>
        {children}
      </a>
    );
  }
  return (
    <Link to={resolved} style={style} className={className}>
      {children}
    </Link>
  );
}
