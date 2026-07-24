import { CSSProperties, ReactNode, SVGProps } from "react";
import { Link, useNavigate } from "react-router-dom";
import { LogOut, Shield, User } from "lucide-react";
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

function IconChevron(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}

export const LandingIcons = { arrow: IconArrow, check: IconCheck, external: IconExternal };

// ──────────────────────────────────────────────────────────────────────
// Header — sticky, auth-aware, reused on every landing-scope page
// ──────────────────────────────────────────────────────────────────────
type NavLink = { label: string; href: string };
type NavItem = NavLink | { label: string; children: NavLink[] };

const NAV_LINKS: NavItem[] = [
  { label: "Tools", href: "/features" },
  {
    label: "Company",
    children: [
      { label: "About", href: "/about" },
      { label: "Team", href: "/team" },
    ],
  },
  { label: "Pricing", href: "/pricing" },
  { label: "Docs", href: "/docs" },
];

const NAV_LINK_STYLE: CSSProperties = {
  padding: "8px 12px",
  fontSize: 14,
  color: "var(--fg)",
  textDecoration: "none",
  opacity: 0.78,
  fontWeight: 500,
};

const NAV_DROPDOWN_TRIGGER_STYLE: CSSProperties = {
  ...NAV_LINK_STYLE,
  background: "none",
  border: "none",
  cursor: "pointer",
  fontFamily: "inherit",
  display: "inline-flex",
  alignItems: "center",
  gap: 4,
};

// Module-scoped so it isn't re-created on each header render
// (vercel-react-best-practices: rerender-no-inline-components). Uses real
// <Link>s inside the menu for accessible, right-click-friendly navigation.
function NavDropdown({ label, items }: { label: string; items: NavLink[] }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button type="button" style={NAV_DROPDOWN_TRIGGER_STYLE}>
          {label}
          <IconChevron style={{ width: 14, height: 14, opacity: 0.7 }} />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        {items.map((it) => (
          <DropdownMenuItem key={it.label} asChild>
            <Link to={it.href} className="cursor-pointer">
              {it.label}
            </Link>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

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
          {NAV_LINKS.map((item) =>
            "children" in item ? (
              <NavDropdown key={item.label} label={item.label} items={item.children} />
            ) : (
              <LinkOrAnchor key={item.label} href={item.href} style={NAV_LINK_STYLE}>
                {item.label}
              </LinkOrAnchor>
            ),
          )}
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
                    <span>Profile & billing</span>
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
          The Music Business
          <br />
          <span className="lp-hero-gradient" style={{ fontStyle: "italic", fontWeight: 700 }}>
            Simplified.
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
          Streamline your workflow with innovative tools built for music professionals.
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
// Logo strip — infinite horizontal marquee of integration logos.
// Pure CSS animation (no RAF) so it doesn't compete with the demos.
// ──────────────────────────────────────────────────────────────────────
type LogoEntry = { name: string; src: string };

const LOGOS: readonly LogoEntry[] = [
  { name: "Spotify", src: "/spotify.svg" },
  { name: "Apple Music", src: "/apple_music.png" },
  { name: "SoundCloud", src: "/soundcloud.png" },
  { name: "Google Drive", src: "/drive.webp" },
  { name: "Slack", src: "/slack.png" },
  { name: "PayPal", src: "/paypal.png" },
];

// Doubled so the keyframes can translate -50% and loop seamlessly.
const MARQUEE_LOGOS = [...LOGOS, ...LOGOS];

export function LogoStrip() {
  return (
    <section style={{ padding: "8px 32px 40px" }}>
      <div style={{ maxWidth: 1080, margin: "0 auto" }}>
        <p
          style={{
            textAlign: "center",
            fontSize: 12,
            letterSpacing: "0.16em",
            textTransform: "uppercase",
            color: "var(--muted-fg)",
            fontWeight: 500,
            margin: "0 0 28px",
          }}
        >
          Plugged into the tools you already use
        </p>
        <div className="lp-marquee" aria-label="Integrations">
          <div className="lp-marquee-track">
            {MARQUEE_LOGOS.map((logo, i) => (
              <img
                key={`${logo.name}-${i}`}
                src={logo.src}
                alt={logo.name}
                className="lp-marquee-logo"
                loading="eager"
                decoding="async"
                draggable={false}
                aria-hidden={i >= LOGOS.length}
              />
            ))}
          </div>
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

// ToolShowcase — topic-grouped tool rows with live demos. Extracted from the old
// ToolsSection so it can live on the dedicated /features page (the design's
// "ToolsShowcase"). The compact landing summary is ToolsOverview, below.
export function ToolShowcase() {
  return (
      <section style={{ padding: "24px 32px 32px" }}>
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
                    {tool.tagline && (
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
                    )}
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
  );
}

// ──────────────────────────────────────────────────────────────────────
// ToolsOverview — compact landing summary (no live demos) that redirects to
// the dedicated /features page.
// ──────────────────────────────────────────────────────────────────────
export function ToolsOverview() {
  return (
    <section id="tools" style={{ padding: "32px 32px 72px" }}>
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
          Your whole operation in one place.
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
          A focused tool for each job. Royalties, contracts, splits, metadata and projects, all sharing one source of
          truth to streamline your infrastructure.
        </p>
        <div
          className="lp-tools-grid"
          style={{ marginTop: 44, display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}
        >
          {TOOLS.map((tool) => (
            <Link key={tool.id} to={`/features#${tool.id}`} className="lp-tool-card">
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                <span className="tighter" style={{ fontSize: 18, fontWeight: 700, letterSpacing: "-0.02em" }}>
                  {tool.name}
                </span>
                <IconArrow style={{ width: 15, height: 15, color: "var(--accent)" }} />
              </div>
              {tool.tagline && (
                <span
                  style={{
                    marginTop: 8,
                    fontSize: 11,
                    fontWeight: 700,
                    color: "var(--accent)",
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                  }}
                >
                  {tool.tagline}
                </span>
              )}
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

// ──────────────────────────────────────────────────────────────────────
// (The former embedded "About" section now lives as a dedicated page at
// /about — see src/pages/About.tsx.)

// ──────────────────────────────────────────────────────────────────────
// (Final CTA removed — the homepage now closes with the clickable tool grid
//  rendered by ToolsOverview, above.)
// ──────────────────────────────────────────────────────────────────────
// ──────────────────────────────────────────────────────────────────────
// Footer
// ──────────────────────────────────────────────────────────────────────
const FOOTER_COLS: Array<[string, Array<[string, string]>]> = [
  [
    "Product",
    [
      ["OneClick", "/docs?section=oneclick"],
      ["Zoe", "/docs?section=zoe"],
      ["Split Sheet", "/docs?section=split-sheet"],
      ["Metadata Registry", "/docs?section=rights-registry"],
      ["Portfolio", "/docs?section=portfolio"],
      ["Workspace", "/docs?section=workspace"],
    ],
  ],
  [
    "Company",
    [
      ["About", "/about"],
      ["Team", "/team"],
      ["Contact", "mailto:hello@msanii.app"],
    ],
  ],
  ["Resources", [["Docs", "/docs"]]],
  ["Legal", [["Privacy & Security", "/security"]]],
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
