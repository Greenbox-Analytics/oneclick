import { LandingFooter, LandingHeader } from "@/components/landing/LandingSections";

const TEAM = [
  { initials: "KN", name: "Kenji Niyokindi", role: "Co-founder & CEO", hue: 152 },
  { initials: "YK", name: "Yash Khapre", role: "Co-founder & CTO", hue: 168 },
  { initials: "RY", name: "Romes Young", role: "Co-founder & CIO", hue: 138 },
];

const VALUES = [
  { v: "Correct", l: "Down to the cent. Every calculation gets a paper trail." },
  { v: "Composable", l: "Each tool feeds the next. No re-keying, no exports." },
  { v: "Quiet", l: "Built to disappear into your week. No tabs, no theatre." },
];

function TeamHero() {
  return (
    <section style={{ padding: "120px 32px 32px", textAlign: "center", position: "relative" }}>
      <div
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          pointerEvents: "none",
          background:
            "radial-gradient(700px 320px at 50% -20%, color-mix(in srgb, var(--accent) 14%, transparent), transparent 60%)",
        }}
      />
      <div style={{ position: "relative", maxWidth: 880, margin: "0 auto" }}>
        <p
          style={{
            fontSize: 12,
            letterSpacing: "0.16em",
            textTransform: "uppercase",
            color: "var(--accent)",
            fontWeight: 600,
            margin: "0 0 18px",
          }}
        >
          The team
        </p>
        <h1
          className="tighter"
          style={{
            fontSize: "clamp(56px, 8vw, 88px)",
            lineHeight: 0.98,
            margin: 0,
            fontWeight: 700,
            letterSpacing: "-0.04em",
          }}
        >
          Three people.
          <br />
          One suite.
        </h1>
        <p
          style={{
            maxWidth: 580,
            margin: "28px auto 0",
            fontSize: 19,
            lineHeight: 1.55,
            color: "var(--muted-fg)",
          }}
        >
          We&apos;re a small team of musicians, managers and engineers building the back-office we always wished
          existed. Based in Toronto.
        </p>
      </div>
    </section>
  );
}

function TeamGrid() {
  return (
    <section style={{ padding: "64px 32px 32px" }}>
      <div
        className="lp-team-grid"
        style={{ maxWidth: 1080, margin: "0 auto", display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 28 }}
      >
        {TEAM.map((m) => (
          <article
            key={m.name}
            style={{
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              background: "var(--card)",
              padding: 24,
              boxShadow: "var(--shadow-md)",
            }}
          >
            <div
              style={{
                width: "100%",
                aspectRatio: "1 / 1",
                borderRadius: "calc(var(--radius) - 4px)",
                background: `linear-gradient(135deg, hsl(${m.hue} 40% 92%), hsl(${m.hue} 30% 78%))`,
                display: "grid",
                placeItems: "center",
                marginBottom: 18,
                position: "relative",
                overflow: "hidden",
              }}
            >
              <div
                className="tighter"
                style={{
                  fontSize: 80,
                  fontWeight: 700,
                  color: `hsl(${m.hue} 50% 28%)`,
                  letterSpacing: "-0.04em",
                }}
              >
                {m.initials}
              </div>
              <div
                className="mono"
                style={{
                  position: "absolute",
                  bottom: 10,
                  right: 12,
                  fontSize: 10,
                  color: `hsl(${m.hue} 30% 35%)`,
                  opacity: 0.5,
                }}
              >
                [photo]
              </div>
            </div>
            <div style={{ fontSize: 19, fontWeight: 600, letterSpacing: "-0.01em" }}>{m.name}</div>
            <div style={{ fontSize: 13.5, color: "var(--accent)", fontWeight: 500, marginTop: 2, marginBottom: 16 }}>
              {m.role}
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <a
                href="#"
                style={{
                  fontSize: 12,
                  padding: "5px 12px",
                  borderRadius: 999,
                  border: "1px solid var(--border)",
                  color: "var(--fg)",
                  textDecoration: "none",
                  background: "var(--bg)",
                }}
              >
                LinkedIn
              </a>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function OurMission() {
  return (
    <section style={{ padding: "80px 32px 96px" }}>
      <div style={{ maxWidth: 1080, margin: "0 auto", borderTop: "1px solid var(--border)", paddingTop: 64 }}>
        <div
          className="lp-mission-grid"
          style={{ display: "grid", gridTemplateColumns: "0.85fr 1.15fr", gap: 64, alignItems: "start" }}
        >
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
              Our mission
            </p>
            <h2
              className="tighter"
              style={{
                fontSize: "clamp(32px, 5vw, 48px)",
                lineHeight: 1.04,
                margin: 0,
                fontWeight: 700,
                letterSpacing: "-0.035em",
              }}
            >
              The back-office
              <br />
              independent music
              <br />
              deserves.
            </h2>
          </div>
          <div>
            <p style={{ fontSize: 18, lineHeight: 1.65, margin: 0, color: "var(--fg)" }}>
              Independent artists and the managers who back them lose hours every week to spreadsheets, scattered
              contracts and &ldquo;wait, what did we agree?&rdquo; — work that big labels long ago automated, and that
              nobody else has built tooling for.
            </p>
            <p style={{ fontSize: 16, lineHeight: 1.7, marginTop: 18, color: "var(--muted-fg)" }}>
              We&rsquo;re here to close that gap. One suite — contracts, splits, royalties, and the people they&rsquo;re
              owed to — so the artists and teams we serve can spend their time on what actually matters: the music.
            </p>
            <div
              className="lp-values-grid"
              style={{ marginTop: 28, display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 20 }}
            >
              {VALUES.map((s) => (
                <div key={s.v}>
                  <div className="tighter" style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.02em" }}>
                    {s.v}
                  </div>
                  <div style={{ fontSize: 13, color: "var(--muted-fg)", marginTop: 4, lineHeight: 1.5 }}>{s.l}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

const Team = () => {
  return (
    <div className="landing-page min-h-screen" style={{ background: "var(--bg)", color: "var(--fg)" }}>
      <LandingHeader />
      <TeamHero />
      <TeamGrid />
      <OurMission />
      <LandingFooter />
    </div>
  );
};

export default Team;
