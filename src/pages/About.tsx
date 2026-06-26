import { LandingFooter, LandingHeader } from "@/components/landing/LandingSections";

// ──────────────────────────────────────────────────────────────────────
// About page — ported from the Claude Design `about.html`, re-skinned onto
// the shared `.landing-page` token scope so it adapts to light + dark mode.
// Structure mirrors the source App(): Story → Values → CTA.
// ──────────────────────────────────────────────────────────────────────

function OurStory() {
  return (
    <section style={{ padding: "120px 32px 96px", position: "relative" }}>
      <div
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          pointerEvents: "none",
          background:
            "radial-gradient(960px 460px at 50% -12%, color-mix(in srgb, var(--accent) 26%, transparent), transparent 62%)",
        }}
      />
      <div style={{ maxWidth: 760, margin: "0 auto", position: "relative" }}>
        <h2
          className="tighter"
          style={{
            fontSize: 44,
            lineHeight: 1.05,
            margin: "0 0 16px",
            fontWeight: 700,
            letterSpacing: "-0.035em",
            maxWidth: 640,
            textIndent: "-0.46em",
          }}
        >
          &ldquo;Being an artist today is a full-time business. So let&rsquo;s start treating it like one.&rdquo;
        </h2>
        <p
          style={{
            fontSize: 12,
            letterSpacing: "0.16em",
            textTransform: "uppercase",
            color: "var(--accent)",
            fontWeight: 600,
            margin: "0 0 36px",
          }}
        >
          Our purpose
        </p>
        <p style={{ fontSize: 20, lineHeight: 1.65, margin: "0 0 24px", color: "var(--fg)", textAlign: "justify" }}>
          An artist&rsquo;s career in the modern day entails much more than just the creation of music. The music
          business and all its moving parts can be extremely daunting, especially without an infrastructure in place
          surrounding one&rsquo;s artistry. There is a considerable amount of data to ingest, revenue to calculate,
          splits to keep track of, collaborators to pay, paperwork to interpret and metadata to record. Msanii was
          established to address the pain points and alleviate the extensive work effort that is required in the
          additional core areas of an artist&rsquo;s business.
        </p>
        <p style={{ fontSize: 20, lineHeight: 1.65, margin: "0 0 24px", color: "var(--fg)", textAlign: "justify" }}>
          Msanii was built to simplify and streamline this work. Our team has spent years alongside artists and
          creatives, with deep roots in data analysis at major labels and in software engineering. We know both the
          realities of the music industry and the tools it takes to improve them. The goal is simple: take the workload
          off the artist and their team, so more of their time goes to creating, collaborating, and growing their
          career.
        </p>
        <p style={{ fontSize: 20, lineHeight: 1.65, margin: "0 0 24px", color: "var(--fg)", textAlign: "justify" }}>
          Our tools weren&rsquo;t built in isolation, they were built to work together. Each one represents a step in the
          life cycle of a release, connected end to end so your splits, contracts, royalties, and payouts all speak the
          same language. That&rsquo;s the difference: not a collection of features, but a single ecosystem that gives
          your team real operational structure.
        </p>
        <p style={{ fontSize: 20, lineHeight: 1.65, margin: 0, color: "var(--fg)", textAlign: "justify" }}>
          Our team has leveraged AI the{" "}
          <strong>
            <em>right</em>
          </strong>{" "}
          way. We&rsquo;ve applied AI with intention, automating the tedious and repetitive while sharpening the business
          decisions that matter. From automatic royalty calculations and contract assistance to instant split sheet
          creation, intuitive project management tools, and centralized metadata storage, Msanii gives modern music
          professionals the infrastructure to stay organized and run every side of their business with confidence.
        </p>
      </div>
    </section>
  );
}

const VALUES: Array<{ v: string; l: string }> = [
  { v: "Solution-oriented", l: "We start with the problem in front of you and build the most direct path through it." },
  { v: "Efficiency", l: "Every workflow is tuned to save hours, not minutes. Less busywork, more music." },
  { v: "Transparency", l: "Clear numbers, visible splits, and a paper trail behind every calculation." },
  { v: "Accuracy", l: "Correct down to the cent, so your figures always hold up to scrutiny." },
  {
    v: "Infrastructure",
    l: "One connected foundation your whole catalog and team can rely on, with every tool feeding the next so nothing gets re-keyed.",
  },
];

function Values() {
  return (
    <section style={{ padding: "72px 32px 96px" }}>
      <div style={{ maxWidth: 880, margin: "0 auto" }}>
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
          What we stand for
        </p>
        <h2
          className="tighter"
          style={{
            fontSize: 40,
            lineHeight: 1.05,
            margin: "0 0 24px",
            fontWeight: 700,
            letterSpacing: "-0.035em",
            maxWidth: 560,
          }}
        >
          The values we build on.
        </h2>
        <div>
          {VALUES.map((s, i) => (
            <div
              key={s.v}
              className="lp-about-value-row"
              style={{
                display: "grid",
                gridTemplateColumns: "64px 230px 1fr",
                alignItems: "baseline",
                gap: 24,
                padding: "24px 0",
                borderTop: i === 0 ? "none" : "1px solid var(--border)",
              }}
            >
              <div className="mono" style={{ fontSize: 14, color: "var(--accent)", fontWeight: 600 }}>
                0{i + 1}
              </div>
              <div className="tighter" style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.02em" }}>
                {s.v}
              </div>
              <div style={{ fontSize: 15, lineHeight: 1.6, color: "var(--muted-fg)" }}>{s.l}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Divider() {
  return (
    <div style={{ maxWidth: 1080, margin: "0 auto", padding: "0 32px" }}>
      <hr style={{ border: "none", borderTop: "1px solid var(--border)", margin: 0 }} />
    </div>
  );
}

const About = () => {
  return (
    <div className="landing-page min-h-screen" style={{ background: "var(--bg)", color: "var(--fg)" }}>
      <LandingHeader />
      <OurStory />
      <Divider />
      <Values />
      <LandingFooter />
    </div>
  );
};

export default About;
