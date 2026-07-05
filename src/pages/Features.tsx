import { useEffect } from "react";
import { useLocation } from "react-router-dom";

import { LandingFooter, LandingHeader, ToolShowcase } from "@/components/landing/LandingSections";

// ──────────────────────────────────────────────────────────────────────
// Features (public "Tools") page — ported from the Claude Design `tools.html`.
// Reuses the live tool-demo showcase (ToolShowcase) extracted from the landing
// page, fronted by the design's hero. Re-skinned onto the shared `.landing-page`
// token scope so it adapts to light + dark mode.
// ──────────────────────────────────────────────────────────────────────

function ToolsHero() {
  return (
    <section style={{ padding: "120px 32px 8px", textAlign: "center", position: "relative" }}>
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
      <div style={{ position: "relative", maxWidth: 1000, margin: "0 auto" }}>
        <h1
          className="tighter"
          style={{
            fontSize: "clamp(30px, 5.4vw, 46px)",
            lineHeight: 1.0,
            margin: 0,
            fontWeight: 700,
            letterSpacing: "-0.04em",
          }}
        >
          Intuitive Tools, For Tedious Tasks
        </h1>
        <p
          style={{
            maxWidth: 660,
            margin: "26px auto 0",
            fontSize: 18,
            lineHeight: 1.6,
            color: "var(--muted-fg)",
            textAlign: "justify",
          }}
        >
          Each tool is designed to enhance your business processes as it pertains to several aspects of an
          artist&rsquo;s workflow. Through rigorous assessment and practical experience, each tool is intentionally
          created to alleviate painpoints, so creativity and artistry can return to the forefront.
        </p>
      </div>
    </section>
  );
}

const Features = () => {
  // Deep links from the homepage tool grid (/features#oneclick, …) — scroll the
  // matching tool section into view once it has rendered.
  const { hash } = useLocation();
  useEffect(() => {
    if (!hash) return;
    const raf = requestAnimationFrame(() => {
      document.getElementById(hash.slice(1))?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    return () => cancelAnimationFrame(raf);
  }, [hash]);

  return (
    <div className="landing-page min-h-screen" style={{ background: "var(--bg)", color: "var(--fg)" }}>
      <LandingHeader />
      <ToolsHero />
      <ToolShowcase />
      <LandingFooter />
    </div>
  );
};

export default Features;
