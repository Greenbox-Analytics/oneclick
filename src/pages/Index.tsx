import { useAuth } from "@/contexts/AuthContext";
import { Hero, LandingFooter, LandingHeader, LogoStrip, ToolsOverview } from "@/components/landing/LandingSections";

const Index = () => {
  const { user } = useAuth();
  const ctaHref = user ? "/dashboard" : "/auth";
  const ctaLabel = user ? "Go to dashboard" : "Start free";

  return (
    <div
      className="landing-page min-h-screen"
      style={{ position: "relative", background: "var(--bg)", color: "var(--fg)" }}
    >
      {/* Green framing glow — top + both sides (from the index.html design) */}
      <div
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          pointerEvents: "none",
          zIndex: 0,
          background: [
            "radial-gradient(900px 440px at 50% 0%, color-mix(in srgb, var(--accent) 16%, transparent), transparent 60%)",
            "radial-gradient(540px 95% at -4% 42%, color-mix(in srgb, var(--accent) 17%, transparent), transparent 60%)",
            "radial-gradient(540px 95% at 104% 42%, color-mix(in srgb, var(--accent) 17%, transparent), transparent 60%)",
          ].join(", "),
        }}
      />
      <div style={{ position: "relative", zIndex: 1 }}>
        <LandingHeader />
        <Hero primaryHref={ctaHref} primaryLabel={ctaLabel} />
        <LogoStrip />
        <ToolsOverview />
        <LandingFooter />
      </div>
    </div>
  );
};

export default Index;
