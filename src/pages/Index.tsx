import { useAuth } from "@/contexts/AuthContext";
import {
  AboutSection,
  FinalCTA,
  Hero,
  LandingFooter,
  LandingHeader,
  LogoStrip,
  ToolsSection,
} from "@/components/landing/LandingSections";

const Index = () => {
  const { user } = useAuth();
  const ctaHref = user ? "/dashboard" : "/auth";
  const ctaLabel = user ? "Go to dashboard" : "Start free";

  return (
    <div className="landing-page min-h-screen" style={{ background: "var(--bg)", color: "var(--fg)" }}>
      <LandingHeader />
      <Hero primaryHref={ctaHref} primaryLabel={ctaLabel} />
      <LogoStrip />
      <ToolsSection />
      <AboutSection />
      <FinalCTA primaryHref={ctaHref} primaryLabel={user ? "Go to dashboard" : "Get started for free"} />
      <LandingFooter />
    </div>
  );
};

export default Index;
