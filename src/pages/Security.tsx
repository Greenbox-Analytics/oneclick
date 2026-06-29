import { LandingFooter, LandingHeader } from "@/components/landing/LandingSections";

const PILLARS = [
  {
    badge: "Encryption",
    title: "Your data is encrypted",
    body:
      "Everything you keep in Msanii is encrypted — both while it travels to us and while it sits in our systems. Your contracts and royalty figures are never left in the open.",
  },
  {
    badge: "Privacy",
    title: "Only you see your data",
    body:
      "Your account is walled off from everyone else's. There's no “admin sees everything” switch — your splits, contracts, and numbers are visible only to you and the people you choose to share them with.",
  },
  {
    badge: "Sign-in",
    title: "Secure sign-in",
    body:
      "Sign in with an email and password, a one-time link, or your Google account. Passwords are always protected and never stored in plain text, and optional two-step verification adds another layer.",
  },
  {
    badge: "Backups",
    title: "Backed up automatically",
    body:
      "Your work is backed up automatically and often, so nothing is lost if something goes wrong. Those backups are encrypted too.",
  },
];

const PARTNERS: Array<[string, string]> = [
  ["Supabase", "Stores your data and handles sign-in"],
  ["Stripe", "Billing and payments"],
  ["Resend", "Sends account and notification emails"],
  ["Anthropic / OpenAI", "Powers Zoe, our contract assistant"],
  ["Google", "Google Drive connection"],
  ["Slack", "Slack connection"],
];

const Security = () => {
  return (
    <div className="landing-page min-h-screen" style={{ background: "var(--bg)", color: "var(--fg)" }}>
      <LandingHeader />

      <section style={{ padding: "96px 32px 32px" }}>
        <div className="doc">
          <p className="eyebrow">Trust</p>
          <h1 className="tighter">Privacy &amp; Security</h1>
          <p className="meta">Last updated · June 28, 2026</p>

          <p style={{ fontSize: 18, marginTop: 28 }}>
            Music contracts and royalty data are sensitive — sometimes more sensitive than the music itself. Here&apos;s
            how we keep yours safe, in plain language. Questions? Email{" "}
            <a href="mailto:security@msanii.app">security@msanii.app</a>.
          </p>

          <div className="pillar-grid">
            {PILLARS.map((p) => (
              <div key={p.title} className="pillar">
                <span className="badge">{p.badge}</span>
                <h3>{p.title}</h3>
                <p>{p.body}</p>
              </div>
            ))}
          </div>

          <h2 id="infra">Built on trusted foundations</h2>
          <p>
            We run Msanii on established, reputable cloud providers — the same kind of infrastructure relied on by banks
            and healthcare companies — rather than servers we manage ourselves. Access to the live system is tightly
            restricted to a small number of people, and only when it&apos;s needed to keep the service running.
          </p>

          <h2 id="ai">AI &amp; Zoe</h2>
          <p>
            Zoe, our contract assistant, reads the documents you point it at so it can answer your questions. We have
            agreements with our AI providers that prohibit them from training their models on your data, and we only
            keep basic activity records — never the contents of your contracts — to keep the service reliable.
          </p>

          <h2 id="retention">How long we keep your data</h2>
          <p>
            The short version (full detail lives in our <a href="/privacy">Privacy Policy</a>):
          </p>
          <ul>
            <li>Your content stays with you for as long as your account is active.</li>
            <li>When you delete something, it&apos;s removed shortly after and clears out of backups soon afterward.</li>
            <li>If you close your account, we delete your data, keeping only what the law requires us to (such as invoices).</li>
          </ul>

          <h2 id="incident">If something goes wrong</h2>
          <p>
            If we ever discover a security problem that affects your data, we&apos;ll let you know within{" "}
            <strong>72 hours</strong> of confirming it — sooner where the law requires. Our approach is simple: contain
            it, investigate, tell you, fix it, and learn from it.
          </p>

          <h2 id="partners">The partners we work with</h2>
          <p>To run Msanii, we rely on a handful of trusted companies, each for a specific job:</p>
          <table className="subprocessor-table">
            <thead>
              <tr>
                <th>Company</th>
                <th>What they help with</th>
              </tr>
            </thead>
            <tbody>
              {PARTNERS.map(([name, purpose]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{purpose}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted">
            We keep this list current as our toolkit changes. To be notified of updates, email{" "}
            <a href="mailto:security@msanii.app">security@msanii.app</a>.
          </p>

          <h2 id="responsible-disclosure">Found a problem?</h2>
          <p>
            We&apos;re a small team that takes security seriously. If you spot something that doesn&apos;t look right,
            please email <a href="mailto:security@msanii.app">security@msanii.app</a> with the details. We&apos;ll get
            back to you within a couple of business days, and we&apos;re happy to credit you publicly (with your
            permission) for valid reports.
          </p>

          <h2 id="roadmap">What&apos;s next</h2>
          <p>Security is never finished. A few things we&apos;re working toward:</p>
          <ul>
            <li>An independent, third-party security review (SOC 2).</li>
            <li>Single sign-on for larger teams and organizations.</li>
            <li>More controls over how your data is stored and shared.</li>
          </ul>
        </div>
      </section>

      <div style={{ marginTop: 96 }}>
        <LandingFooter />
      </div>
    </div>
  );
};

export default Security;
