import { LandingFooter, LandingHeader } from "@/components/landing/LandingSections";

const PILLARS = [
  {
    badge: "Encryption",
    title: "Encrypted in transit & at rest",
    body:
      "TLS 1.2+ for every connection. AES-256 at rest for both database and object storage. Keys managed by our cloud provider's KMS with strict rotation.",
  },
  {
    badge: "Access",
    title: "Row-Level Security by default",
    body:
      "Every row in our database is gated by Supabase RLS policies. A user can only ever see data their role permits. No “admin sees everything” backdoors.",
  },
  {
    badge: "Auth",
    title: "Modern authentication",
    body:
      "Email/password with hashing, magic links, OAuth (Google), and session tokens with short TTLs. Optional 2FA. Passwords are never stored or logged in cleartext.",
  },
  {
    badge: "Backups",
    title: "Daily encrypted backups",
    body:
      "Point-in-time recovery for the last 7 days; daily snapshots retained 30 days. Backup storage is encrypted with separate keys.",
  },
];

const SUBPROCESSORS: Array<[string, string, string]> = [
  ["Supabase", "Database, auth, storage", "Canada"],
  ["Stripe", "Billing & payments", "US / Ireland"],
  ["Resend", "Transactional email", "US"],
  ["Anthropic / OpenAI", "AI inference (Zoe)", "US"],
  ["Google Cloud", "Drive integration OAuth", "Global"],
  ["Slack", "Slack integration OAuth", "US"],
];

const Security = () => {
  return (
    <div className="landing-page min-h-screen" style={{ background: "var(--bg)", color: "var(--fg)" }}>
      <LandingHeader />

      <section style={{ padding: "96px 32px 32px" }}>
        <div className="doc">
          <p className="eyebrow">Trust</p>
          <h1 className="tighter">Security at Msanii</h1>
          <p className="meta">Last updated · May 23, 2026</p>

          <div className="placeholder-note">
            <strong>Placeholder copy.</strong> This page describes our current security posture. We&apos;ll keep it
            up to date as the platform evolves. Questions? Email{" "}
            <a href="mailto:security@msanii.app">security@msanii.app</a>.
          </div>

          <p style={{ fontSize: 18 }}>
            Music contracts and royalty data are sensitive — sometimes more sensitive than the music itself.
            Here&apos;s how we keep yours safe.
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

          <h2 id="infra">Infrastructure</h2>
          <p>
            Msanii runs on managed cloud infrastructure (Supabase + a Canadian-region cloud provider). We don&apos;t
            operate our own data centres or self-host our primary database — those things are best left to providers
            whose entire business is doing them well.
          </p>
          <ul>
            <li>
              Primary region: <strong>ca-central-1 (Montreal)</strong>. Your data stays in Canada by default.
            </li>
            <li>Network isolation: production has its own VPC with no public database endpoints.</li>
            <li>
              Least-privilege IAM: engineers access production only through audited break-glass procedures.
            </li>
          </ul>

          <h2 id="ai">AI &amp; Zoe</h2>
          <p>
            Zoe (our contract analyst) sends your prompts and the document text you reference to a
            large-language-model provider for inference. We have agreements with these providers that prohibit them
            from training models on your data, and we keep AI request metadata (timing, length, error codes) — never
            the content itself — for 30 days to detect abuse.
          </p>

          <h2 id="retention">Data retention</h2>
          <p>
            Summary (full detail in our <a href="/privacy">Privacy Policy</a>):
          </p>
          <ul>
            <li>Active account content: kept as long as your account is active.</li>
            <li>Deleted content: soft-deleted for 30 days, then permanently purged. Backups roll off within 90 days.</li>
            <li>Closed accounts: deleted within 30 days, except invoicing records (7 years, Canadian tax law).</li>
            <li>Audit logs: 12 months.</li>
          </ul>

          <h2 id="incident">Incident response</h2>
          <p>
            If we discover a security incident affecting your data, we will notify you within{" "}
            <strong>72 hours</strong> of confirmation — sooner where required by law. Our process: contain,
            investigate, notify, remediate, post-mortem.
          </p>

          <h2 id="subprocessors">Sub-processors</h2>
          <p>We share information with these vendors strictly to run the Service:</p>
          <table className="subprocessor-table">
            <thead>
              <tr>
                <th>Vendor</th>
                <th>Purpose</th>
                <th>Region</th>
              </tr>
            </thead>
            <tbody>
              {SUBPROCESSORS.map(([v, p, r]) => (
                <tr key={v}>
                  <td>{v}</td>
                  <td>{p}</td>
                  <td>{r}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted">
            We update this list as our stack changes. To get notified of changes, email{" "}
            <a href="mailto:security@msanii.app">security@msanii.app</a>.
          </p>

          <h2 id="responsible-disclosure">Responsible disclosure</h2>
          <p>
            Found something? We&apos;re a small team that takes security reports seriously. Please email{" "}
            <a href="mailto:security@msanii.app">security@msanii.app</a> with details. We&apos;ll acknowledge within
            2 business days and aim to triage within 5. We do not currently offer a paid bug bounty but will credit
            you publicly (with permission) for valid reports.
          </p>

          <h2 id="roadmap">On the roadmap</h2>
          <ul>
            <li>SOC 2 Type II (target: 2027)</li>
            <li>SSO / SAML for enterprise customers</li>
            <li>Customer-managed encryption keys (CMEK)</li>
            <li>Region selection for non-Canadian customers</li>
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
