import { LandingFooter, LandingHeader } from "@/components/landing/LandingSections";

const TOC = [
  ["#overview", "1. Overview"],
  ["#what", "2. What we collect"],
  ["#how", "3. How we use it"],
  ["#share", "4. Who we share it with"],
  ["#retention", "5. Data retention"],
  ["#rights", "6. Your rights"],
  ["#dpa", "7. Data processing (DPA)"],
  ["#contact", "8. Contact"],
];

const Privacy = () => {
  return (
    <div className="landing-page min-h-screen" style={{ background: "var(--bg)", color: "var(--fg)" }}>
      <LandingHeader />

      <section style={{ padding: "96px 32px 32px" }}>
        <div className="doc">
          <p className="eyebrow">Legal</p>
          <h1 className="tighter">Privacy policy</h1>
          <p className="meta">Last updated · May 23, 2026 · Effective immediately</p>

          <div className="placeholder-note">
            <strong>Placeholder copy.</strong> This page sketches what we&apos;ll cover. The final text will be
            reviewed by counsel before launch. If you have questions, email{" "}
            <a href="mailto:privacy@msanii.app">privacy@msanii.app</a>.
          </div>

          <div className="toc">
            {TOC.map(([href, label]) => (
              <a key={href} href={href}>
                {label}
              </a>
            ))}
          </div>

          <h2 id="overview">1. Overview</h2>
          <p>
            Msanii is operated by <strong>Greenbox Analytics Inc.</strong>, a Canadian company headquartered in
            Toronto, Ontario. This policy explains what personal information we collect when you use the Msanii
            platform, why we collect it, how long we keep it, and what choices you have.
          </p>
          <p className="muted">
            We try to write this in plain language. Where a defined term is used (e.g. &ldquo;Personal Information,&rdquo;
            &ldquo;Service&rdquo;), it has the meaning given in our Terms.
          </p>

          <h2 id="what">2. What we collect</h2>
          <h3>Account information</h3>
          <p>
            Email address, name, password hash (we never store passwords in cleartext), organisation, and role.
            Optional: avatar image, time zone, billing details.
          </p>
          <h3>Content you upload</h3>
          <p>
            Contracts, royalty statements, split sheets, project files, audio, notes, and any other material you
            choose to store in Msanii. This content belongs to you.
          </p>
          <h3>Usage data</h3>
          <p>
            Pages visited, features used, errors encountered, approximate IP geolocation, device and browser
            metadata. We use this to debug, improve the product, and detect abuse — not to build advertising
            profiles.
          </p>
          <h3>
            What we do <em>not</em> collect
          </h3>
          <ul>
            <li>We don&apos;t sell your data. Full stop.</li>
            <li>We don&apos;t use third-party advertising trackers or social pixels.</li>
            <li>
              We don&apos;t read or analyse the content of your contracts or files for any purpose other than running
              the feature you invoked.
            </li>
          </ul>

          <h2 id="how">3. How we use it</h2>
          <ul>
            <li>To operate the Service — calculations, AI analysis, exports.</li>
            <li>To bill you (for paid plans).</li>
            <li>
              To notify you of changes, security issues, or new features (you can opt out of marketing email).
            </li>
            <li>To improve the product based on aggregate, de-identified usage patterns.</li>
            <li>To comply with applicable law.</li>
          </ul>

          <h2 id="share">4. Who we share it with</h2>
          <p>We share Personal Information only with:</p>
          <ul>
            <li>
              <strong>Sub-processors</strong> who help us run the Service (hosting, payments, email delivery, AI
              inference). See the full list in our Security &amp; Subprocessors page.
            </li>
            <li>
              <strong>Other users in your organisation</strong>, according to the role-based permissions you
              configure.
            </li>
            <li>
              <strong>Authorities</strong> when legally compelled — we will notify you unless prohibited from doing
              so.
            </li>
          </ul>
          <p>We never sell or rent your information to third parties for their own marketing.</p>

          <h2 id="retention">5. Data retention</h2>
          <p>We keep your data only as long as it is useful to you or required by law.</p>
          <ul>
            <li>
              <strong>Active accounts:</strong> we retain your content for as long as your account is active.
            </li>
            <li>
              <strong>Deleted content:</strong> when you delete a file, project, or artist, it is soft-deleted for
              30 days and then permanently purged from primary storage. Encrypted backups roll off within 90 days.
            </li>
            <li>
              <strong>Closed accounts:</strong> if you close your account, we delete your content within 30 days,
              except where we are legally required to retain it (e.g. invoicing records — kept 7 years per Canadian
              tax law).
            </li>
            <li>
              <strong>Audit logs:</strong> security and access logs are retained for 12 months.
            </li>
            <li>
              <strong>AI inputs:</strong> prompts and documents sent to Zoe are processed in-session and not
              retained by our model providers for training. We log query metadata (not content) for 30 days for
              abuse detection.
            </li>
          </ul>

          <h2 id="rights">6. Your rights</h2>
          <p>Depending on where you live (PIPEDA, GDPR, CCPA, etc.), you have the right to:</p>
          <ul>
            <li>Access the personal information we hold about you</li>
            <li>Correct inaccurate information</li>
            <li>Delete your information (&ldquo;right to be forgotten&rdquo;)</li>
            <li>Export your information in a portable format</li>
            <li>Object to certain processing</li>
            <li>Withdraw consent at any time</li>
          </ul>
          <p>
            To exercise any of these, email <a href="mailto:privacy@msanii.app">privacy@msanii.app</a>. We respond
            within 30 days.
          </p>

          <h2 id="dpa">7. Data processing (DPA)</h2>
          <p>
            For most users, the commitments above are sufficient. <strong>Business and enterprise customers</strong>{" "}
            who need a signed Data Processing Agreement (e.g. to satisfy GDPR Article 28 obligations) can request
            one from <a href="mailto:privacy@msanii.app">privacy@msanii.app</a>.
          </p>
          <p className="muted">
            We don&apos;t host a public DPA today because the obligations differ meaningfully across customers;
            rolling it into Privacy + an on-request signed agreement keeps things simple while still satisfying the
            legal requirement.
          </p>

          <h2 id="contact">8. Contact</h2>
          <p>
            Greenbox Analytics Inc.
            <br />
            Toronto, Ontario, Canada
            <br />
            <a href="mailto:privacy@msanii.app">privacy@msanii.app</a>
          </p>
        </div>
      </section>

      <div style={{ marginTop: 96 }}>
        <LandingFooter />
      </div>
    </div>
  );
};

export default Privacy;
