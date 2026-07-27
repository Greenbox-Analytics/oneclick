import { useRef, useState, type CSSProperties, type SVGProps } from "react";

import { LandingFooter, LandingHeader, LandingIcons, LinkOrAnchor } from "@/components/landing/LandingSections";
import { API_URL, apiFetch } from "@/lib/apiFetch";

// ──────────────────────────────────────────────────────────────────────
// Contact page — ported from the Claude Design `Contact.html`, re-skinned
// onto the shared `.landing-page` token scope so it adapts to light + dark.
// Structure mirrors the source App(): Hero → (ContactForm | Sidebar).
//
// The prototype faked submission (client-generated reference, nothing sent).
// Here the form POSTs to /contact-submissions, which records the row and
// notifies ops; the reference shown on success comes back from the server.
// ──────────────────────────────────────────────────────────────────────

type Mode = "ticket" | "message";

const CI = {
  ticket: (p: SVGProps<SVGSVGElement>) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}>
      <path d="M3 9a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v1a2 2 0 0 0 0 4v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1a2 2 0 0 0 0-4Z" />
      <path d="M13 7v10" strokeDasharray="2 3" />
    </svg>
  ),
  chat: (p: SVGProps<SVGSVGElement>) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}>
      <path d="M21 15a2 2 0 0 1-2 2H8l-5 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2Z" />
    </svg>
  ),
  clock: (p: SVGProps<SVGSVGElement>) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </svg>
  ),
  book: (p: SVGProps<SVGSVGElement>) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}>
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5Z" />
    </svg>
  ),
  paperclip: (p: SVGProps<SVGSVGElement>) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}>
      <path d="M21.44 11.05 12.25 20.24a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
    </svg>
  ),
  file: (p: SVGProps<SVGSVGElement>) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}>
      <path d="M14 3v4a1 1 0 0 0 1 1h4" />
      <path d="M17 21H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7l5 5v11a2 2 0 0 1-2 2Z" />
    </svg>
  ),
  x: (p: SVGProps<SVGSVGElement>) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}>
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  ),
};

// Client-side mirrors of the server's caps in contact/router.py. The server is
// authoritative — these exist so the user hears about a bad file immediately
// instead of after an upload round-trip.
const MAX_FILES = 3;
const MAX_BYTES = 2.5 * 1024 * 1024;
const ACCEPT =
  "image/*,.pdf,.doc,.docx,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document";
const OK_EXT = /\.(png|jpe?g|gif|webp|heic|pdf|docx?)$/i;

const PRODUCTS = [
  "OneClick",
  "Zoe (AI assistant)",
  "Split Sheet",
  "Rights Registry",
  "Royalty Payments",
  "Account & Billing",
  "Something else",
];
const TOPICS = ["General inquiry", "Partnership", "Product feedback"];

function fmtSize(b: number) {
  return b < 1024 * 1024 ? `${(b / 1024).toFixed(0)} KB` : `${(b / 1024 / 1024).toFixed(1)} MB`;
}

type FileLike = { name: string; size: number; type: string };

/**
 * Apply the attachment rules to a batch of picked/dropped files.
 *
 * Pure so the rules can be tested directly — and so the error message is
 * derived in one pass rather than accumulated inside a state updater.
 * Reports the last rejection reason, matching the design's single-line error.
 */
export function filterAttachments<T extends FileLike>(
  existingCount: number,
  incoming: T[]
): { accepted: T[]; error: string } {
  let error = "";
  const ok: T[] = [];
  for (const f of incoming) {
    if (!OK_EXT.test(f.name) && !f.type.startsWith("image/")) {
      error = "Only images and Word/PDF files are supported.";
      continue;
    }
    if (f.size > MAX_BYTES) {
      error = `${f.name} is over the 2.5 MB limit.`;
      continue;
    }
    ok.push(f);
  }
  const room = Math.max(0, MAX_FILES - existingCount);
  if (ok.length > room) {
    error = `You can attach up to ${MAX_FILES} files.`;
  }
  return { accepted: ok.slice(0, room), error };
}

const PAIR_STYLE: CSSProperties = { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 };

function Field({ label, optional, children }: { label: string; optional?: boolean; children: React.ReactNode }) {
  return (
    <label style={{ display: "block" }}>
      <span className="field-label">
        {label}
        {optional && <span className="opt">(optional)</span>}
      </span>
      {children}
    </label>
  );
}

function Segmented({
  options,
  value,
  onChange,
}: {
  options: Array<{ id: Mode; label: string; icon: (p: SVGProps<SVGSVGElement>) => JSX.Element }>;
  value: Mode;
  onChange: (m: Mode) => void;
}) {
  return (
    <div
      style={{
        display: "flex",
        gap: 4,
        padding: 4,
        background: "var(--muted-bg)",
        border: "1px solid var(--border)",
        borderRadius: 999,
      }}
    >
      {options.map((o) => {
        const active = o.id === value;
        const Icon = o.icon;
        return (
          <button
            key={o.id}
            type="button"
            onClick={() => onChange(o.id)}
            aria-pressed={active}
            style={{
              flex: 1,
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 8,
              padding: "11px 14px",
              borderRadius: 999,
              border: "none",
              cursor: "pointer",
              fontFamily: "inherit",
              fontSize: 14,
              fontWeight: 600,
              letterSpacing: "-0.01em",
              background: active ? "var(--primary)" : "transparent",
              color: active ? "var(--primary-fg)" : "var(--muted-fg)",
              boxShadow: active ? "var(--shadow-sm)" : "none",
              transition: "all .18s",
            }}
          >
            <Icon style={{ width: 16, height: 16 }} />
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

function Success({
  mode,
  refId,
  email,
  onReset,
}: {
  mode: Mode;
  refId: string;
  email: string;
  onReset: () => void;
}) {
  return (
    <div style={{ padding: "48px 40px", textAlign: "center" }}>
      <div
        style={{
          width: 60,
          height: 60,
          borderRadius: 999,
          background: "var(--accent-soft)",
          color: "var(--primary)",
          display: "grid",
          placeItems: "center",
          margin: "0 auto 22px",
        }}
      >
        <LandingIcons.check style={{ width: 30, height: 30 }} />
      </div>
      <h3 className="tighter" style={{ fontSize: 28, fontWeight: 700, margin: "0 0 10px", letterSpacing: "-0.03em" }}>
        {mode === "ticket" ? "Ticket received." : "Message sent."}
      </h3>
      <p style={{ fontSize: 15.5, lineHeight: 1.6, color: "var(--muted-fg)", margin: "0 auto", maxWidth: 400 }}>
        {mode === "ticket" ? (
          <>
            Our support team is on it. We&rsquo;ll reply to{" "}
            <strong style={{ color: "var(--fg)" }}>{email}</strong> — usually within one business day.
          </>
        ) : (
          <>
            Thanks for reaching out. We&rsquo;ll get back to{" "}
            <strong style={{ color: "var(--fg)" }}>{email}</strong> shortly.
          </>
        )}
      </p>
      {mode === "ticket" && (
        <div
          className="mono"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 10,
            marginTop: 24,
            padding: "10px 16px",
            background: "var(--muted-bg)",
            border: "1px solid var(--border)",
            borderRadius: 999,
            fontSize: 13.5,
          }}
        >
          <span style={{ color: "var(--muted-fg)" }}>Reference</span>
          <span style={{ fontWeight: 600, color: "var(--primary)" }}>{refId}</span>
        </div>
      )}
      <div style={{ marginTop: 28 }}>
        <button
          type="button"
          onClick={onReset}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            fontFamily: "inherit",
            fontSize: 14,
            fontWeight: 600,
            color: "var(--primary)",
          }}
        >
          {mode === "ticket" ? "Submit another ticket" : "Send another message"} →
        </button>
      </div>
    </div>
  );
}

function AttachField({ files, setFiles }: { files: File[]; setFiles: React.Dispatch<React.SetStateAction<File[]>> }) {
  const ref = useRef<HTMLInputElement>(null);
  const [err, setErr] = useState("");

  const add = (list: FileList | null) => {
    const { accepted, error } = filterAttachments(files.length, Array.from(list || []));
    setErr(error);
    if (accepted.length) setFiles((prev) => [...prev, ...accepted]);
  };

  const remove = (i: number) => setFiles((prev) => prev.filter((_, j) => j !== i));

  return (
    <div>
      <span className="field-label">
        Attachments<span className="opt">(optional)</span>
      </span>
      <div
        onClick={() => ref.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          add(e.dataTransfer.files);
        }}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 10,
          padding: "18px 16px",
          cursor: "pointer",
          textAlign: "center",
          border: "1px dashed var(--border-strong)",
          borderRadius: "var(--radius-sm)",
          background: "var(--muted-bg)",
          color: "var(--muted-fg)",
          transition: "border-color .15s",
        }}
      >
        <CI.paperclip style={{ width: 18, height: 18, color: "var(--accent)" }} />
        <span style={{ fontSize: 13.5 }}>
          <strong style={{ color: "var(--fg)", fontWeight: 600 }}>Click to upload</strong> or drag &amp; drop
        </span>
      </div>
      <input
        ref={ref}
        type="file"
        multiple
        accept={ACCEPT}
        onChange={(e) => {
          add(e.target.files);
          // Clear the input so re-picking the same file still fires onChange.
          e.target.value = "";
        }}
        style={{ display: "none" }}
      />
      <p style={{ fontSize: 12, color: "var(--muted-fg)", margin: "8px 2px 0" }}>
        Images, PDF, or Word — up to 2.5 MB each.
      </p>
      {err && (
        <p style={{ fontSize: 12.5, color: "hsl(0 60% 45%)", margin: "8px 2px 0", fontWeight: 500 }}>{err}</p>
      )}
      {files.length > 0 && (
        <div style={{ display: "grid", gap: 8, marginTop: 12 }}>
          {files.map((f, i) => (
            <div
              key={`${f.name}-${i}`}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "9px 12px",
                background: "var(--bg)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-sm)",
              }}
            >
              <CI.file style={{ width: 16, height: 16, color: "var(--primary)", flexShrink: 0 }} />
              <span
                style={{
                  fontSize: 13,
                  fontWeight: 500,
                  flex: 1,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {f.name}
              </span>
              <span className="mono" style={{ fontSize: 11.5, color: "var(--muted-fg)" }}>
                {fmtSize(f.size)}
              </span>
              <button
                type="button"
                onClick={() => remove(i)}
                title="Remove"
                aria-label={`Remove ${f.name}`}
                style={{
                  display: "grid",
                  placeItems: "center",
                  width: 22,
                  height: 22,
                  border: "none",
                  background: "none",
                  cursor: "pointer",
                  color: "var(--muted-fg)",
                  padding: 0,
                }}
              >
                <CI.x style={{ width: 15, height: 15 }} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

type Errors = Partial<Record<"name" | "email" | "subject" | "message", boolean>>;

function ContactForm() {
  const [mode, setMode] = useState<Mode>("ticket");
  const [files, setFiles] = useState<File[]>([]);
  const [errs, setErrs] = useState<Errors>({});
  const [submitting, setSubmitting] = useState(false);
  const [sendError, setSendError] = useState("");
  const [done, setDone] = useState<{ mode: Mode; refId: string; email: string } | null>(null);
  const formRef = useRef<HTMLFormElement>(null);

  const submit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const fd = new FormData(form);

    const next: Errors = {};
    if (!String(fd.get("name") || "").trim()) next.name = true;
    const email = String(fd.get("email") || "").trim();
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) next.email = true;
    if (!String(fd.get("subject") || "").trim()) next.subject = true;
    if (!String(fd.get("message") || "").trim()) next.message = true;
    setErrs(next);
    setSendError("");
    if (Object.keys(next).length) return;

    fd.set("mode", mode);
    for (const f of files) fd.append("attachments", f);

    setSubmitting(true);
    try {
      const res = await apiFetch<{ ok: boolean; reference_id: string }>(`${API_URL}/contact-submissions`, {
        method: "POST",
        body: fd,
      });
      setDone({ mode, refId: res.reference_id, email });
    } catch (err) {
      setSendError(
        err instanceof Error && err.message
          ? err.message
          : "Something went wrong sending your message. Please try again."
      );
    } finally {
      setSubmitting(false);
    }
  };

  const reset = () => {
    setDone(null);
    setErrs({});
    setFiles([]);
    setSendError("");
    formRef.current?.reset();
  };

  const cx = (k: keyof Errors) => `inp${errs[k] ? " err" : ""}`;

  return (
    <div
      style={{
        background: "var(--card)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-lg)",
        overflow: "hidden",
      }}
    >
      {done ? (
        <Success {...done} onReset={reset} />
      ) : (
        <div style={{ padding: "32px 36px 36px" }}>
          <Segmented
            value={mode}
            onChange={(m) => {
              setMode(m);
              setErrs({});
              setSendError("");
            }}
            options={[
              { id: "ticket", label: "Support ticket", icon: CI.ticket },
              { id: "message", label: "General message", icon: CI.chat },
            ]}
          />
          <p style={{ fontSize: 13.5, lineHeight: 1.55, color: "var(--muted-fg)", margin: "16px 2px 26px" }}>
            {mode === "ticket"
              ? "Hit a snag or found a bug? File a ticket and our team will track it to resolution."
              : "Questions, partnerships, press, or just saying hello — drop us a line."}
          </p>

          <form ref={formRef} onSubmit={submit} noValidate style={{ display: "grid", gap: 20 }}>
            {/* Honeypot — hidden from humans and from assistive tech; bots that
                fill every field trip it and are silently discarded server-side. */}
            <input
              type="text"
              name="website"
              tabIndex={-1}
              autoComplete="off"
              aria-hidden="true"
              style={{ position: "absolute", left: "-9999px", width: 1, height: 1, opacity: 0 }}
            />

            <div className="lp-contact-pair" style={PAIR_STYLE}>
              <Field label="Full name">
                <input name="name" className={cx("name")} placeholder="Jane Doe" />
              </Field>
              <Field label="Email">
                <input name="email" type="email" className={cx("email")} placeholder="you@studio.com" />
              </Field>
            </div>

            {mode === "ticket" ? (
              <div className="lp-contact-pair" style={PAIR_STYLE}>
                <Field label="Related product">
                  <select name="product" className="sel" defaultValue="">
                    <option value="" disabled>
                      Select a tool…
                    </option>
                    {PRODUCTS.map((p) => (
                      <option key={p}>{p}</option>
                    ))}
                  </select>
                </Field>
                <Field label="Account email" optional>
                  <input name="account_email" className="inp" placeholder="workspace login" />
                </Field>
              </div>
            ) : (
              <div className="lp-contact-pair" style={PAIR_STYLE}>
                <Field label="Company / role" optional>
                  <input name="company" className="inp" placeholder="Independent artist" />
                </Field>
                <Field label="Topic">
                  <select name="topic" className="sel" defaultValue="General inquiry">
                    {TOPICS.map((t) => (
                      <option key={t}>{t}</option>
                    ))}
                  </select>
                </Field>
              </div>
            )}

            <Field label="Subject">
              <input
                name="subject"
                className={cx("subject")}
                placeholder={mode === "ticket" ? "Brief summary of the issue" : "What's this about?"}
              />
            </Field>
            <Field label={mode === "ticket" ? "Describe the issue" : "Message"}>
              <textarea
                name="message"
                className={`txt${errs.message ? " err" : ""}`}
                placeholder={
                  mode === "ticket"
                    ? "What happened, what you expected, and any steps to reproduce it…"
                    : "Tell us what you have in mind…"
                }
              />
            </Field>

            <AttachField files={files} setFiles={setFiles} />

            {sendError && (
              <p
                role="alert"
                style={{ fontSize: 13, color: "hsl(0 60% 45%)", margin: 0, fontWeight: 500, lineHeight: 1.5 }}
              >
                {sendError}
              </p>
            )}

            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 16,
                flexWrap: "wrap",
              }}
            >
              <p style={{ fontSize: 12.5, lineHeight: 1.5, color: "var(--muted-fg)", margin: 0, maxWidth: 320 }}>
                By submitting you agree to our <LinkOrAnchor href="/privacy">privacy policy</LinkOrAnchor>.
              </p>
              <button
                type="submit"
                disabled={submitting}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 8,
                  padding: "14px 24px",
                  borderRadius: 999,
                  background: "var(--primary)",
                  color: "var(--primary-fg)",
                  border: "none",
                  cursor: submitting ? "not-allowed" : "pointer",
                  opacity: submitting ? 0.65 : 1,
                  fontFamily: "inherit",
                  fontSize: 15,
                  fontWeight: 600,
                  whiteSpace: "nowrap",
                  boxShadow: "var(--shadow-md)",
                }}
              >
                {submitting ? "Sending…" : mode === "ticket" ? "Submit ticket" : "Send message"}
                {!submitting && <LandingIcons.arrow style={{ width: 16, height: 16 }} />}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}

function ChannelCard({
  icon: Icon,
  title,
  children,
}: {
  icon: (p: SVGProps<SVGSVGElement>) => JSX.Element;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        display: "flex",
        gap: 14,
        padding: "20px 22px",
        background: "var(--card)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
      }}
    >
      <div
        style={{
          flexShrink: 0,
          width: 40,
          height: 40,
          borderRadius: "var(--radius-sm)",
          background: "var(--accent-soft)",
          color: "var(--primary)",
          display: "grid",
          placeItems: "center",
        }}
      >
        <Icon style={{ width: 20, height: 20 }} />
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 15, fontWeight: 600, letterSpacing: "-0.01em", marginBottom: 3 }}>{title}</div>
        <div style={{ fontSize: 13.5, lineHeight: 1.55, color: "var(--muted-fg)" }}>{children}</div>
      </div>
    </div>
  );
}

function Sidebar() {
  return (
    <div style={{ display: "grid", gap: 14, alignContent: "start" }}>
      <ChannelCard icon={CI.clock} title="Response time">
        Tickets answered within <strong style={{ color: "var(--fg)" }}>1 business day</strong>.
      </ChannelCard>
      <ChannelCard icon={CI.book} title="Browse the docs">
        Many answers live in our{" "}
        <LinkOrAnchor href="/docs" style={{ textDecoration: "none", fontWeight: 500 }}>
          documentation
        </LinkOrAnchor>{" "}
        — setup, integrations, and how each tool works.
      </ChannelCard>
    </div>
  );
}

function Hero() {
  return (
    <section id="contact" style={{ position: "relative", padding: "84px 32px 20px", overflow: "hidden" }}>
      <div
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          pointerEvents: "none",
          background: "radial-gradient(760px 320px at 50% -30%, var(--accent-soft), transparent 62%)",
        }}
      />
      <div style={{ maxWidth: 1080, margin: "0 auto", position: "relative", textAlign: "center" }}>
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
          Contact
        </p>
        <h1
          className="tighter"
          style={{
            margin: 0,
            fontSize: "clamp(40px, 7vw, 68px)",
            lineHeight: 1.0,
            fontWeight: 700,
            letterSpacing: "-0.04em",
          }}
        >
          How can we help?
        </h1>
        <p style={{ maxWidth: 560, margin: "22px auto 0", fontSize: 19, lineHeight: 1.55, color: "var(--muted-fg)" }}>
          File a support ticket or just reach out. Either way, a real person on the team reads every message.
        </p>
      </div>
    </section>
  );
}

const Contact = () => {
  return (
    <div className="landing-page min-h-screen" style={{ background: "var(--bg)", color: "var(--fg)" }}>
      <LandingHeader />
      <Hero />
      <section style={{ padding: "40px 32px 96px" }}>
        <div
          className="lp-contact-grid"
          style={{
            maxWidth: 1080,
            margin: "0 auto",
            display: "grid",
            gridTemplateColumns: "1.55fr 1fr",
            gap: 40,
            alignItems: "start",
          }}
        >
          <ContactForm />
          <Sidebar />
        </div>
      </section>
      <LandingFooter />
    </div>
  );
};

export default Contact;
