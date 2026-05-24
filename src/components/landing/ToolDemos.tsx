import {
  CSSProperties,
  ComponentType,
  Fragment,
  ReactNode,
  memo,
  useEffect,
  useRef,
  useState,
} from "react";

// ──────────────────────────────────────────────────────────────────────
// Shared clock driving the auto-looping demos via requestAnimationFrame
// ──────────────────────────────────────────────────────────────────────
function useDemoClock(duration: number, playing: boolean): number {
  const [t, setT] = useState(0);
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let start: number | null = null;
    const loop = (now: number) => {
      if (start == null) start = now;
      const elapsed = (now - start) % duration;
      setT(elapsed / duration);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [duration, playing]);
  return t;
}

// Pause the RAF clock when this demo is off-screen — saves frames the
// user can't see and stops 5 simultaneous animations on a long page.
function useInViewport<T extends HTMLElement>(): [
  React.RefObject<T>,
  boolean,
] {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) setInView(e.isIntersecting);
      },
      { rootMargin: "100px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return [ref, inView];
}

// ──────────────────────────────────────────────────────────────────────
// Device-frame chrome shared across the demos
// ──────────────────────────────────────────────────────────────────────
type DemoFrameProps = {
  children: ReactNode;
  title?: string;
  tab?: string;
  height?: number;
};

function DemoFrame({ children, title, tab = "/tools", height = 440 }: DemoFrameProps) {
  return (
    <div
      style={{
        borderRadius: "var(--radius)",
        background: "var(--card)",
        border: "1px solid var(--border)",
        boxShadow: "var(--shadow-lg)",
        overflow: "hidden",
        height,
        position: "relative",
      }}
    >
      <div
        style={{
          height: 40,
          borderBottom: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "0 14px",
          background: "color-mix(in srgb, var(--muted-bg) 60%, var(--card))",
        }}
      >
        <div style={{ display: "flex", gap: 6 }}>
          {DOT_COLORS.map((c, i) => (
            <span
              key={i}
              style={{
                width: 11,
                height: 11,
                borderRadius: 999,
                background: c,
                opacity: 0.7,
              }}
            />
          ))}
        </div>
        <div
          className="mono"
          style={{
            flex: 1,
            height: 22,
            borderRadius: 6,
            background: "var(--bg)",
            border: "1px solid var(--border)",
            display: "flex",
            alignItems: "center",
            padding: "0 10px",
            fontSize: 11,
            color: "var(--muted-fg)",
          }}
        >
          <span style={{ opacity: 0.5 }}>msanii.app</span>
          <span style={{ color: "var(--fg)" }}>{tab}</span>
        </div>
        {title ? (
          <div style={{ fontSize: 11.5, color: "var(--muted-fg)", fontWeight: 500 }}>
            {title}
          </div>
        ) : null}
      </div>
      <div
        style={{
          height: `calc(${height}px - 40px)`,
          position: "relative",
          overflow: "hidden",
        }}
      >
        {children}
      </div>
    </div>
  );
}

const DOT_COLORS = ["#fb7185", "#fbbf24", "#84cc16"];

// ──────────────────────────────────────────────────────────────────────
// Wrapper that pauses the demo when off-screen
// ──────────────────────────────────────────────────────────────────────
type DemoBodyProps = { playing: boolean; height: number };

function makeLazyDemo(Inner: ComponentType<DemoBodyProps>) {
  const Wrapped = memo(function LazyDemo({ playing, height }: DemoBodyProps) {
    const [ref, inView] = useInViewport<HTMLDivElement>();
    return (
      <div ref={ref}>
        <Inner playing={playing && inView} height={height} />
      </div>
    );
  });
  Wrapped.displayName = `LazyDemo(${Inner.displayName ?? Inner.name})`;
  return Wrapped;
}

// ======================================================================
// 1. OneClick — royalty calculator
// ======================================================================
const ONECLICK_ROWS = [
  { song: "Cold Mornings", party: "Mara Greene", role: "Producer", pct: 25, pay: "$1,240.00" },
  { song: "Cold Mornings", party: "Sasha Lin", role: "Co-writer", pct: 15, pay: "$744.00" },
  { song: "Lila", party: "Jane Doe", role: "Artist", pct: 50, pay: "$2,300.00" },
  { song: "Lila", party: "Kibet", role: "Mix Eng.", pct: 5, pay: "$230.00" },
  { song: "Nine to Six", party: "Jane Doe", role: "Artist", pct: 50, pay: "$1,815.50" },
];

const ONECLICK_STAGES = ["starting", "downloading", "extracting_royalty", "processing", "complete"];

const ONECLICK_SEGS = [
  { label: "Jane Doe", value: 55, color: "var(--primary)" },
  { label: "Mara Greene", value: 18, color: "var(--accent)" },
  { label: "Sasha Lin", value: 10, color: "hsl(168 50% 55%)" },
  { label: "Others", value: 17, color: "hsl(150 20% 75%)" },
];

function arcPoint(frac: number) {
  const a = frac * 2 * Math.PI - Math.PI / 2;
  return { x: 60 + 48 * Math.cos(a), y: 60 + 48 * Math.sin(a) };
}

function DemoOneClickInner({ playing, height }: DemoBodyProps) {
  const t = useDemoClock(9000, playing);
  const phase = t < 0.2 ? 0 : t < 0.55 ? 1 : 2;
  const progress = phase === 1 ? Math.min(1, (t - 0.2) / 0.35) : phase === 2 ? 1 : 0;
  const rowsVisible = phase === 2 ? Math.floor(((t - 0.55) / 0.45) * 5) + 1 : 0;
  const stageIdx = Math.min(4, Math.floor(progress * 5));

  return (
    <DemoFrame tab="/tools/oneclick" title="OneClick" height={height}>
      <div style={{ height: "100%", display: "grid", gridTemplateColumns: "240px 1fr", gap: 0 }}>
        <div style={{ borderRight: "1px solid var(--border)", padding: 16, background: "var(--muted-bg)" }}>
          <div style={LABEL_STYLE}>Contracts</div>
          {["Mara_Production.pdf", "Sasha_Cowrite_v2.pdf"].map((f) => (
            <div key={f} style={FILE_CHIP_STYLE}>
              <span style={CHECK_DOT_STYLE}>✓</span>
              <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f}</span>
            </div>
          ))}
          <div style={{ ...LABEL_STYLE, marginTop: 18 }}>Royalty statement</div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "8px 10px",
              borderRadius: 8,
              border: "1px solid var(--accent)",
              background: "color-mix(in srgb, var(--accent) 12%, var(--card))",
              fontSize: 12,
            }}
          >
            <span style={CHECK_DOT_STYLE}>✓</span>
            <span>Spotify_Q3_2025.csv</span>
          </div>
          <button
            type="button"
            style={{
              marginTop: 18,
              width: "100%",
              padding: "10px 14px",
              background: phase === 0 ? "var(--primary)" : "var(--muted-bg)",
              color: phase === 0 ? "var(--bg)" : "var(--muted-fg)",
              border: phase === 0 ? "none" : "1px solid var(--border)",
              borderRadius: 8,
              fontWeight: 600,
              fontSize: 12.5,
              cursor: "pointer",
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 6,
              transform: phase === 0 ? `scale(${1 + Math.sin(t * 25) * 0.02})` : "none",
              transition: "all 0.2s",
            }}
          >
            {phase === 0 ? "Calculate royalties" : phase === 1 ? "Calculating…" : "Recalculate"}
          </button>
        </div>

        <div style={{ padding: 18, position: "relative" }}>
          {phase === 2 ? (
            <div style={{ height: "100%", display: "flex", flexDirection: "column", gap: 10, minHeight: 0 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <div>
                  <div style={{ fontSize: 11.5, color: "var(--muted-fg)" }}>Total payments owed</div>
                  <div className="tighter" style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.02em" }}>
                    $6,329.50
                  </div>
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  <span
                    style={{
                      padding: "5px 10px",
                      borderRadius: 999,
                      fontSize: 11,
                      fontWeight: 600,
                      background: "color-mix(in srgb, var(--accent) 14%, transparent)",
                      color: "var(--accent)",
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 5,
                    }}
                  >
                    <span style={{ width: 6, height: 6, borderRadius: 999, background: "var(--accent)" }} />
                    Cached
                  </span>
                  <button type="button" style={MINI_BTN}>CSV</button>
                  <button type="button" style={MINI_BTN}>Excel</button>
                  <button
                    type="button"
                    style={{
                      ...MINI_BTN,
                      border: "none",
                      background: "var(--primary)",
                      color: "var(--bg)",
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 4,
                    }}
                  >
                    ↑ Share
                  </button>
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 140px", gap: 14, minHeight: 0, flex: 1 }}>
                <div style={{ overflow: "hidden" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                    <thead>
                      <tr style={{ color: "var(--muted-fg)", textAlign: "left" }}>
                        {["Song", "Payee", "Role", "Share"].map((h) => (
                          <th
                            key={h}
                            style={{
                              fontWeight: 500,
                              padding: "6px 4px",
                              borderBottom: "1px solid var(--border)",
                            }}
                          >
                            {h}
                          </th>
                        ))}
                        <th
                          style={{
                            fontWeight: 500,
                            padding: "6px 4px",
                            borderBottom: "1px solid var(--border)",
                            textAlign: "right",
                          }}
                        >
                          Amount owed
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {ONECLICK_ROWS.map((r, i) => (
                        <tr
                          key={`${r.song}-${r.party}`}
                          style={{
                            opacity: i < rowsVisible ? 1 : 0,
                            transform: i < rowsVisible ? "translateY(0)" : "translateY(4px)",
                            transition: "all 0.3s",
                          }}
                        >
                          <td style={CELL}>{r.song}</td>
                          <td style={CELL}>{r.party}</td>
                          <td style={{ ...CELL, color: "var(--muted-fg)" }}>{r.role}</td>
                          <td style={CELL} className="mono">
                            {r.pct}%
                          </td>
                          <td style={{ ...CELL, textAlign: "right", fontWeight: 600 }} className="mono">
                            {r.pay}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div style={{ marginTop: 10, display: "flex", gap: 6, flexWrap: "wrap" }}>
                    <span style={SHARE_CHIP}>
                      <span style={{ width: 12, height: 12, borderRadius: 3, background: "#1a73e8" }} />
                      Save to Google Drive
                    </span>
                    <span style={SHARE_CHIP}>
                      <span style={{ width: 12, height: 12, borderRadius: 3, background: "#611f69" }} />
                      Share to Slack
                    </span>
                  </div>
                </div>
                <OneClickPie />
              </div>
            </div>
          ) : (
            <div style={{ display: "grid", placeItems: "center", height: "100%" }}>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 14 }}>Calculating Royalties</div>
                <div style={{ position: "relative", width: 128, height: 128, margin: "0 auto" }}>
                  <svg viewBox="0 0 120 120" width="128" height="128">
                    <circle cx="60" cy="60" r="50" fill="none" stroke="var(--border)" strokeWidth="8" />
                    <circle
                      cx="60"
                      cy="60"
                      r="50"
                      fill="none"
                      stroke="var(--primary)"
                      strokeWidth="8"
                      strokeLinecap="round"
                      strokeDasharray={`${progress * 314} 314`}
                      transform="rotate(-90 60 60)"
                      style={{ transition: "stroke-dasharray 0.15s linear" }}
                    />
                  </svg>
                  <div
                    style={{
                      position: "absolute",
                      inset: 0,
                      display: "grid",
                      placeItems: "center",
                      fontSize: 26,
                      fontWeight: 700,
                      letterSpacing: "-0.02em",
                    }}
                  >
                    {Math.round(progress * 100)}%
                  </div>
                </div>
                <div
                  className="mono"
                  style={{
                    marginTop: 14,
                    fontSize: 12,
                    color: "var(--muted-fg)",
                  }}
                >
                  {phase === 0 ? "ready to calculate" : `${ONECLICK_STAGES[stageIdx]}…`}
                </div>
                <div style={{ marginTop: 14, display: "flex", gap: 6, justifyContent: "center" }}>
                  {ONECLICK_STAGES.map((s, i) => (
                    <span
                      key={s}
                      style={{
                        width: 24,
                        height: 4,
                        borderRadius: 2,
                        background: i <= stageIdx && phase === 1 ? "var(--accent)" : "var(--border)",
                        transition: "background 0.2s",
                      }}
                    />
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </DemoFrame>
  );
}

function OneClickPie() {
  let acc = 0;
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
      <div style={{ ...LABEL_STYLE, marginBottom: 6, fontSize: 10 }}>By payee</div>
      <svg viewBox="0 0 120 120" width="104" height="104">
        {ONECLICK_SEGS.map((s) => {
          const start = arcPoint(acc / 100);
          acc += s.value;
          const end = arcPoint(acc / 100);
          const large = s.value > 50 ? 1 : 0;
          return (
            <path
              key={s.label}
              d={`M60,60 L${start.x},${start.y} A48,48 0 ${large} 1 ${end.x},${end.y} Z`}
              fill={s.color}
              opacity={0.9}
            />
          );
        })}
        <circle cx="60" cy="60" r="22" fill="var(--card)" />
      </svg>
      <div style={{ marginTop: 8, display: "grid", gap: 4 }}>
        {ONECLICK_SEGS.slice(0, 3).map((s) => (
          <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10 }}>
            <span style={{ width: 7, height: 7, borderRadius: 2, background: s.color }} />
            <span>{s.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

const DemoOneClick = makeLazyDemo(DemoOneClickInner);

// ======================================================================
// 2. Zoe — AI contract chat
// ======================================================================
type ZoeMessage = { role: "user" | "zoe"; text: string; cite?: string };

const ZOE_MESSAGES: ZoeMessage[] = [
  { role: "user", text: "What's the producer royalty on this contract?" },
  {
    role: "zoe",
    text:
      "In the agreement with Mara Greene, the producer royalty is **3% of suggested retail list price**, escalating to 4% after 250k units.",
    cite: "Greene_Production_Agreement.pdf · §4.1",
  },
  { role: "user", text: "When does it kick in?" },
  {
    role: "zoe",
    text:
      "The escalation applies to all sales **after** the 250,000th unit, not retroactively. It's in §4.2.",
    cite: '§4.2 · "applicable solely on sales in excess of"',
  },
];

function formatBoldHtml(text: string): string {
  // narrow safe transform — we only support **bold** markers in canned demo strings
  return text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

function DemoZoeInner({ playing, height }: DemoBodyProps) {
  const t = useDemoClock(9000, playing);
  const visible = Math.min(ZOE_MESSAGES.length, Math.floor(t * 5) + 1);
  const last = ZOE_MESSAGES[visible - 1];
  const isTypingZoe = !!last && last.role === "zoe";
  const localT = t * 5 - (visible - 1);
  const typedChars = isTypingZoe ? Math.floor(localT * last.text.length) : last ? last.text.length : 0;

  return (
    <DemoFrame tab="/tools/zoe" title="Zoe — AI contract analyst" height={height}>
      <div style={{ height: "100%", display: "grid", gridTemplateColumns: "210px 1fr" }}>
        <div style={{ borderRight: "1px solid var(--border)", padding: 14, background: "var(--muted-bg)" }}>
          <div style={LABEL_STYLE}>Source</div>
          <div style={{ border: "1px solid var(--border)", borderRadius: 8, padding: 10, background: "var(--card)", fontSize: 11.5 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600 }}>
              <span
                style={{
                  width: 22,
                  height: 28,
                  borderRadius: 3,
                  background: "linear-gradient(180deg, #fef3c7, #fde68a)",
                  border: "1px solid #f59e0b",
                  display: "grid",
                  placeItems: "center",
                  fontSize: 8,
                  fontWeight: 700,
                  color: "#92400e",
                }}
              >
                PDF
              </span>
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                Greene_Production.pdf
              </span>
            </div>
            <div style={{ marginTop: 8, height: 6, borderRadius: 3, background: "var(--muted-bg)", overflow: "hidden" }}>
              <div style={{ width: "100%", height: "100%", background: "var(--accent)" }} />
            </div>
            <div style={{ fontSize: 10, color: "var(--muted-fg)", marginTop: 6 }}>Indexed · 18 sections</div>
          </div>
          <div style={{ ...LABEL_STYLE, marginTop: 18 }}>Suggested</div>
          {["Producer royalty?", "Escalation tiers", "Termination clauses"].map((q) => (
            <div
              key={q}
              style={{
                padding: "7px 10px",
                fontSize: 11.5,
                marginBottom: 6,
                border: "1px solid var(--border)",
                borderRadius: 999,
                background: "var(--card)",
              }}
            >
              {q}
            </div>
          ))}
        </div>
        <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 10, overflow: "hidden" }}>
          {ZOE_MESSAGES.slice(0, visible).map((m, i) => {
            const isLast = i === visible - 1;
            const text = isLast && m.role === "zoe" ? m.text.slice(0, typedChars) : m.text;
            return (
              <div
                key={i}
                style={{
                  alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                  maxWidth: "82%",
                  background: m.role === "user" ? "var(--primary)" : "var(--muted-bg)",
                  color: m.role === "user" ? "var(--bg)" : "var(--fg)",
                  padding: "10px 14px",
                  borderRadius: m.role === "user" ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
                  fontSize: 13,
                  lineHeight: 1.5,
                  animation: "lp-fade-up 0.3s ease-out",
                  border: m.role === "zoe" ? "1px solid var(--border)" : "none",
                }}
              >
                {m.role === "zoe" ? (
                  <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6, fontSize: 11, color: "var(--accent)", fontWeight: 600 }}>
                    <span style={{ width: 6, height: 6, borderRadius: 999, background: "var(--accent)" }} />
                    Zoe
                  </div>
                ) : null}
                <div dangerouslySetInnerHTML={{ __html: formatBoldHtml(text) }} />
                {isLast && m.role === "zoe" && typedChars < m.text.length ? (
                  <span
                    style={{
                      display: "inline-block",
                      width: 7,
                      height: 14,
                      marginLeft: 2,
                      background: "var(--fg)",
                      verticalAlign: "middle",
                      animation: "lp-blink 1s steps(2) infinite",
                    }}
                  />
                ) : null}
                {m.cite && (!isLast || typedChars >= m.text.length) ? (
                  <div
                    className="mono"
                    style={{
                      marginTop: 8,
                      fontSize: 10.5,
                      color: "var(--muted-fg)",
                      padding: "4px 8px",
                      background: "var(--card)",
                      border: "1px solid var(--border)",
                      borderRadius: 6,
                      display: "inline-block",
                    }}
                  >
                    📎 {m.cite}
                  </div>
                ) : null}
              </div>
            );
          })}
          <div
            style={{
              marginTop: "auto",
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "10px 14px",
              border: "1px solid var(--border)",
              borderRadius: 999,
              background: "var(--bg)",
            }}
          >
            <span style={{ fontSize: 12.5, color: "var(--muted-fg)" }}>Ask about your contracts…</span>
            <span
              style={{
                marginLeft: "auto",
                width: 28,
                height: 28,
                borderRadius: 999,
                background: "var(--primary)",
                display: "grid",
                placeItems: "center",
                color: "var(--bg)",
                fontSize: 12,
              }}
            >
              ↑
            </span>
          </div>
        </div>
      </div>
    </DemoFrame>
  );
}

const DemoZoe = makeLazyDemo(DemoZoeInner);

// ======================================================================
// 3. Split Sheet — form + PDF preview
// ======================================================================
const SPLIT_COLLABORATORS = [
  { name: "Jane Doe", role: "Songwriter", pub: 45, mas: 50 },
  { name: "Mara Greene", role: "Producer", pub: 25, mas: 30 },
  { name: "Kibet", role: "Lyricist", pub: 20, mas: 10 },
  { name: "Adila", role: "Featured", pub: 10, mas: 10 },
];

function DemoSplitSheetInner({ playing, height }: DemoBodyProps) {
  const t = useDemoClock(8000, playing);
  const filled = Math.min(SPLIT_COLLABORATORS.length, Math.floor(t * 6) + 1);
  const pubTotal = SPLIT_COLLABORATORS.slice(0, filled).reduce((s, c) => s + c.pub, 0);
  const masTotal = SPLIT_COLLABORATORS.slice(0, filled).reduce((s, c) => s + c.mas, 0);
  const balanced = pubTotal === 100 && masTotal === 100;

  return (
    <DemoFrame tab="/tools/split-sheet" title="Split Sheet" height={height}>
      <div style={{ height: "100%", display: "grid", gridTemplateColumns: "1fr 1fr" }}>
        <div style={{ padding: 16, borderRight: "1px solid var(--border)", overflow: "hidden" }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Track</div>
          <div style={{ padding: "7px 10px", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12, background: "var(--bg)" }}>
            Cold Mornings
          </div>

          <div
            style={{
              fontSize: 12,
              fontWeight: 600,
              marginTop: 14,
              marginBottom: 6,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <span>Contributors</span>
            <span className="mono" style={{ fontSize: 10, color: "var(--muted-fg)", fontWeight: 500 }}>
              name · role · pub% · mas%
            </span>
          </div>
          <div style={{ display: "grid", gap: 5 }}>
            {SPLIT_COLLABORATORS.map((c, i) => (
              <div
                key={c.name}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1.1fr 0.95fr 42px 42px",
                  gap: 4,
                  opacity: i < filled ? 1 : 0.3,
                  transform: i < filled ? "translateY(0)" : "translateY(4px)",
                  transition: "all 0.3s",
                }}
              >
                <div style={SPLIT_CELL}>{c.name}</div>
                <div style={{ ...SPLIT_CELL, color: "var(--muted-fg)" }}>{c.role}</div>
                <div className="mono" style={SPLIT_PCT_CELL}>
                  {c.pub}%
                </div>
                <div className="mono" style={SPLIT_PCT_CELL}>
                  {c.mas}%
                </div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 10, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
            <div style={SPLIT_TOTAL_BOX}>
              <span style={{ color: "var(--muted-fg)" }}>Publishing</span>
              <span className="mono" style={{ fontWeight: 700, color: pubTotal === 100 ? "var(--accent)" : "var(--muted-fg)" }}>
                {pubTotal}%
              </span>
            </div>
            <div style={SPLIT_TOTAL_BOX}>
              <span style={{ color: "var(--muted-fg)" }}>Master</span>
              <span className="mono" style={{ fontWeight: 700, color: masTotal === 100 ? "var(--accent)" : "var(--muted-fg)" }}>
                {masTotal}%
              </span>
            </div>
          </div>
          <div style={{ marginTop: 12, display: "flex", gap: 6 }}>
            <button
              type="button"
              style={{
                flex: 1,
                padding: "9px 12px",
                background: balanced ? "var(--primary)" : "var(--muted-bg)",
                color: balanced ? "var(--bg)" : "var(--muted-fg)",
                border: "none",
                borderRadius: 6,
                fontWeight: 600,
                fontSize: 12,
              }}
            >
              ↓ PDF
            </button>
            <button
              type="button"
              style={{
                flex: 1,
                padding: "9px 12px",
                background: "var(--card)",
                color: "var(--fg)",
                border: "1px solid var(--border-strong)",
                borderRadius: 6,
                fontWeight: 600,
                fontSize: 12,
              }}
            >
              ↓ DOCX
            </button>
          </div>
        </div>
        <div style={{ background: "var(--muted-bg)", padding: 16, display: "grid", placeItems: "center", overflow: "hidden" }}>
          <div
            style={{
              width: "88%",
              aspectRatio: "0.77",
              background: "white",
              color: "#1f2937",
              borderRadius: 6,
              boxShadow: "0 14px 30px -10px rgba(15, 38, 26, 0.18)",
              padding: 14,
              fontSize: 7,
              lineHeight: 1.5,
              transform: "rotate(-1.2deg)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", borderBottom: "1px solid #e5e7eb", paddingBottom: 5 }}>
              <strong style={{ fontSize: 8.5 }}>SPLIT SHEET · Cold Mornings</strong>
              <span style={{ color: "#6b7280" }}>2026-05-23</span>
            </div>
            <div
              style={{
                marginTop: 6,
                display: "grid",
                gridTemplateColumns: "1.1fr 0.9fr 24px 24px",
                columnGap: 4,
                rowGap: 3,
              }}
            >
              <strong>Name</strong>
              <strong>Role</strong>
              <strong style={{ textAlign: "right" }}>Pub</strong>
              <strong style={{ textAlign: "right" }}>Mas</strong>
              {SPLIT_COLLABORATORS.slice(0, filled).map((c) => (
                <Fragment key={c.name}>
                  <span>{c.name}</span>
                  <span style={{ color: "#6b7280" }}>{c.role}</span>
                  <span style={{ textAlign: "right" }}>{c.pub}%</span>
                  <span style={{ textAlign: "right" }}>{c.mas}%</span>
                </Fragment>
              ))}
            </div>
            <div style={{ marginTop: 10, borderTop: "1px solid #e5e7eb", paddingTop: 5, color: "#6b7280" }}>
              All parties acknowledge and agree to the above ownership stakes.
            </div>
            <div style={{ marginTop: 14, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              {[0, 1].map((i) => (
                <div key={i}>
                  <div style={{ borderBottom: "1px solid #1f2937", height: 12 }} />
                  <div style={{ fontSize: 6, color: "#6b7280", marginTop: 2 }}>Signature</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </DemoFrame>
  );
}

const DemoSplitSheet = makeLazyDemo(DemoSplitSheetInner);

// ======================================================================
// 4. Portfolio — artist cards + projects list
// ======================================================================
const PORTFOLIO_ARTISTS = [
  { name: "Jane Doe", tracks: 24, status: "Active", hue: 152 },
  { name: "KOBI", tracks: 11, status: "Releasing", hue: 168 },
  { name: "Mara Greene", tracks: 7, status: "Onboarding", hue: 138 },
  { name: "Owen Bly", tracks: 18, status: "Active", hue: 175 },
];

const PORTFOLIO_PROJECTS = [
  { title: "Cold Mornings — EP", artist: "Jane Doe", files: 12, due: "Jun 14" },
  { title: "KOBI · Single Q3", artist: "KOBI", files: 4, due: "Jul 02" },
  { title: "Greene Demo Sessions", artist: "Mara Greene", files: 9, due: "Aug 21" },
];

function statusChipStyle(status: string): CSSProperties {
  if (status === "Active") {
    return {
      background: "color-mix(in srgb, var(--accent) 16%, transparent)",
      color: "var(--accent)",
    };
  }
  if (status === "Releasing") {
    return {
      background: "color-mix(in srgb, #f59e0b 16%, transparent)",
      color: "#b45309",
    };
  }
  return { background: "var(--muted-bg)", color: "var(--muted-fg)" };
}

function DemoPortfolioInner({ playing, height }: DemoBodyProps) {
  const t = useDemoClock(8000, playing);
  const focused = Math.floor(t * 4) % 4;
  return (
    <DemoFrame tab="/portfolio" title="Portfolio" height={height}>
      <div style={{ padding: 18, height: "100%", display: "grid", gridTemplateRows: "auto 1fr", gap: 14 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <div style={{ fontSize: 17, fontWeight: 700, letterSpacing: "-0.01em" }}>Your roster</div>
            <div style={{ fontSize: 12, color: "var(--muted-fg)" }}>4 artists · 60 tracks</div>
          </div>
          <button
            type="button"
            style={{
              padding: "7px 12px",
              borderRadius: 8,
              background: "var(--primary)",
              color: "var(--bg)",
              fontSize: 12,
              fontWeight: 600,
              border: "none",
              display: "inline-flex",
              alignItems: "center",
              gap: 4,
            }}
          >
            + New artist
          </button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, minHeight: 0 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, alignContent: "start" }}>
            {PORTFOLIO_ARTISTS.map((a, i) => (
              <div
                key={a.name}
                style={{
                  padding: 12,
                  borderRadius: 10,
                  border: "1px solid var(--border)",
                  background: "var(--card)",
                  transition: "all 0.35s",
                  transform: focused === i ? "translateY(-3px)" : "translateY(0)",
                  boxShadow:
                    focused === i
                      ? "0 12px 24px -10px color-mix(in srgb, var(--primary) 25%, transparent)"
                      : "none",
                }}
              >
                <div
                  style={{
                    width: 38,
                    height: 38,
                    borderRadius: 999,
                    background: `linear-gradient(135deg, hsl(${a.hue} 40% 92%), hsl(${a.hue} 30% 78%))`,
                    display: "grid",
                    placeItems: "center",
                    fontSize: 13,
                    fontWeight: 700,
                    color: `hsl(${a.hue} 50% 28%)`,
                    marginBottom: 8,
                  }}
                >
                  {a.name
                    .split(" ")
                    .map((s) => s[0])
                    .slice(0, 2)
                    .join("")}
                </div>
                <div style={{ fontSize: 13, fontWeight: 600 }}>{a.name}</div>
                <div style={{ fontSize: 11, color: "var(--muted-fg)" }}>{a.tracks} tracks</div>
                <div
                  style={{
                    marginTop: 8,
                    display: "inline-block",
                    padding: "2px 8px",
                    borderRadius: 999,
                    fontSize: 10,
                    fontWeight: 600,
                    ...statusChipStyle(a.status),
                  }}
                >
                  {a.status}
                </div>
              </div>
            ))}
          </div>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <div style={{ ...LABEL_STYLE, marginBottom: 8 }}>Active projects</div>
            <div style={{ display: "grid", gap: 8 }}>
              {PORTFOLIO_PROJECTS.map((p) => (
                <div
                  key={p.title}
                  style={{
                    padding: "10px 12px",
                    borderRadius: 10,
                    border: "1px solid var(--border)",
                    background: "var(--card)",
                    display: "grid",
                    gridTemplateColumns: "1fr auto",
                    alignItems: "center",
                    gap: 8,
                  }}
                >
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600 }}>{p.title}</div>
                    <div style={{ fontSize: 11, color: "var(--muted-fg)" }}>
                      {p.artist} · {p.files} files
                    </div>
                  </div>
                  <div className="mono" style={{ fontSize: 11, color: "var(--muted-fg)" }}>
                    {p.due}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </DemoFrame>
  );
}

const DemoPortfolio = makeLazyDemo(DemoPortfolioInner);

// ======================================================================
// 5. Workspace — Kanban + integration chips
// ======================================================================
const WORKSPACE_COL_BASE = [
  { id: "todo", title: "To do", tone: "var(--muted-fg)", cards: ["Update artist bio", "Renew DSP token", "Q3 statement upload"] },
  { id: "doing", title: "In progress", tone: "#b45309", cards: ["Mix notes for Kibet", "Onboard Owen Bly"] },
  { id: "done", title: "Done", tone: "var(--accent)", cards: ["Submit registry", "Pay producers"] },
];

const WORKSPACE_INTEGRATIONS = [
  { l: "Google Drive", d: "#1a73e8" },
  { l: "Slack", d: "#611f69" },
  { l: "Notion", d: "#000" },
  { l: "Atlassian", d: "#0052cc" },
];

const TRAVELING_CARD = "Cold Mornings · Master";

function DemoWorkspaceInner({ playing, height }: DemoBodyProps) {
  const t = useDemoClock(9000, playing);
  const stage = Math.floor(t * 3);
  const microT = t * 3 - stage;
  const cols = WORKSPACE_COL_BASE.map((c, i) =>
    i === stage ? { ...c, cards: [TRAVELING_CARD, ...c.cards] } : c,
  );

  return (
    <DemoFrame tab="/workspace/boards" title="Workspace · Boards" height={height}>
      <div style={{ height: "100%", display: "grid", gridTemplateRows: "auto 1fr" }}>
        <div
          style={{
            padding: "10px 18px",
            borderBottom: "1px solid var(--border)",
            display: "flex",
            alignItems: "center",
            gap: 8,
            flexWrap: "wrap",
            background: "color-mix(in srgb, var(--muted-bg) 70%, var(--card))",
          }}
        >
          <span style={{ fontSize: 11, color: "var(--muted-fg)", fontWeight: 500, marginRight: 4 }}>Connected:</span>
          {WORKSPACE_INTEGRATIONS.map((c) => (
            <span
              key={c.l}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                padding: "4px 10px",
                borderRadius: 999,
                background: "var(--card)",
                border: "1px solid var(--border)",
                fontSize: 11.5,
                fontWeight: 500,
              }}
            >
              <span style={{ width: 6, height: 6, borderRadius: 999, background: c.d }} />
              {c.l}
            </span>
          ))}
          <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--accent)", display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 6, height: 6, borderRadius: 999, background: "var(--accent)", animation: "lp-pulse-ring 1.5s infinite" }} />
            Auto-synced 8s ago
          </span>
        </div>
        <div style={{ padding: 14, display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, minHeight: 0 }}>
          {cols.map((col) => (
            <div
              key={col.id}
              style={{
                background: "var(--muted-bg)",
                borderRadius: 10,
                padding: 10,
                display: "flex",
                flexDirection: "column",
                gap: 8,
                minHeight: 0,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "0 4px" }}>
                <span style={{ width: 8, height: 8, borderRadius: 999, background: col.tone }} />
                <span style={{ fontSize: 11.5, fontWeight: 600 }}>{col.title}</span>
                <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--muted-fg)" }}>{col.cards.length}</span>
              </div>
              {col.cards.map((c, i) => {
                const isTrav = c === TRAVELING_CARD;
                return (
                  <div
                    key={`${c}-${i}`}
                    style={{
                      padding: "10px 12px",
                      borderRadius: 8,
                      background: "var(--card)",
                      border: isTrav ? "1px solid var(--accent)" : "1px solid var(--border)",
                      fontSize: 12.5,
                      fontWeight: 500,
                      boxShadow: isTrav
                        ? "0 12px 22px -10px color-mix(in srgb, var(--accent) 40%, transparent)"
                        : "none",
                      transform: isTrav
                        ? `translateY(${Math.sin(microT * Math.PI) * -4}px) rotate(${microT * 2 - 1}deg)`
                        : "none",
                      transition: "transform 0.2s",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      {isTrav ? <span style={{ fontSize: 12 }}>🎵</span> : null}
                      <span>{c}</span>
                    </div>
                    {isTrav ? (
                      <div style={{ marginTop: 6, display: "flex", gap: 4, alignItems: "center", fontSize: 10, color: "var(--muted-fg)" }}>
                        <span style={{ padding: "1px 6px", borderRadius: 4, background: "var(--muted-bg)" }}>Jane Doe</span>
                        <span>· due Jun 14</span>
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </DemoFrame>
  );
}

const DemoWorkspace = makeLazyDemo(DemoWorkspaceInner);

// ──────────────────────────────────────────────────────────────────────
// Shared style fragments
// ──────────────────────────────────────────────────────────────────────
const LABEL_STYLE: CSSProperties = {
  fontSize: 11,
  fontWeight: 600,
  color: "var(--muted-fg)",
  letterSpacing: "0.06em",
  textTransform: "uppercase",
  marginBottom: 10,
};

const FILE_CHIP_STYLE: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  padding: "8px 10px",
  borderRadius: 8,
  marginBottom: 6,
  border: "1px solid var(--border)",
  background: "var(--card)",
  fontSize: 12,
};

const CHECK_DOT_STYLE: CSSProperties = {
  width: 14,
  height: 14,
  borderRadius: 3,
  background: "var(--accent)",
  display: "grid",
  placeItems: "center",
  color: "white",
  fontSize: 9,
  fontWeight: 700,
};

const MINI_BTN: CSSProperties = {
  padding: "5px 10px",
  borderRadius: 6,
  fontSize: 11,
  fontWeight: 600,
  border: "1px solid var(--border)",
  background: "var(--card)",
};

const CELL: CSSProperties = {
  padding: "7px 4px",
  borderBottom: "1px solid var(--border)",
};

const SHARE_CHIP: CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  padding: "5px 10px",
  borderRadius: 999,
  border: "1px solid var(--border)",
  fontSize: 10.5,
  background: "var(--card)",
};

const SPLIT_CELL: CSSProperties = {
  padding: "6px 8px",
  border: "1px solid var(--border)",
  borderRadius: 5,
  fontSize: 11,
  background: "var(--bg)",
};

const SPLIT_PCT_CELL: CSSProperties = {
  padding: "6px 6px",
  border: "1px solid var(--border)",
  borderRadius: 5,
  fontSize: 11,
  textAlign: "right",
  fontWeight: 600,
  background: "var(--bg)",
};

const SPLIT_TOTAL_BOX: CSSProperties = {
  padding: "7px 10px",
  background: "var(--muted-bg)",
  borderRadius: 6,
  fontSize: 11,
  display: "flex",
  justifyContent: "space-between",
};

// ──────────────────────────────────────────────────────────────────────
// Tool registry + topic grouping
// ──────────────────────────────────────────────────────────────────────
export type Tool = {
  id: string;
  name: string;
  tagline: string;
  blurb: string;
  bullets: string[];
  cta: string;
  href: string;
  Demo: ComponentType<DemoBodyProps>;
};

export const TOOLS: Tool[] = [
  {
    id: "oneclick",
    name: "OneClick",
    tagline: "Royalty calculator",
    blurb:
      "Cross-reference any contract against a royalty statement and get a per-song, per-payee breakdown in seconds. Share the PDF straight to Drive or Slack — every receipt, every collaborator, every quarter.",
    bullets: [
      "Streams progress as it works",
      "CSV, Excel & PDF export",
      "Save to Drive · Share to Slack",
    ],
    cta: "Open the calculator",
    href: "/tools/oneclick",
    Demo: DemoOneClick,
  },
  {
    id: "zoe",
    name: "Zoe",
    tagline: "AI contract analyst",
    blurb:
      "Drop a contract in and ask anything. Cited answers grounded in your actual document — no hallucinated clauses, no generic LLM mush. Royalty rates, escalations, term lengths, exits — all at a chat prompt's distance.",
    bullets: [
      "Streaming responses",
      "Inline citations to page & clause",
      "Indexes royalty statements too",
    ],
    cta: "Ask Zoe",
    href: "/tools/zoe",
    Demo: DemoZoe,
  },
  {
    id: "splitsheet",
    name: "Split Sheet",
    tagline: "Splits generator",
    blurb:
      "Auto-balancing split sheets that won't ship until publishing and master both sum to 100. Export PDF or DOCX with signature blocks ready to go, and the receipts stay attached to the project.",
    bullets: [
      "Live totals for pub + master",
      "PDF & DOCX in one click",
      "Signature blocks built-in",
    ],
    cta: "Generate a split sheet",
    href: "/tools/split-sheet",
    Demo: DemoSplitSheet,
  },
  {
    id: "portfolio",
    name: "Portfolio",
    tagline: "Artists & projects",
    blurb:
      "One place for every artist on your roster — projects, files, audio, notes, members. Granular roles (owner / admin / editor / viewer) and RLS-gated so collaborators see only what they should.",
    bullets: [
      "Artist profiles & DSP links",
      "Project files & audio",
      "Owner / admin / editor / viewer roles",
    ],
    cta: "Open Portfolio",
    href: "/portfolio",
    Demo: DemoPortfolio,
  },
  {
    id: "workspace",
    name: "Workspace",
    tagline: "Boards & integrations",
    blurb:
      "Kanban and calendar that talk to Drive, Slack, Notion and the rest. Move a card, sync a folder — the back-office stays out of your way and your team stays in their existing tools.",
    bullets: [
      "Drive · Slack · Notion sync",
      "Kanban + calendar views",
      "Notifications & audit log",
    ],
    cta: "Open Workspace",
    href: "/workspace",
    Demo: DemoWorkspace,
  },
];

export type Topic = {
  id: string;
  label: string;
  headline: string;
  toolIds: string[];
};

export const TOPICS: Topic[] = [
  {
    id: "pay",
    label: "Royalties",
    headline: "Calculate and pay royalties to collaborators.",
    toolIds: ["oneclick"],
  },
  {
    id: "analyse",
    label: "Contracts",
    headline: "Analyse contracts without re-reading them.",
    toolIds: ["zoe"],
  },
  {
    id: "manage",
    label: "Projects",
    headline: "Build and manage your projects.",
    toolIds: ["splitsheet", "portfolio", "workspace"],
  },
];
