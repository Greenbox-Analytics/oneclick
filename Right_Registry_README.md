# Msanii — Rights Registry & Zoe (Prototype)

A high-fidelity HTML/React prototype for **Msanii**, a music rights-management
product. It covers two surfaces:

| Surface | File | What it is |
|---|---|---|
| **Rights Registry** | `Rights Registry.html` | Multi-page app: dashboard, portfolio, project detail, work editor, Add Work wizard |
| **Zoe chat** | `Zoe Chat Redesign.html` | AI contract-analyst chat with a comparison-context picker |
| **Standalone export** | `Msanii Rights Registry (standalone).html` | Single-file offline bundle of the Rights Registry (no server or internet needed) |

---

## Running it

- **Standalone file** — open `Msanii Rights Registry (standalone).html` directly
  in any modern browser. Everything (React, fonts, styles, code) is inlined.
- **Source version** — `Rights Registry.html` loads its modules from the
  `registry/` folder, so it must be served over HTTP (any static server works);
  opening it via `file://` will block the module fetches.

All state is **in-memory sample data** — changes (new works, edited splits,
deleted files) reset on reload. There is no backend.

## Code layout (`registry/`)

| File | Responsibility |
|---|---|
| `app.jsx` | Shell: header, routing (`#/`, `#/portfolio`, `#/project/:id`, `#/work/:id`), theme, toasts (with Undo), global header search, state mutations (`addWork`, `updateWork`, `createProject`) |
| `data.jsx` | Sample artists / projects / works / contracts / files, permissions helpers, completeness checks (`workIssues`, `splitTotals`) |
| `ui.jsx` | Shared primitives: Modal, Segmented, Avatar, badges, formatters |
| `icons.jsx` | 24×24 stroke icon set |
| `dashboard.jsx` | Rights Registry home: stat-card filters, artist ▸ year ▸ project grouping, sorting, work rows with attention flags |
| `portfolio.jsx` | Portfolio grid + project detail (works / files & contracts / members tabs) |
| `work-editor.jsx` | Single-work page: metadata, royalty-splits table, collaborators, documents, traceability + registration checklist |
| `add-work.jsx` | Add Work wizard: project picker, released (Spotify-style lookup) / unreleased branches, AI royalty-split parsing |
| `contracts.jsx` | Documents panel + link/upload modals |
| `styles.css` | Full design system (CSS custom properties, light/dark) |

## Key features

### Dashboard
- **Clickable stat cards** filter the list (Total / Released / Unreleased /
  Need attention / Shared with you).
- Works grouped **artist ▸ year ▸ project**, with per-year dividers.
- Sorting: recently added, title A–Z, release date.
- **Needs-attention flags** on any work that is released without an ISRC, has
  no contracts linked, or whose splits don't total 100%.
- Project names on rows cross-link into the Portfolio.

### Global
- **Header search** across works, projects, contracts, and files.
- Destructive actions (remove collaborator, remove document, delete file)
  offer **Undo** in the toast.

### Add Work wizard
- Scrollable, fixed-height project picker.
- Released branch: track lookup → confirm pulled metadata.
- Unreleased branch: manual entry incl. optional **ISRC / UPC**.
- Final step — **royalty splits**: either attach the related contract and let
  AI extract per-party master/publishing splits (review, edit, add/remove
  parties, 100% totals check), or set your own split manually / skip.
- Duplicate detection warns when a same-title or same-ISRC work exists.

### Work page
- Unified **Royalty splits** table (you + collaborators) with one Edit toggle
  and live totals validation.
- **Split provenance**: "Parsed from <contract> · date" vs "Edited manually".
- **Collaborators**: add, rename, re-role, remove.
- **Registration checklist** gating a "Submit for registration" action.

### Portfolio
- Project cards show works/contracts/member counts **plus attention counts**.
- Files & contracts tab shows **per-file usage** ("Linked to 2 works"),
  blocks deletion of linked files, and supports uploads.

### Zoe chat (`Zoe Chat Redesign.html`)
- Fixed-height comparison-context dropdown with three independently
  scrollable, individually searchable sections (Artists / Projects /
  Documents), cascading selection.

## Known limitations

- **AI parsing is simulated** — contract parsing returns deterministic mock
  splits; wiring a real model is a backend task.
- Spotify lookup, uploads, invites, and PDF export are mocked.
- No persistence; no authentication; permissions are sample data.
